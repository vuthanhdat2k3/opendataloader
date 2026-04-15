import fitz
import math
import numpy as np
import cv2

try:
    from doctr.models import crop_orientation_predictor
except Exception:
    crop_orientation_predictor = None


class NativePDFTableDetector:
    def __init__(self, angle_threshold=2.0, spatial_dist_threshold=100):
        # Bỏ qua rotation < 2° (noise)
        self.angle_threshold = angle_threshold
        # Khoảng cách không gian (pixels) để gom các text nghiêng thành 1 bảng
        self.spatial_dist_threshold = spatial_dist_threshold
        self._orientation_predictor = None
        self._orientation_predictor_failed = False

    def process_page(self, page):
        results = {
            "page_rotation": page.rotation,
            "straight_tables": [],
            "rotated_tables": [],
        }

        groups = self._group_spans_by_angle_and_space(page)

        for group_id, group_data in groups.items():
            angle = group_data["angle"]
            spans = group_data["spans"]
            
            if abs(angle) < self.angle_threshold:
                # Bỏ qua cụm text thẳng ở hàm này
                continue
            else:
                # Dual OBB:
                # - loose: robust OCR crop (includes nearby table lines/graphics)
                # - tight: minimal replacement/redaction box
                obb_loose = self._compute_obb(
                    page,
                    spans,
                    angle,
                    include_drawings=True,
                    padding_mode="loose",
                    padding=30,
                )
                obb_tight = self._compute_obb(
                    page,
                    spans,
                    angle,
                    # Keep vector border lines for tight box too, but with tight margins.
                    include_drawings=True,
                    padding_mode="tight",
                    padding=0,
                )
                
                # Render và cắt crop kéo thẳng
                patch_before, patch_after = self._extract_cells_from_rotated(
                    page, obb_loose, angle
                )
                
                results["rotated_tables"].append({
                    # Backward-compatible alias: keep obb as OCR crop box.
                    "obb": obb_loose,
                    "obb_loose": obb_loose,
                    "obb_tight": obb_tight,
                    "angle": angle,
                    "spans": spans,
                    "patch_image": patch_after,
                    "patch_before_rotate": patch_before,
                })
        results["rotated_tables"] = self._deduplicate_rotated_tables(
            results["rotated_tables"]
        )
        return results

    def _deduplicate_rotated_tables(self, tables, iou_threshold=0.6):
        """Remove duplicate detections for the same physical table."""
        if not tables:
            return []

        def to_aabb(obb):
            cx, cy, w, h = obb["cx"], obb["cy"], obb["w"], obb["h"]
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

        def iou(box_a, box_b):
            x0 = max(box_a[0], box_b[0])
            y0 = max(box_a[1], box_b[1])
            x1 = min(box_a[2], box_b[2])
            y1 = min(box_a[3], box_b[3])
            if x1 <= x0 or y1 <= y0:
                return 0.0
            inter = (x1 - x0) * (y1 - y0)
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        sorted_tables = sorted(
            tables,
            key=lambda t: (
                len(t.get("spans", [])),
                t.get("obb", {}).get("w", 0.0) * t.get("obb", {}).get("h", 0.0),
            ),
            reverse=True,
        )

        kept = []
        kept_boxes = []
        for table in sorted_tables:
            obb = table.get("obb")
            if not obb:
                continue
            box = to_aabb(obb)
            if any(iou(box, existing) >= iou_threshold for existing in kept_boxes):
                continue
            kept.append(table)
            kept_boxes.append(box)
        return kept

    def _group_spans_by_angle_and_space(self, page):
        blocks = page.get_text("rawdict", flags=0)["blocks"]

        def bbox_distance(b1, b2):
            x0a, y0a, x1a, y1a = b1
            x0b, y0b, x1b, y1b = b2
            dx = max(x0a - x1b, x0b - x1a, 0.0)
            dy = max(y0a - y1b, y0b - y1a, 0.0)
            return math.hypot(dx, dy)
        
        rotated_spans = []
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                dx, dy = line["dir"]
                angle = round(-math.degrees(math.atan2(dy, dx)) / 5) * 5
                
                if abs(angle) < self.angle_threshold:
                    continue
                    
                for span in line["spans"]:
                    text = span.get("text")
                    if text is None:
                        text = "".join(ch.get("c", "") for ch in span.get("chars", []))

                    if text.strip():
                        rotated_spans.append({
                            "text": text,
                            "bbox": span["bbox"],
                            "angle": angle,
                            "origin": span["origin"],
                        })
        
        # Gom nhóm theo góc VÀ khoảng cách (vượt qua nhược điểm A)
        groups = {}
        group_counter = 0
        
        for span in rotated_spans:
            assigned_group = -1
            for gid, gdata in groups.items():
                if abs(gdata["angle"] - span["angle"]) <= 5:
                    for gspan in gdata["spans"]:
                        dist = bbox_distance(span["bbox"], gspan["bbox"])
                        if dist < self.spatial_dist_threshold:
                            assigned_group = gid
                            break
                if assigned_group != -1:
                    break
            
            if assigned_group == -1:
                assigned_group = group_counter
                groups[assigned_group] = {"angle": span["angle"], "spans": []}
                group_counter += 1
                
            groups[assigned_group]["spans"].append(span)
            
        return groups

    def _compute_obb(
        self,
        page,
        spans,
        angle_deg,
        padding=10,
        include_drawings=True,
        padding_mode="loose",
    ):
        angle_rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        all_pts = []
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            all_pts += [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]

        if spans and include_drawings:
            span_x0 = min(s["bbox"][0] for s in spans)
            span_y0 = min(s["bbox"][1] for s in spans)
            span_x1 = max(s["bbox"][2] for s in spans)
            span_y1 = max(s["bbox"][3] for s in spans)
            span_rect = fitz.Rect(span_x0, span_y0, span_x1, span_y1)
            if padding_mode == "tight":
                span_w = max(1.0, span_rect.width)
                span_h = max(1.0, span_rect.height)
                # Small margin just to include outer border strokes.
                draw_margin = min(8.0, max(2.0, max(span_w, span_h) * 0.02))
            else:
                draw_margin = max(float(padding), 20.0)
            expanded_span_rect = fitz.Rect(
                max(0.0, span_rect.x0 - draw_margin),
                max(0.0, span_rect.y0 - draw_margin),
                min(page.rect.width, span_rect.x1 + draw_margin),
                min(page.rect.height, span_rect.y1 + draw_margin),
            )

            for drawing in page.get_drawings():
                rect = drawing.get("rect")
                if rect is None:
                    continue
                draw_rect = fitz.Rect(rect)
                if not draw_rect.intersects(expanded_span_rect):
                    continue

                all_pts += [
                    (draw_rect.x0, draw_rect.y0),
                    (draw_rect.x1, draw_rect.y0),
                    (draw_rect.x1, draw_rect.y1),
                    (draw_rect.x0, draw_rect.y1),
                ]

        if not all_pts:
            return {"cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0, "angle": angle_deg}
            
        rot = [(x*cos_a - y*sin_a, x*sin_a + y*cos_a) for x, y in all_pts]
        min_u = min(p[0] for p in rot)
        max_u = max(p[0] for p in rot)
        min_v = min(p[1] for p in rot)
        max_v = max(p[1] for p in rot)

        content_w = max_u - min_u
        content_h = max_v - min_v
        content_span = max(content_w, content_h)
        if padding_mode == "tight":
            # Keep box close to table content.
            effective_padding = min(8.0, max(2.0, content_span * 0.025, float(padding)))
        else:
            # Loose mode keeps current behavior for OCR robustness.
            effective_padding = max(float(padding), content_span * 0.18)
        min_u -= effective_padding
        max_u += effective_padding
        min_v -= effective_padding
        max_v += effective_padding
        
        cu, cv = (min_u+max_u)/2, (min_v+max_v)/2
        ci = math.cos(-angle_rad)
        si = math.sin(-angle_rad)
        
        return {
            "cx": cu*ci - cv*si, "cy": cu*si + cv*ci,
            "w": max_u-min_u, "h": max_v-min_v,
            "angle": angle_deg
        }

    def _extract_cells_from_rotated(self, page, obb, angle_deg):
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
        cx = obb["cx"] * 2
        cy = obb["cy"] * 2
        w  = obb["w"]  * 2
        h  = obb["h"]  * 2

        def warp_with_ccw_angle(ccw_angle_deg):
            # Khi xoay ảnh, để box chứa được toàn bộ table thì ta phải tính toán lại kích thước
            # của box kết quả theo độ lệch (góc xoay cuối cùng so với bảng gốc)
            final_skew_deg = angle_deg + ccw_angle_deg
            final_skew_rad = math.radians(final_skew_deg)
            abs_cos = abs(math.cos(final_skew_rad))
            abs_sin = abs(math.sin(final_skew_rad))
            
            # Kích thước box bao lấy toàn bộ table tại góc quay final_skew
            out_w = w * abs_cos + h * abs_sin
            out_h = w * abs_sin + h * abs_cos
            
            angle_rad = math.radians(ccw_angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            corners_src = np.float32([
                [cx - out_w/2*cos_a + out_h/2*sin_a, cy - out_w/2*sin_a - out_h/2*cos_a],
                [cx + out_w/2*cos_a + out_h/2*sin_a, cy + out_w/2*sin_a - out_h/2*cos_a],
                [cx + out_w/2*cos_a - out_h/2*sin_a, cy + out_w/2*sin_a + out_h/2*cos_a],
                [cx - out_w/2*cos_a - out_h/2*sin_a, cy - out_w/2*sin_a + out_h/2*cos_a],
            ])

            dst_w = max(2, int(math.ceil(out_w)) + 2)
            dst_h = max(2, int(math.ceil(out_h)) + 2)
            corners_dst = np.float32(
                [[1, 1], [dst_w - 2, 1], [dst_w - 2, dst_h - 2], [1, dst_h - 2]]
            )

            matrix = cv2.getPerspectiveTransform(corners_src, corners_dst)
            return cv2.warpPerspective(
                img,
                matrix,
                (dst_w, dst_h),
                borderValue=(255, 255, 255),
            )

        # Ảnh crop trước deskew để phục vụ debug/so sánh.
        before_deskew = warp_with_ccw_angle(0.0)

        # Quy ước deskew theo dấu góc detect:
        # -20° -> quay ngược chiều kim đồng hồ 20°
        # +20° -> quay theo chiều kim đồng hồ 20°
        # Với quy ước hàm này: góc dương là CCW, nên dùng góc ngược dấu.
        detected_skew_deg = float(angle_deg)
        deskew_ccw_deg = -detected_skew_deg
        deskewed = warp_with_ccw_angle(deskew_ccw_deg)
        return before_deskew, deskewed

    @staticmethod
    def _tight_crop_white_border(
        patch_rgb: np.ndarray,
        *,
        bg_threshold: int = 250,
        pad: int = 2,
        min_size: int = 8,
    ) -> np.ndarray:
        """
        Crop away near-white border introduced by perspective warp/rotation.
        Keeps a small padding to avoid cutting table strokes.
        """
        if patch_rgb is None or not isinstance(patch_rgb, np.ndarray):
            return patch_rgb
        if patch_rgb.ndim != 3 or patch_rgb.shape[2] < 3:
            return patch_rgb

        h, w = int(patch_rgb.shape[0]), int(patch_rgb.shape[1])
        if h < min_size or w < min_size:
            return patch_rgb

        content = np.any(patch_rgb[:, :, :3] < bg_threshold, axis=2)
        if not np.any(content):
            return patch_rgb

        ys, xs = np.where(content)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h - 1, y1 + pad)
        x1 = min(w - 1, x1 + pad)

        cropped = patch_rgb[y0 : y1 + 1, x0 : x1 + 1]
        if cropped.shape[0] < min_size or cropped.shape[1] < min_size:
            return patch_rgb
        return cropped

    def _rotate_right_angle(self, image_rgb, clockwise_angle):
        angle = int(clockwise_angle) % 360
        if angle == 90:
            return cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image_rgb, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image_rgb

    def _get_orientation_predictor(self):
        if self._orientation_predictor_failed:
            return None
        if self._orientation_predictor is not None:
            return self._orientation_predictor
        if crop_orientation_predictor is None:
            self._orientation_predictor_failed = True
            return None

        try:
            predictor = crop_orientation_predictor(pretrained=True)
            try:
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                predictor = predictor.to(device).eval()
            except Exception:
                pass
            self._orientation_predictor = predictor
        except Exception:
            self._orientation_predictor_failed = True
            self._orientation_predictor = None

        return self._orientation_predictor

    def _normalize_patch_upright(self, patch_rgb):
        predictor = self._get_orientation_predictor()
        if predictor is None:
            # Fallback nhẹ: ưu tiên bảng nằm ngang để giảm trường hợp dọc 90°.
            if patch_rgb.shape[0] > patch_rgb.shape[1]:
                return self._rotate_right_angle(patch_rgb, 90)
            return patch_rgb

        try:
            candidates = [
                (0, patch_rgb),
                (90, self._rotate_right_angle(patch_rgb, 90)),
                (180, self._rotate_right_angle(patch_rgb, 180)),
                (270, self._rotate_right_angle(patch_rgb, 270)),
            ]

            best_score = float("inf")
            best_patch = patch_rgb

            for _, cand in candidates:
                _, angles, confidences = predictor([cand])
                pred_angle = float(angles[0]) if len(angles) else 0.0
                conf = float(confidences[0]) if len(confidences) else 0.0
                quantized = int(round(pred_angle / 90.0) * 90) % 360

                # Ưu tiên patch mà model nhận là 0° với confidence cao.
                mismatch = 0.0 if quantized == 0 else 1.0
                score = mismatch + (1.0 - conf) * 0.1
                if score < best_score:
                    best_score = score
                    best_patch = cand

            return best_patch
        except Exception:
            if patch_rgb.shape[0] > patch_rgb.shape[1]:
                return self._rotate_right_angle(patch_rgb, 90)
            return patch_rgb
