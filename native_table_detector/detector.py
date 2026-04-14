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
                # Tính Oriented Bounding Box cho cụm có padding 10px để lấy trọn viền đồ họa mảnh
                obb = self._compute_obb(page, spans, angle, padding=30)
                
                # Render và cắt crop kéo thẳng
                patch_before, patch_after = self._extract_cells_from_rotated(page, obb, angle)
                
                results["rotated_tables"].append({
                    "obb": obb,
                    "angle": angle,
                    "spans": spans,
                    "patch_image": patch_after,
                    "patch_before_rotate": patch_before,
                })
        return results

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

    def _compute_obb(self, page, spans, angle_deg, padding=10):
        angle_rad = math.radians(-angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        all_pts = []
        for s in spans:
            x0, y0, x1, y1 = s["bbox"]
            all_pts += [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]

        if spans:
            span_x0 = min(s["bbox"][0] for s in spans)
            span_y0 = min(s["bbox"][1] for s in spans)
            span_x1 = max(s["bbox"][2] for s in spans)
            span_y1 = max(s["bbox"][3] for s in spans)
            span_rect = fitz.Rect(span_x0, span_y0, span_x1, span_y1)
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
            
        rot = [(x*cos_a - y*sin_a, x*sin_a + y*cos_a) for x, y in all_pts]
        min_u = min(p[0] for p in rot)
        max_u = max(p[0] for p in rot)
        min_v = min(p[1] for p in rot)
        max_v = max(p[1] for p in rot)
        
        # Thêm padding để lấy trọn lề đồ họa kẻ bảng (Khắc phục nhược điểm C & D)
        content_span = max(max_u - min_u, max_v - min_v)
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

            corners_dst = np.float32([
                [0, 0], [out_w, 0], [out_w, out_h], [0, out_h]
            ])

            matrix = cv2.getPerspectiveTransform(corners_src, corners_dst)
            return cv2.warpPerspective(img, matrix, (int(out_w), int(out_h)), borderValue=(255, 255, 255))

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
            self._orientation_predictor = crop_orientation_predictor(pretrained=True)
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
