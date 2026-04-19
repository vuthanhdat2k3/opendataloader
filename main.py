import os

from ui.app import RUNS_DIR_ABS, demo


def main() -> None:
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        share=os.getenv("GRADIO_SHARE", "0") == "1",
        allowed_paths=[str(RUNS_DIR_ABS)],
    )


if __name__ == "__main__":
    main()
