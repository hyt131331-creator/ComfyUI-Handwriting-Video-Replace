from __future__ import annotations

import tempfile
import traceback
from pathlib import Path


NODE_DIR = Path(__file__).resolve().parent
TEMPLATE_VIDEO = NODE_DIR / "templates" / "fixed_template.mp4"


class HandwritingVideoReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "line_one": ("STRING", {"default": "嘻嘻嘻", "multiline": False}),
                "line_two": ("STRING", {"default": "20060804", "multiline": False}),
                "font_size": ("INT", {"default": 36, "min": 12, "max": 160, "step": 1}),
                "font_color": ("STRING", {"default": "#000000", "multiline": False}),
                "max_seconds": ("FLOAT", {"default": 23.0, "min": 1.0, "max": 300.0, "step": 0.001}),
                "text_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.001}),
                "fade_seconds": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.001}),
                "output_fps": ("FLOAT", {"default": 60.0, "min": 8.0, "max": 60.0, "step": 1.0}),
                "text_position": (["bottom-center", "center", "top-center"], {"default": "bottom-center"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate"
    CATEGORY = "Video/Handwriting"
    OUTPUT_NODE = True

    def generate(
        self,
        line_one: str,
        line_two: str,
        font_size: int,
        font_color: str,
        max_seconds: float,
        text_seconds: float,
        fade_seconds: float,
        output_fps: float,
        text_position: str,
    ):
        try:
            if not TEMPLATE_VIDEO.is_file():
                raise FileNotFoundError(f"Template video not found: {TEMPLATE_VIDEO}")

            from .renderer import RenderOptions, render_video

            output_dir = Path(tempfile.gettempdir()) / "comfyui_handwriting_video_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            options = RenderOptions(
                line_one=line_one,
                line_two=line_two,
                text_position=text_position,
                font_size=int(font_size),
                font_color=font_color or "#000000",
                tracking_mode="paper_track",
                max_seconds=float(max_seconds),
                text_seconds=float(text_seconds),
                fade_seconds=float(fade_seconds),
                output_fps=float(output_fps),
            )
            output_path = render_video(TEMPLATE_VIDEO, output_dir, options)
            return (str(output_path),)
        except Exception as exc:
            detail = traceback.format_exc()
            raise RuntimeError(f"HandwritingVideoReplace failed: {exc}\n{detail}") from exc


class HandwritingVideoEnvCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"run_check": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "Video/Handwriting"

    def check(self, run_check: bool):
        if not run_check:
            return ("check skipped",)

        lines = []
        for package_name in ("cv2", "PIL", "numpy", "imageio_ffmpeg"):
            try:
                __import__(package_name)
                lines.append(f"{package_name}: OK")
            except Exception as exc:
                lines.append(f"{package_name}: FAILED ({exc})")

        try:
            from .renderer import _mux_audio_if_possible  # noqa: F401

            lines.append("renderer import: OK")
        except Exception as exc:
            lines.append(f"renderer import: FAILED ({exc})")

        lines.append(f"template video: {'OK' if TEMPLATE_VIDEO.is_file() else 'MISSING'}")
        lines.append(f"template path: {TEMPLATE_VIDEO}")
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "HandwritingVideoReplace": HandwritingVideoReplace,
    "HandwritingVideoEnvCheck": HandwritingVideoEnvCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HandwritingVideoReplace": "Handwriting Video Replace",
    "HandwritingVideoEnvCheck": "Handwriting Video Env Check",
}
