from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path


class RunningHubEnvCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_check": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = "RunningHub/Test"

    def check(self, run_check: bool):
        if not run_check:
            return ("check skipped",)

        lines = []
        lines.append("RunningHub custom node environment check")
        lines.append("")

        for package_name in ("cv2", "PIL", "numpy"):
            spec = importlib.util.find_spec(package_name)
            lines.append(f"{package_name}: {'OK' if spec else 'MISSING'}")

        ffmpeg_path = shutil.which("ffmpeg")
        lines.append(f"ffmpeg in PATH: {ffmpeg_path or 'MISSING'}")
        if ffmpeg_path:
            try:
                result = subprocess.run(
                    [ffmpeg_path, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                first_line = (result.stdout or result.stderr).splitlines()[0]
                lines.append(f"ffmpeg version: {first_line}")
            except Exception as exc:
                lines.append(f"ffmpeg version check failed: {exc}")

        try:
            with tempfile.TemporaryDirectory(prefix="rh_node_check_") as temp_dir:
                test_path = Path(temp_dir) / "write_test.txt"
                test_path.write_text("ok", encoding="utf-8")
                lines.append(f"temp write: OK ({test_path.read_text(encoding='utf-8')})")
        except Exception as exc:
            lines.append(f"temp write: FAILED ({exc})")

        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont

            image = Image.new("RGB", (320, 180), "white")
            draw = ImageDraw.Draw(image)
            draw.text((20, 70), "RunningHub OK", fill="black", font=ImageFont.load_default())
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            lines.append(f"opencv/pillow render: OK ({frame.shape[1]}x{frame.shape[0]})")
        except Exception as exc:
            lines.append(f"opencv/pillow render: FAILED ({exc})")

        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "RunningHubEnvCheck": RunningHubEnvCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubEnvCheck": "RunningHub Env Check",
}
