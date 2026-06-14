# RunningHub Custom Node Environment Check

Node name:

```text
RunningHubEnvCheck
```

Display name:

```text
RunningHub Env Check
```

Category:

```text
RunningHub/Test
```

This is a minimal ComfyUI custom node used to test whether the RunningHub
environment supports the dependencies needed by the handwriting video node.

It checks:

- `opencv-python` / `cv2`
- `Pillow`
- `numpy`
- `ffmpeg` command availability
- temporary file write permission
- simple OpenCV + Pillow image rendering

Expected output is a text report from the node.

If all checks pass, the full video node can be packaged with:

- `renderer.py`
- fixed template video
- handwriting font file
- node wrapper

