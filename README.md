# ComfyUI Handwriting Video Replace

Custom ComfyUI node for RunningHub.

It generates an MP4 video from a fixed template video and overlays handwriting-style text on the red paper area.

## Node Names

Main node:

```text
HandwritingVideoReplace
```

Display name:

```text
Handwriting Video Replace
```

Environment check node:

```text
HandwritingVideoEnvCheck
```

Display name:

```text
Handwriting Video Env Check
```

Category:

```text
Video/Handwriting
```

## Files

```text
__init__.py
nodes.py
renderer.py
requirements.txt
assets/aaa.TTF
templates/fixed_template.mp4
```

## Dependencies

```text
opencv-python
Pillow
numpy
imageio-ffmpeg
```

The renderer uses OpenCV for frame processing, Pillow for text rendering, and FFmpeg for final MP4/audio muxing.

## Test Steps On RunningHub

1. Install this custom node repository.
2. Search for `Handwriting Video Env Check`.
3. Run the environment check node first.
4. Confirm the report shows the required packages and template video are available.
5. Search for `Handwriting Video Replace`.
6. Run the main node with default parameters.

The main node returns a local MP4 output path as a string.

## Inputs

- `line_one`: first text line
- `line_two`: second text line
- `font_size`: default `36`
- `font_color`: default `#000000`
- `max_seconds`: default `23`
- `text_seconds`: default `0`
- `fade_seconds`: default `0.12`
- `output_fps`: default `60`
- `text_position`: `bottom-center`, `center`, or `top-center`

## Notes

This node uses `templates/fixed_template.mp4` as the fixed source material.
No video upload input is required.

