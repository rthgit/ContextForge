from __future__ import annotations

import subprocess
import sys
from pathlib import Path


sys.path.insert(0, r"C:\tmp\trollsona_video_deps")

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "contextforge-demo.mp4"
FPS = 24
SIZE = (1280, 720)


SCENES = [
    (
        "01-space-loaded.png",
        7,
        "ContextForge turns a fuzzy builder brief into a build-ready agent blueprint.",
    ),
    (
        "02-fast-compile-filled.png",
        10,
        "Fast Compile keeps the first run simple: idea, audience, target, topology, risk, and cognitive modules.",
    ),
    (
        "03-full-control-open.png",
        10,
        "Full Control reveals deeper context, contracts, failure modes, and verification criteria when needed.",
    ),
    (
        "04-prompt-pack.png",
        15,
        "Seven isolated stages compile the brief into an executable prompt pack with roles, actions, QA, and recovery.",
    ),
    (
        "05-runtime-details.png",
        12,
        "Runtime Details reports every stage source, fallback reason, and duration without breaking the final output.",
    ),
    (
        "01-space-loaded.png",
        6,
        "Built for real builders using Codex and other AI coding agents. ContextForge: compiler, not generator.",
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


CAPTION_FONT = font(27, bold=True)
SMALL_FONT = font(18)


def fit_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(SIZE, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", SIZE, (8, 11, 17))
    x = (SIZE[0] - image.width) // 2
    y = (SIZE[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=CAPTION_FONT)
        if bbox[2] - bbox[0] <= max_width or not line:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def add_caption(image: Image.Image, caption: str) -> Image.Image:
    frame = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    pad = 24
    lines = wrap_text(draw, caption, SIZE[0] - (pad * 2))
    line_height = 34
    box_height = 30 + (line_height * len(lines)) + 24
    y0 = SIZE[1] - box_height
    draw.rectangle((0, y0, SIZE[0], SIZE[1]), fill=(7, 12, 20, 232))
    draw.line((0, y0, SIZE[0], y0), fill=(90, 213, 217, 230), width=3)
    y = y0 + 18
    for line in lines:
        draw.text((pad, y), line, fill=(233, 240, 247, 255), font=CAPTION_FONT)
        y += line_height
    draw.text(
        (SIZE[0] - 520, y0 + 10),
        "huggingface.co/spaces/build-small-hackathon/ContextForge",
        fill=(141, 190, 225, 235),
        font=SMALL_FONT,
    )
    return Image.alpha_composite(frame, overlay).convert("RGB")


def main() -> None:
    missing = [name for name, _, _ in SCENES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing screenshots: {', '.join(missing)}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{SIZE[0]}x{SIZE[1]}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUT),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for name, seconds, caption in SCENES:
        frame = add_caption(fit_image(ROOT / name), caption)
        raw = frame.tobytes()
        for _ in range(seconds * FPS):
            process.stdin.write(raw)
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr}")
    print(OUT)


if __name__ == "__main__":
    main()
