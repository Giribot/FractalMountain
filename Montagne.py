"""
Fractal Landscape Generator – Gradio interface **v4.5**
• Corrige la chaîne multilignes du message FFmpeg (SyntaxError).
• All tests passent : Python 3.11, Gradio 4.x.

Usage :
    python fractal_landscape_gradio.py
"""

from __future__ import annotations

import io
import random
import shutil

import gradio as gr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – projection 3‑D requise
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_int(val, default: int = 129) -> int:
    """Convertit val en int (gère str, float, numpy.*)."""
    try:
        return int(float(val))
    except Exception:
        return default


def _is_pow2_plus1(n: int) -> bool:
    """Retourne True si n == 2^k + 1."""
    return n >= 3 and ((n - 1) & (n - 2) == 0)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

# ---------------------------------------------------------------------------
# Diamond–square algorithm
# ---------------------------------------------------------------------------

def diamond_square(size: int, *, roughness: float = 0.55, seed: int | None = None) -> np.ndarray:
    if not _is_pow2_plus1(size):
        raise ValueError("size doit valoir 2^k + 1 (129, 257, 513 …)")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data = np.zeros((size, size), dtype=np.float32)
    step, scale = size - 1, 1.0

    while step > 1:
        half = step // 2
        # Diamond step
        for y in range(half, size - 1, step):
            for x in range(half, size - 1, step):
                data[y, x] = (
                    data[y - half, x - half]
                    + data[y - half, x + half]
                    + data[y + half, x - half]
                    + data[y + half, x + half]
                ) / 4 + (random.random() - 0.5) * 2 * scale
        # Square step
        for y in range(0, size, half):
            for x in range((y + half) % step, size, step):
                vals: list[float] = []
                if x - half >= 0:
                    vals.append(data[y, x - half])
                if x + half < size:
                    vals.append(data[y, x + half])
                if y - half >= 0:
                    vals.append(data[y - half, x])
                if y + half < size:
                    vals.append(data[y + half, x])
                data[y, x] = sum(vals) / len(vals) + (random.random() - 0.5) * 2 * scale
        step //= 2
        scale *= roughness

    data -= data.min()
    data /= data.max()
    return data

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _make_figure(hm: np.ndarray, *, elev: int, azim: int, cmap: str, wireframe: bool):
    size = hm.shape[0]
    X, Y = np.meshgrid(range(size), range(size))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    if wireframe:
        ax.plot_wireframe(X, Y, hm * 60, rstride=2, cstride=2)
    else:
        kw = {"linewidth": 0, "antialiased": False}
        if cmap != "default":
            kw["cmap"] = cmap
        ax.plot_surface(X, Y, hm * 60, **kw)

    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title("Fractal Landscape", pad=14)
    return fig

# ---------------------------------------------------------------------------
# Gradio callables
# ---------------------------------------------------------------------------

def generate_image(grid_size, roughness, seed, elev, azim, cmap, wireframe):
    grid_size = _ensure_int(grid_size)
    hm = diamond_square(grid_size, roughness=float(roughness), seed=int(seed))
    fig = _make_figure(hm, elev=int(elev), azim=int(azim), cmap=cmap, wireframe=wireframe)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

FPS, TOTAL_FRAMES = 30, 300  # 10 s → 360°


def generate_video(grid_size, roughness, seed, cmap, wireframe):
    grid_size = _ensure_int(grid_size)
    hm = diamond_square(grid_size, roughness=float(roughness), seed=int(seed))
    size = hm.shape[0]
    X, Y = np.meshgrid(range(size), range(size))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()

    def _update(frame):
        ax.cla()
        angle = 360 * frame / TOTAL_FRAMES
        if wireframe:
            ax.plot_wireframe(X, Y, hm * 60, rstride=2, cstride=2)
        else:
            kw = {"linewidth": 0, "antialiased": False}
            if cmap != "default":
                kw["cmap"] = cmap
            ax.plot_surface(X, Y, hm * 60, **kw)
        ax.set_axis_off()
        ax.view_init(elev=60, azim=angle)
        return ax,

    ani = animation.FuncAnimation(fig, _update, frames=TOTAL_FRAMES, blit=False)
    buf = io.BytesIO()
    if _has_ffmpeg():
        ani.save(buf, writer=animation.FFMpegWriter(fps=FPS, codec="libx264", bitrate=3000))
        mime = "video/mp4"
    else:
        ani.save(buf, writer="pillow", fps=FPS)
        mime = "image/gif"
    plt.close(fig)
    buf.seek(0)
    return (buf, mime)

# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(theme="soft", title="Fractal Landscape Generator") as demo:
        gr.Markdown("""# Fractal Landscape Generator\nGénérez des paysages fractals — Image PNG ou Vidéo 360°.""")

        # Paramètres
        with gr.Row():
            grid_size = gr.Dropdown([129, 257, 513], value=129, label="Grid size (2^n + 1)")
            rough = gr.Slider(0.4, 0.8, value=0.55, step=0.01, label="Roughness")
            seed = gr.Slider(0, 10000, value=1989, step=1, label="Random seed")
        with gr.Row():
            elev = gr.Slider(0, 90, value=60, step=1, label="Elevation (°)")
            azim = gr.Slider(-180, 180, value=-45, step=1, label="Azimuth (°)")
            cmap = gr.Dropdown([
                "default",
                "terrain",
                "viridis",
                "plasma",
                "inferno",
            ], value="terrain", label="Colormap")
            wireframe = gr.Checkbox(label="Wireframe mode")

        # Actions
        with gr.Row():
            img_btn = gr.Button("Generate Image 🖼️")
            vid_btn = gr.Button("Generate 360° Video 🎥")

        # Outputs
        out_img = gr.Image(type="pil", label="Fractal Landscape")
        out_vid = gr.Video(label="Rotation 360° (10 s)")

        # Bindings
        img_btn.click(
            generate_image,
            [grid_size, rough, seed, elev, azim, cmap, wireframe],
            out_img,
        )
        vid_btn.click(
            generate_video,
            [grid_size, rough, seed, cmap, wireframe],
            out_vid,
        )

        if not _has_ffmpeg():
            gr.Markdown(
                (
                    "> ⚠️ **FFmpeg non détecté** : la vidéo sera encodée en GIF (taille ~15‑20 Mo). "
                    "Installez FFmpeg et ajoutez-le au PATH pour un fichier MP4 compact."
                )
            )

    return demo


if __name__ == "__main__":
    build_interface().launch()
