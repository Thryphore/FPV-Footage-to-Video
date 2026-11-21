"""Resolve and apply the desktop app icon (window + Windows taskbar)."""

from __future__ import annotations

import sys
from pathlib import Path


def icon_paths() -> tuple[Path, Path]:
    """Return ``(ico_path, png_path)`` next to packaged assets."""
    root = Path(__file__).resolve().parent / "assets"
    return root / "app_icon.ico", root / "app_icon.png"


def set_windows_app_id(app_id: str = "FPVFootageToVideo.Desktop") -> None:
    """Keep the taskbar entry under our app id instead of pythonw.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:  # noqa: BLE001
        pass


def apply_window_icon(window) -> None:
    """Set title-bar / taskbar icon on a Tk or CustomTkinter window."""
    ico, png = icon_paths()
    if sys.platform == "win32" and ico.is_file():
        try:
            window.iconbitmap(default=str(ico))
        except Exception:  # noqa: BLE001
            try:
                window.iconbitmap(str(ico))
            except Exception:  # noqa: BLE001
                pass
    if png.is_file():
        try:
            from PIL import Image, ImageTk

            # Keep a reference on the window so Tk does not GC the photo.
            img = Image.open(png).convert("RGBA")
            photo = ImageTk.PhotoImage(img)
            window._app_icon_photo = photo  # noqa: SLF001
            window.iconphoto(True, photo)
        except Exception:  # noqa: BLE001
            pass
