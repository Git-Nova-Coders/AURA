"""
AURA Holographic Desktop Splash Screen
Lightweight, standalone floating cybernetic GUI that displays immediately
upon running AURA, rendering an animated Arc Reactor, live boot logs, and a synapse progress bar.
"""

import sys
import math
import time
import threading
from typing import Optional, List

try:
    import tkinter as tk
except ImportError:
    tk = None


class HolographicSplash:
    """
    Floating holographic desktop splash screen matching AURA's cybernetic HUD aesthetic.
    Runs asynchronously in a dedicated thread with zero heavy dependencies.
    """

    def __init__(
        self,
        title: str = "A.U.R.A.  V1.0",
        subtitle: str = "UNIVERSAL COGNITIVE OS & CORE ACCELERATOR",
        width: int = 680,
        height: int = 400,
    ):
        self.title_text = title
        self.subtitle_text = subtitle
        self.width = width
        self.height = height

        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self._progress = 0.05
        self._status_text = "INITIALIZING CORE SYNAPSE MATRIX..."
        self._angle = 0.0

        # Boot log lines
        self.boot_logs: List[str] = [
            "INITIALIZING MULTIMODAL CONTEXT STREAM... ACTIVE",
            "SYNCING LOCAL MEMORY VECTOR DATABASE... COMPILED",
            "RESOLVING HARDWARE ACCELERATOR (RTX 3050 Ti)... NOMINAL",
            "ROUTING AUDIO PRE-PROCESSING FILTERS... ACTIVE",
            "CALIBRATING 21-LANDMARK 3D TRACKING SENSORS... SYNCHRONIZED",
            "CONNECTING KNOWLEDGE RETRIEVER & RAG STACK... SECURED",
            "OPTIMIZING CORE THREAD POOL PIPELINES... STABLE",
        ]
        self._visible_log_count = 1
        self._last_log_time = time.time()

    def start(self) -> "HolographicSplash":
        """Launches the splash screen in a background thread."""
        if tk is None:
            return self

        self.running = True
        self.thread = threading.Thread(target=self._run_gui, daemon=True, name="AURA_Splash")
        self.thread.start()
        # Allow GUI window to initialize
        time.sleep(0.08)
        return self

    def _run_gui(self) -> None:
        try:
            self.root = tk.Tk()
            self.root.title("AURA Boot Loader")
            self.root.overrideredirect(True)  # Frameless floating window
            self.root.attributes("-topmost", True)  # Float above all windows
            
            # Try setting transparency
            try:
                self.root.attributes("-alpha", 0.94)
            except Exception:
                pass

            # Center on screen
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            x = (sw - self.width) // 2
            y = (sh - self.height) // 2
            self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")
            self.root.configure(bg="#050a14")

            self.canvas = tk.Canvas(
                self.root,
                width=self.width,
                height=self.height,
                bg="#050a14",
                highlightthickness=0,
            )
            self.canvas.pack(fill="both", expand=True)

            # Start animation loop
            self._animate()
            self.root.mainloop()
        except Exception:
            self.running = False

    def _animate(self) -> None:
        if not self.running or not self.root or not self.canvas:
            return

        try:
            self.canvas.delete("all")
            self._draw_splash()
            self._angle += 0.05
            
            # Stepwise log disclosure
            now = time.time()
            if now - self._last_log_time > 0.45 and self._visible_log_count < len(self.boot_logs):
                self._visible_log_count += 1
                self._last_log_time = now

            self.root.after(30, self._animate)
        except Exception:
            pass

    def _draw_splash(self) -> None:
        w, h = self.width, self.height

        # 1. Subtle Cyber Grid Background
        for gx in range(0, w, 32):
            self.canvas.create_line(gx, 0, gx, h, fill="#081426", width=1)
        for gy in range(0, h, 32):
            self.canvas.create_line(0, gy, w, gy, fill="#081426", width=1)

        # 2. Glowing Neon Border
        self.canvas.create_rectangle(
            2, 2, w - 2, h - 2,
            outline="#00f0ff", width=2
        )
        self.canvas.create_rectangle(
            6, 6, w - 6, h - 6,
            outline="#005577", width=1
        )

        # 3. Corner HUD Brackets
        bk_len = 24
        # Top-Left
        self.canvas.create_line(12, 12, 12 + bk_len, 12, fill="#00f0ff", width=3)
        self.canvas.create_line(12, 12, 12, 12 + bk_len, fill="#00f0ff", width=3)
        # Top-Right
        self.canvas.create_line(w - 12 - bk_len, 12, w - 12, 12, fill="#00f0ff", width=3)
        self.canvas.create_line(w - 12, 12, w - 12, 12 + bk_len, fill="#00f0ff", width=3)
        # Bottom-Left
        self.canvas.create_line(12, h - 12, 12 + bk_len, h - 12, fill="#00f0ff", width=3)
        self.canvas.create_line(12, h - 12 - bk_len, 12, h - 12, fill="#00f0ff", width=3)
        # Bottom-Right
        self.canvas.create_line(w - 12 - bk_len, h - 12, w - 12, h - 12, fill="#00f0ff", width=3)
        self.canvas.create_line(w - 12, h - 12 - bk_len, w - 12, h - 12, fill="#00f0ff", width=3)

        # 4. Header: A.U.R.A. V1.0
        self.canvas.create_text(
            28, 38,
            text=self.title_text,
            anchor="w",
            fill="#00f0ff",
            font=("Impact", 24, "bold") if sys.platform == "win32" else ("Helvetica", 24, "bold"),
        )
        self.canvas.create_text(
            30, 64,
            text=self.subtitle_text,
            anchor="w",
            fill="#00ff9d",
            font=("Consolas", 9, "bold") if sys.platform == "win32" else ("Courier", 9, "bold"),
        )

        # 5. Left Panel: Live System Boot Sequence Logs with Tree Lines
        log_y = 100
        for i in range(min(self._visible_log_count, len(self.boot_logs))):
            log = self.boot_logs[i]
            prefix = "└── " if i == len(self.boot_logs) - 1 else "├── "
            
            # Tree line
            self.canvas.create_text(
                28, log_y,
                text=prefix + log,
                anchor="w",
                fill="#8ec5ff",
                font=("Consolas", 8) if sys.platform == "win32" else ("Courier", 8),
            )
            log_y += 24

        # 6. Right Panel: Animated Arc Reactor / Holographic Iris
        cx = w - 150
        cy = 190

        # Outer Ticked Ring
        self.canvas.create_oval(
            cx - 85, cy - 85, cx + 85, cy + 85,
            outline="#007799", width=1
        )
        # Rotating dashed middle ring
        dash_count = 12
        for d in range(dash_count):
            a1 = self._angle + (d * 2 * math.pi / dash_count)
            a2 = a1 + (math.pi / dash_count * 0.7)
            x1 = cx + math.cos(a1) * 72
            y1 = cy + math.sin(a1) * 72
            x2 = cx + math.cos(a2) * 72
            y2 = cy + math.sin(a2) * 72
            self.canvas.create_line(x1, y1, x2, y2, fill="#00f0ff", width=2)

        # Counter-rotating inner ring
        dash_count_inner = 8
        for d in range(dash_count_inner):
            a1 = -self._angle * 1.5 + (d * 2 * math.pi / dash_count_inner)
            a2 = a1 + (math.pi / dash_count_inner * 0.6)
            x1 = cx + math.cos(a1) * 48
            y1 = cy + math.sin(a1) * 48
            x2 = cx + math.cos(a2) * 48
            y2 = cy + math.sin(a2) * 48
            self.canvas.create_line(x1, y1, x2, y2, fill="#00ff9d", width=2.5)

        # Inner Glowing Core
        self.canvas.create_oval(
            cx - 24, cy - 24, cx + 24, cy + 24,
            fill="#005577", outline="#00f0ff", width=2
        )
        self.canvas.create_oval(
            cx - 12, cy - 12, cx + 12, cy + 12,
            fill="#ffffff", outline="#00ff9d", width=1
        )

        # Orbiting Energy Particle
        ox = cx + math.cos(self._angle * 2.2) * 86
        oy = cy + math.sin(self._angle * 2.2) * 86
        self.canvas.create_oval(ox - 3, oy - 3, ox + 3, oy + 3, fill="#00f0ff", outline="#ffffff", width=1)

        # 7. Bottom: Glowing Progress Bar & Status Text
        bar_x = 28
        bar_y = h - 50
        bar_w = w - 56
        bar_h = 10

        # Status Text
        self.canvas.create_text(
            bar_x, bar_y - 12,
            text=self._status_text,
            anchor="w",
            fill="#00f0ff",
            font=("Consolas", 8, "bold") if sys.platform == "win32" else ("Courier", 8, "bold"),
        )
        # Percent
        pct_text = f"{int(self._progress * 100)}%"
        self.canvas.create_text(
            bar_x + bar_w, bar_y - 12,
            text=pct_text,
            anchor="e",
            fill="#00ff9d",
            font=("Consolas", 8, "bold") if sys.platform == "win32" else ("Courier", 8, "bold"),
        )

        # Bar background
        self.canvas.create_rectangle(
            bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
            fill="#061224", outline="#004466", width=1
        )
        # Bar fill
        fill_w = max(4, int(bar_w * min(1.0, self._progress)))
        self.canvas.create_rectangle(
            bar_x, bar_y, bar_x + fill_w, bar_y + bar_h,
            fill="#00f0ff", outline="#00ff9d", width=1
        )

    def update_step(self, progress: float, message: str) -> None:
        """Thread-safely updates the progress bar and status text."""
        self._progress = max(0.0, min(1.0, progress))
        self._status_text = message

    def close(self) -> None:
        """Fades out and closes the splash window."""
        self.running = False
        if self.root:
            try:
                def _destroy():
                    try:
                        self.root.destroy()
                        self.root.quit()
                    except Exception:
                        pass
                self.root.after(0, _destroy)
            except Exception:
                pass


def launch_splash() -> HolographicSplash:
    """Convenience helper to launch the holographic splash screen."""
    splash = HolographicSplash()
    splash.start()
    return splash
