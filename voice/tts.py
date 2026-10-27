"""
AURA Text-To-Speech (TTS) Module (Milestone 7)
Provides non-blocking, asynchronous voice synthesis using pyttsx3
(Windows SAPI5 / Linux eSpeak / macOS) with queueing and console fallback.
"""

import queue
import logging
import threading
from typing import Optional, Callable, Any

try:
    import pyttsx3
    _HAS_PYTTSX3 = True
except ImportError:
    _HAS_PYTTSX3 = False

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Asynchronous Text-To-Speech worker thread.
    Prevents speech synthesis from blocking the real-time 30+ FPS vision loop.
    """

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 175,
        volume: float = 1.0,
        voice_index: int = 0,
        on_start: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str], None]] = None,
    ):
        self.enabled = enabled
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.voice_index = voice_index
        self.on_start = on_start
        self.on_done = on_done

        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._is_speaking = False
        self._engine: Optional[Any] = None
        self._lock = threading.Lock()

        # Start background TTS worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="AURA_TTS_Worker")
        self._worker_thread.start()

    @property
    def is_available(self) -> bool:
        return _HAS_PYTTSX3 and self.enabled

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    def _init_engine(self) -> Optional[Any]:
        """Initializes a local pyttsx3 engine instance for the thread."""
        if not _HAS_PYTTSX3 or not self.enabled:
            return None

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            voices = engine.getProperty("voices")
            if voices and 0 <= self.voice_index < len(voices):
                engine.setProperty("voice", voices[self.voice_index].id)
            return engine
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3 TTS engine: {e}. Falling back to console voice mode.")
            return None

    def _worker_loop(self) -> None:
        """Background thread loop that pulls text messages from the queue and speaks them."""
        # Initialize engine inside worker thread for COM threading compliance on Windows
        self._engine = self._init_engine()

        while self._running:
            try:
                text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:  # Sentinel value to exit
                break

            with self._lock:
                self._is_speaking = True

            if self.on_start:
                try:
                    self.on_start(text)
                except Exception as e:
                    logger.debug(f"on_start callback error: {e}")

            logger.info(f"[AURA Speaking] \"{text}\"")

            if self._engine is not None and self.enabled:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    logger.warning(f"TTS synthesis error: {e}")
            else:
                # Silent or console fallback
                pass

            with self._lock:
                self._is_speaking = False

            if self.on_done:
                try:
                    self.on_done(text)
                except Exception as e:
                    logger.debug(f"on_done callback error: {e}")

            self._queue.task_done()

    def speak(self, text: str, clear_existing: bool = False) -> None:
        """
        Enqueues text for asynchronous speech synthesis.
        
        Args:
            text: Sentence or response to speak.
            clear_existing: If True, purges pending messages before adding new text.
        """
        if not text or not text.strip():
            return

        clean_text = text.strip()

        if clear_existing:
            self.stop()

        self._queue.put(clean_text)

    def stop(self) -> None:
        """Flushes the pending speech queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Terminates the background worker thread gracefully."""
        self._running = False
        self._queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
