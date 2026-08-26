"""
AURA Voice Assistant Orchestrator (Milestone 7)
Coordinates Speech-To-Text audio capture, Conversational reasoning,
and Text-To-Speech audio output with push-to-talk triggers and UI callbacks.
"""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any

from brain.conversation import ConversationEngine, ConversationResponse
from .tts import TextToSpeech
from .stt import SpeechToText

logger = logging.getLogger(__name__)


class VoiceAssistant:
    """
    High-level voice interface manager for AURA.
    Enables interactive voice conversation with live visual perception grounding.
    """

    def __init__(
        self,
        conversation_engine: Optional[ConversationEngine] = None,
        tts: Optional[TextToSpeech] = None,
        stt: Optional[SpeechToText] = None,
        enable_tts: bool = True,
        enable_stt: bool = True,
        voice_rate: int = 175,
        voice_volume: float = 1.0,
        on_status_change: Optional[Callable[[str], None]] = None,
        on_query: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[ConversationResponse], None]] = None,
    ):
        self.conversation_engine = conversation_engine or ConversationEngine()
        self.tts = tts or TextToSpeech(enabled=enable_tts, rate=voice_rate, volume=voice_volume)
        self.stt = stt or SpeechToText(enabled=enable_stt)

        self.on_status_change = on_status_change
        self.on_query = on_query
        self.on_response = on_response

        self._status = "IDLE"  # "IDLE", "LISTENING", "THINKING", "SPEAKING"
        self._lock = threading.Lock()
        self._active_thread: Optional[threading.Thread] = None

        # Wire TTS event hooks
        self.tts.on_start = self._on_tts_start
        self.tts.on_done = self._on_tts_done

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def _set_status(self, new_status: str) -> None:
        with self._lock:
            self._status = new_status
        if self.on_status_change:
            try:
                self.on_status_change(new_status)
            except Exception as e:
                logger.debug(f"Status change callback error: {e}")

    def _on_tts_start(self, text: str) -> None:
        self._set_status("SPEAKING")

    def _on_tts_done(self, text: str) -> None:
        self._set_status("IDLE")

    def process_text_query(self, query_text: str, speak_output: bool = True) -> ConversationResponse:
        """
        Processes a textual user query synchronously through the conversation engine.
        Optionally enqueues the resulting response for speech synthesis.
        """
        self._set_status("THINKING")
        if self.on_query:
            try:
                self.on_query(query_text)
            except Exception as e:
                logger.debug(f"Query callback error: {e}")

        # Run conversation engine
        response = self.conversation_engine.respond(query_text)

        if self.on_response:
            try:
                self.on_response(response)
            except Exception as e:
                logger.debug(f"Response callback error: {e}")

        # Enqueue speech
        if speak_output and self.tts.is_available:
            self.tts.speak(response.response_text)
        else:
            self._set_status("IDLE")

        return response

    def trigger_push_to_talk(self, timeout: float = 5.0, speak_output: bool = True) -> None:
        """
        Triggers push-to-talk voice capture asynchronously in a worker thread.
        Does not block the caller (or video loop).
        """
        if self._active_thread and self._active_thread.is_alive():
            logger.info("Voice assistant is already busy with an active query.")
            return

        def _voice_worker():
            self._set_status("LISTENING")
            transcription = self.stt.listen_once(timeout=timeout)
            if transcription:
                self.process_text_query(transcription, speak_output=speak_output)
            else:
                self._set_status("IDLE")

        self._active_thread = threading.Thread(target=_voice_worker, daemon=True, name="AURA_PTT_Worker")
        self._active_thread.start()

    def shutdown(self) -> None:
        """Shuts down speech synthesis and background workers."""
        if self.tts:
            self.tts.shutdown()
