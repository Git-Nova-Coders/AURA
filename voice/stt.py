"""
AURA Speech-To-Text (STT) Module (Milestone 7)
Captures microphone audio, applies ambient noise calibration,
and transcribes speech using speech_recognition with graceful fallback.
"""

import logging
from typing import Optional, Callable, Any

try:
    import speech_recognition as sr
    _HAS_SPEECH_RECOGNITION = True
except ImportError:
    _HAS_SPEECH_RECOGNITION = False

logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Speech recognition engine supporting push-to-talk microphone capture,
    noise calibration, and console input fallback.
    """

    def __init__(
        self,
        enabled: bool = True,
        language: str = "en-US",
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
    ):
        self.enabled = enabled
        self.language = language
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold

        self._recognizer: Optional[Any] = None
        self._microphone: Optional[Any] = None
        self._is_listening = False

        if _HAS_SPEECH_RECOGNITION and self.enabled:
            self._init_recognizer()

    def _init_recognizer(self) -> None:
        """Initializes the SpeechRecognition Recognizer and verifies microphone access."""
        try:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = self.energy_threshold
            self._recognizer.pause_threshold = self.pause_threshold
            self._recognizer.dynamic_energy_threshold = True

            # Check if microphone is accessible
            try:
                self._microphone = sr.Microphone()
                logger.info("Microphone initialized for Speech-to-Text.")
            except Exception as e:
                logger.warning(f"No functional microphone detected: {e}. STT will operate in fallback mode.")
                self._microphone = None
        except Exception as e:
            logger.warning(f"Failed to initialize SpeechRecognizer: {e}")
            self._recognizer = None

    @property
    def is_available(self) -> bool:
        return _HAS_SPEECH_RECOGNITION and self.enabled and (self._recognizer is not None)

    @property
    def has_microphone(self) -> bool:
        return self.is_available and (self._microphone is not None)

    @property
    def is_listening(self) -> bool:
        return self._is_listening

    def calibrate_noise(self, duration: float = 1.0) -> None:
        """Calibrates microphone energy threshold against ambient room noise."""
        if not self.has_microphone:
            return

        try:
            logger.info("Calibrating microphone against ambient noise...")
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=duration)
            logger.info(f"Microphone calibrated. Energy threshold set to: {self._recognizer.energy_threshold}")
        except Exception as e:
            logger.warning(f"Ambient noise calibration failed: {e}")

    def listen_once(self, timeout: float = 5.0, phrase_time_limit: float = 8.0) -> Optional[str]:
        """
        Captures a single audio phrase from the microphone and returns transcribed text.
        
        Args:
            timeout: Maximum seconds to wait for speech to begin.
            phrase_time_limit: Maximum duration of a spoken phrase.
            
        Returns:
            Optional[str]: Transcribed text, or None if no speech / recognition failure.
        """
        if not self.has_microphone:
            logger.debug("Microphone not available for live voice capture.")
            return None

        self._is_listening = True
        try:
            logger.info("AURA is listening for voice input...")
            with self._microphone as source:
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

            logger.info("Audio captured. Transcribing...")
            text = self._recognizer.recognize_google(audio, language=self.language)
            logger.info(f"[AURA Heard] \"{text}\"")
            return text.strip()

        except sr.WaitTimeoutError:
            logger.info("Voice input timed out: no speech detected.")
            return None
        except sr.UnknownValueError:
            logger.info("Speech was unintelligible or noise.")
            return None
        except sr.RequestError as e:
            logger.warning(f"Google Speech Recognition service error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Voice capture error: {e}")
            return None
        finally:
            self._is_listening = False

    def transcribe_audio_file(self, audio_path: str) -> Optional[str]:
        """Transcribes an audio recording from file (useful for automated testing)."""
        if not self.is_available:
            return None

        try:
            with sr.AudioFile(audio_path) as source:
                audio = self._recognizer.record(source)
            text = self._recognizer.recognize_google(audio, language=self.language)
            return text.strip()
        except Exception as e:
            logger.warning(f"Audio file transcription error: {e}")
            return None
