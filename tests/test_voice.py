"""
Unit tests for AURA Voice Subsystem (Milestone 7): Text-To-Speech, Speech-To-Text, and VoiceAssistant.
"""

import time
import unittest
from unittest.mock import patch, MagicMock

from voice.tts import TextToSpeech
from voice.stt import SpeechToText
from voice.engine import VoiceAssistant
from brain.conversation import ConversationEngine, ConversationResponse
from brain.intent import IntentType


class TestVoiceSubsystem(unittest.TestCase):
    def test_tts_worker_lifecycle(self):
        """Verify TTS background worker queues and processes messages without blocking."""
        started = []
        finished = []

        tts = TextToSpeech(
            enabled=False,  # Run in mock/silent mode to avoid hardware audio dependencies in CI
            on_start=lambda t: started.append(t),
            on_done=lambda t: finished.append(t),
        )

        tts.speak("Hello AURA test")
        tts.speak("Second message")

        # Allow worker thread time to process mock messages
        time.sleep(0.3)

        self.assertIn("Hello AURA test", started)
        self.assertIn("Hello AURA test", finished)
        self.assertIn("Second message", started)
        self.assertIn("Second message", finished)

        tts.shutdown()

    def test_stt_initialization_and_fallback(self):
        """Verify SpeechToText initializes cleanly and handles hardware unavailability gracefully."""
        stt = SpeechToText(enabled=True)
        # Verify it has valid properties without raising exceptions
        self.assertIsInstance(stt.is_listening, bool)
        self.assertFalse(stt.is_listening)

    def test_voice_assistant_process_query(self):
        """Verify VoiceAssistant coordinates query processing, status updates, and callbacks."""
        statuses = []
        queries = []
        responses = []

        mock_conv = MagicMock()
        mock_conv.respond.return_value = ConversationResponse(
            query="What do you see?",
            intent=IntentType.SCENE_SUMMARY,
            response_text="I see 1 laptop and 1 cup.",
        )

        tts = TextToSpeech(enabled=False)

        va = VoiceAssistant(
            conversation_engine=mock_conv,
            tts=tts,
            enable_tts=True,
            enable_stt=False,
            on_status_change=lambda s: statuses.append(s),
            on_query=lambda q: queries.append(q),
            on_response=lambda r: responses.append(r),
        )

        resp = va.process_text_query("What do you see?", speak_output=False)

        self.assertEqual(resp.response_text, "I see 1 laptop and 1 cup.")
        self.assertIn("What do you see?", queries)
        self.assertEqual(len(responses), 1)
        self.assertIn("THINKING", statuses)

        va.shutdown()


if __name__ == "__main__":
    unittest.main()
