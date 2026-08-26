"""
Unit tests for AURA LLM reasoning providers.
"""

import unittest
from config.config import IntelligenceConfig
from brain.llm import (
    OfflineReasoningProvider,
    GeminiMultimodalProvider,
    OllamaProvider,
    create_llm_provider,
)


class TestLLMProviders(unittest.TestCase):
    def test_offline_reasoning_provider(self):
        """Verify OfflineReasoningProvider returns fast formatted text."""
        provider = OfflineReasoningProvider()
        resp = provider.generate("I see a laptop on the desk.")
        self.assertEqual(resp.provider, "offline")
        self.assertIn("laptop", resp.text)
        self.assertGreaterEqual(resp.latency_ms, 0.0)

    def test_create_llm_provider_factory(self):
        """Verify factory returns appropriate provider instance."""
        cfg_off = IntelligenceConfig(llm_provider="offline")
        p_off = create_llm_provider(cfg_off)
        self.assertIsInstance(p_off, OfflineReasoningProvider)

        cfg_gem = IntelligenceConfig(llm_provider="gemini", llm_model="gemini-1.5-flash")
        p_gem = create_llm_provider(cfg_gem)
        self.assertIsInstance(p_gem, GeminiMultimodalProvider)


if __name__ == "__main__":
    unittest.main()
