"""
AURA Brain Package (Milestone 6)
Provides context management, spatial reasoning, intent parsing, and conversational reasoning.
"""

from .context import (
    ObjectEntity,
    SpatialRelation,
    SceneContext,
    ContextManager,
)
from .intent import (
    IntentType,
    ParsedQuery,
    IntentClassifier,
)
from .conversation import (
    ConversationResponse,
    ConversationEngine,
)
from .memory import (
    EpisodicEvent,
    EpisodicMemory,
)
from .llm import (
    BaseLLMProvider,
    OfflineReasoningProvider,
    GeminiMultimodalProvider,
    OllamaProvider,
    create_llm_provider,
)

__all__ = [
    "ObjectEntity",
    "SpatialRelation",
    "SceneContext",
    "ContextManager",
    "IntentType",
    "ParsedQuery",
    "IntentClassifier",
    "ConversationResponse",
    "ConversationEngine",
    "EpisodicEvent",
    "EpisodicMemory",
    "BaseLLMProvider",
    "OfflineReasoningProvider",
    "GeminiMultimodalProvider",
    "OllamaProvider",
    "create_llm_provider",
]
