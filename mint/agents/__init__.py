from .base import LMAgent
from .openai_lm_agent import OpenAILMAgent
from .openai_feedback_agent import OpenAIFeedbackAgent

# Optional agents; import lazily to avoid hard deps
try:
    from .vllm_agent import VLLMAgent
except Exception:
    pass
try:
    from .vllm_feedback_agent import VLLMFeedbackAgent
except Exception:
    pass
try:
    from .claude_agent import ClaudeLMAgent
except Exception:
    pass
try:
    from .claude_feedback_agent import ClaudeFeedbackAgent
except Exception:
    pass
try:
    from .bard_agent import BardLMAgent
except Exception:
    pass
