from .base import Task
from .apispec import APISpecTaskSpotify, APISpecTaskTMDB

# Optional tasks; import lazily to avoid hard deps
try:
    from .reasoning import ReasoningTask, MATHTask
except Exception:
    pass
try:
    from .codegen import CodeGenTask, HumanEvalTask, MBPPTask, APPSTask
except Exception:
    pass
try:
    from .alfworld import AlfWorldTask
except Exception:
    pass
try:
    from .tabular import WikiTableQuestionsTask
except Exception:
    pass
