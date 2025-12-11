from .base import Tool
from typing import List
import warnings
import os
import json

warnings.filterwarnings("ignore")


def get_toolset_description(tools: List[Tool]) -> str:
    if len(tools) == 0:
        return ""

    output = "Tool function available (already imported in <execute> environment):\n"
    for i, tool in enumerate(tools):
        output += f"[{i + 1}] {tool.signature}\n"
        output += f"{tool.description}\n"
        spec_path = getattr(tool, "spec_path", None)
        if spec_path and os.path.exists(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec = json.load(f)
                paths = spec.get("paths", {})
                eps = []
                for p, methods in paths.items():
                    for m in methods.keys():
                        eps.append(f"{m.upper()} {p}")
                if eps:
                    limit = 50
                    output += "Endpoints:\n"
                    for ep in eps[:limit]:
                        output += f"- {ep}\n"
                    if len(eps) > limit:
                        output += f"... (+{len(eps) - limit} more)\n"
            except Exception:
                pass

    return output
