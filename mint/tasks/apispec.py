import os
import json
from typing import Iterable, List, Tuple

from mint.tasks.base import Task


def _normalize_plan(x):
    if isinstance(x, list):
        return [i.strip() for i in x]
    try:
        obj = json.loads(x)
        if isinstance(obj, list):
            return [str(i).strip() for i in obj]
    except Exception:
        pass
    lines = [i.strip() for i in str(x).splitlines() if i.strip()]
    return lines


class _APISpecTask(Task):
    def __init__(self, id: str, prompt: str, reference: List[str], api_name: str, **kwargs):
        self.task_name = f"api/{api_name}"
        super().__init__(**kwargs)
        self._id = id
        self._prompt = prompt.strip()
        self._reference = reference

    def extract_answer(self, solution: str):
        return _normalize_plan(solution)

    def success(self, solution: str) -> bool:
        ans = self.extract_answer(solution)
        ref = _normalize_plan(self.reference)
        return ans == ref

    @classmethod
    def _load_common(cls, path: str, api_name: str, limit: int = None) -> Tuple[Iterable["Task"], int]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        n = len(data)
        if limit is not None:
            n = min(n, limit)

        def gen():
            for i, item in enumerate(data):
                if limit is not None and i >= limit:
                    break
                prompt = item.get("query", "")
                ref = item.get("solution", [])
                yield cls(id=str(i), prompt=prompt, reference=ref, api_name=api_name, loaded_history=None)
        return gen(), n


class APISpecTaskSpotify(_APISpecTask):
    task_name = "api/spotify"

    @classmethod
    def load_tasks(cls, path: str = "./experiment/datasets/spotify.json", **kwargs):
        limit = kwargs.get("limit")
        return cls._load_common(path, "spotify", limit)


class APISpecTaskTMDB(_APISpecTask):
    task_name = "api/tmdb"

    @classmethod
    def load_tasks(cls, path: str = "./experiment/datasets/tmdb.json", **kwargs):
        limit = kwargs.get("limit")
        return cls._load_common(path, "tmdb", limit)

