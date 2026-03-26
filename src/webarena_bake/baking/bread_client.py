from __future__ import annotations

import os
from typing import Any


class BreadClient:
    def __init__(self, api_key: str | None = None, dry_run: bool = True):
        self.api_key = api_key or os.environ.get("BREAD_API_KEY")
        self.dry_run = dry_run
        self._client = None
        if not self.dry_run:
            from aibread import Bread  # type: ignore

            self._client = Bread(api_key=self.api_key)

    def repo_set(self, repo_name: str, base_model: str | None = None) -> dict[str, Any]:
        if self.dry_run:
            return {"repo_name": repo_name, "base_model": base_model or "default", "dry_run": True}
        resp = self._client.repo.set(repo_name=repo_name, base_model=base_model)
        return {"repo_name": resp.repo_name, "base_model": resp.base_model}

    def prompt_set(self, repo_name: str, prompt_name: str, content: str) -> dict[str, Any]:
        if self.dry_run:
            return {"repo_name": repo_name, "prompt_name": prompt_name, "dry_run": True}
        self._client.prompts.set(
            repo_name=repo_name,
            prompt_name=prompt_name,
            messages=[{"role": "system", "content": content}],
        )
        return {"repo_name": repo_name, "prompt_name": prompt_name}

    def target_set(
        self,
        repo_name: str,
        target_name: str,
        teacher_prompt_name: str,
        student_prompt_name: str,
        generators: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.dry_run:
            return {"repo_name": repo_name, "target_name": target_name, "dry_run": True}
        self._client.targets.set(
            repo_name=repo_name,
            target_name=target_name,
            template="default",
            overrides={"u": teacher_prompt_name, "v": student_prompt_name, "generators": generators},
        )
        return {"repo_name": repo_name, "target_name": target_name}

    def run_stim(self, repo_name: str, target_name: str) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "complete", "dry_run": True, "lines": 0}
        status = self._client.targets.stim.run(repo_name=repo_name, target_name=target_name, poll=True)
        return {"status": status.status, "lines": getattr(status, "lines", 0)}

    def run_rollout(self, repo_name: str, target_name: str) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "complete", "dry_run": True, "lines": 0}
        status = self._client.targets.rollout.run(repo_name=repo_name, target_name=target_name, poll=True)
        return {"status": status.status, "lines": getattr(status, "lines", 0)}

    def bake_set(self, repo_name: str, bake_name: str, target_name: str, weight: float = 1.0) -> dict[str, Any]:
        if self.dry_run:
            return {"repo_name": repo_name, "bake_name": bake_name, "dry_run": True}
        self._client.bakes.set(
            repo_name=repo_name,
            bake_name=bake_name,
            template="default",
            overrides={"datasets": [{"target": target_name, "weight": weight}]},
        )
        return {"repo_name": repo_name, "bake_name": bake_name}

    def bake_run(self, repo_name: str, bake_name: str) -> dict[str, Any]:
        if self.dry_run:
            return {"status": "started", "bake_id": f"dryrun-{bake_name}", "dry_run": True}
        self._client.bakes.run(repo_name=repo_name, bake_name=bake_name)
        return {"status": "started", "bake_id": bake_name}

