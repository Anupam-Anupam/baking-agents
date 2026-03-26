from __future__ import annotations

from pathlib import Path
from typing import Any

from sales_agent.baking.bake_runner import BakeRunConfig, run_bake_pipeline
from sales_agent.baking.target_builder import TargetSpec
from sales_agent.distillation.batch_rule_synthesis import synthesize_distilled_rules
from sales_agent.distillation.prompt_recipe_builder import build_prompt_recipe
from sales_agent.memory.lesson_store import LessonStore
from sales_agent.observer.episode_to_lessons import extract_atomic_lessons
from sales_agent.schemas.types import Trajectory, TrajectoryStep
from sales_agent.utils.io import read_json, write_json, write_jsonl


def load_trajectories_from_eval(eval_json_path: Path, run_id: str, task_family: str = "qubit_control") -> list[Trajectory]:
    payload = read_json(eval_json_path)
    episodes = payload.get("episodes", [])
    trajectories: list[Trajectory] = []
    for ep in episodes:
        steps = [
            TrajectoryStep(
                step=s["step"],
                action=s["action"],
                action_name=s["action_name"],
                fidelity=s["fidelity"],
                reward=s["reward"],
                observation=s["observation"],
            )
            for s in ep.get("steps", [])
        ]
        trajectories.append(
            Trajectory(
                run_id=run_id,
                task_family=task_family,
                episode_id=ep["episode_id"],
                solved=ep["solved"],
                final_fidelity=ep["final_fidelity"],
                total_reward=ep["total_reward"],
                steps=steps,
                metadata={"source_file": str(eval_json_path)},
            )
        )
    return trajectories


def run_learning_to_bake(
    workspace_dir: Path,
    eval_json_path: Path,
    run_id: str,
    repo_name: str,
    base_model: str | None = None,
    dry_run: bool = True,
    min_support: int = 2,
) -> dict[str, Any]:
    trajectories = load_trajectories_from_eval(eval_json_path=eval_json_path, run_id=run_id)

    lessons = extract_atomic_lessons(trajectories)
    lesson_store = LessonStore(workspace_dir / "results" / "observer" / "atomic_lessons.jsonl")
    lesson_store.add_lessons(lessons)

    all_lessons = lesson_store.all_lessons()
    distilled_rules = synthesize_distilled_rules(all_lessons, min_support=min_support)
    if not distilled_rules and lessons:
        distilled_rules = synthesize_distilled_rules(lessons, min_support=1)

    recipe_version = f"recipe_{run_id}"
    recipe = build_prompt_recipe(
        recipe_version=recipe_version,
        distilled_rules=distilled_rules,
        task_families=["qubit_control"],
        teacher_prompt_name=f"qubit_teacher_{run_id}",
        student_prompt_name=f"qubit_student_{run_id}",
        domain_name="qubit-control",
    )

    distill_dir = workspace_dir / "results" / "distill" / recipe_version
    distill_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(distill_dir / "atomic_lessons.jsonl", [lesson.to_dict() for lesson in lessons])
    write_json(distill_dir / "distilled_rules.json", {"rules": [rule.to_dict() for rule in distilled_rules]})
    write_json(distill_dir / "bread_recipe.json", recipe.to_dict())
    (distill_dir / "teacher_prompt.txt").write_text(recipe.teacher_prompt_text, encoding="utf-8")
    (distill_dir / "student_prompt.txt").write_text(recipe.student_prompt_text, encoding="utf-8")

    bake_summary = run_bake_pipeline(
        recipe=recipe,
        target_spec=TargetSpec(
            repo_name=repo_name,
            target_name=f"qubit_target_{run_id}",
            teacher_prompt_name=recipe.teacher_prompt_name,
            student_prompt_name=recipe.student_prompt_name,
            task_family="qubit_control",
        ),
        config=BakeRunConfig(
            repo_name=repo_name,
            base_model=base_model,
            target_name=f"qubit_target_{run_id}",
            bake_name=f"qubit_bake_{run_id}",
            dry_run=dry_run,
            stim_count=64,
        ),
        workspace_dir=workspace_dir,
    )

    return {
        "run_id": run_id,
        "trajectories": len(trajectories),
        "new_lessons": len(lessons),
        "distilled_rules": len(distilled_rules),
        "distill_output_dir": str(distill_dir),
        "bake_summary": bake_summary,
    }

