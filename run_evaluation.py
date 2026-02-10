#!/usr/bin/env python3
"""
Main Runner Script for Qubit Game Evaluation

This script orchestrates the full evaluation pipeline:
1. Sets up the environment
2. Runs the Bread model agent playing games
3. Has the teacher model (Qwen3) evaluate performance
4. Generates logs viewable in Inspect View

Part of the baking-agents repo: https://github.com/Anupam-Anupam/baking-agents

Usage:
    # Run with defaults (5 episodes, simple scoring)
    python run_evaluation.py
    
    # Run with teacher model evaluation
    python run_evaluation.py --teacher
    
    # Run with custom settings
    python run_evaluation.py --episodes 10 --bread-model "user/repo/bake/1"
    
    # View results in browser
    inspect view
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import dotenv
dotenv.load_dotenv()


def check_dependencies():
    """Verify all required dependencies are installed."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        import inspect_ai
    except ImportError:
        missing.append("inspect-ai")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    return True


def check_ollama_available(model: str = "qwen3:0.6b") -> bool:
    """Check if Ollama is running and has the model available."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if any(model in name or name in model for name in model_names):
                return True
            print(f"⚠️  Ollama is running but '{model}' not found.")
            print(f"   Pull it with: ollama pull {model}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  Ollama is not running. Start it with: ollama serve")
        return False
    except Exception as e:
        print(f"⚠️  Could not check Ollama: {e}")
        return False
    return False


def run_standalone_evaluation(
    num_episodes: int = 5,
    bread_model: str | None = None,
    teacher_model: str = "qwen3:0.6b",
    use_teacher: bool = False,
    seed: int = 42,
    verbose: bool = True,
    output_dir: str = "./logs"
) -> dict:
    """
    Run evaluation without Inspect AI (standalone mode).
    
    Useful for quick testing or when Inspect isn't needed.
    """
    from qubit_game_env import QubitGameEnv, QubitGameConfig
    from bread_game_agent import BreadGameAgent, play_games, GameLog
    
    print("\n" + "=" * 60)
    print("🎮 STANDALONE QUBIT GAME EVALUATION")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Bread Model: {bread_model or 'default'}")
    print(f"Teacher Evaluation: {'Yes' if use_teacher else 'No'}")
    print("=" * 60 + "\n")
    
    # Play games
    try:
        logs = play_games(
            num_episodes=num_episodes,
            model_name=bread_model,
            seed=seed
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Set BREAD_API_KEY environment variable or pass --bread-model")
        return {"error": str(e)}
    
    # Evaluate with teacher model if requested
    teacher_evaluations = []
    if use_teacher:
        print("\n" + "=" * 60)
        print("🎓 TEACHER MODEL EVALUATION")
        print("=" * 60)
        
        if check_ollama_available(teacher_model):
            teacher_evaluations = evaluate_with_teacher(logs, teacher_model, verbose)
        else:
            print("Skipping teacher evaluation (Ollama not available)")
    
    # Generate summary
    summary = generate_summary(logs, teacher_evaluations)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "timestamp": timestamp,
        "config": {
            "num_episodes": num_episodes,
            "bread_model": bread_model,
            "teacher_model": teacher_model if use_teacher else None,
            "seed": seed
        },
        "summary": summary,
        "episodes": [asdict(log) for log in logs],
        "teacher_evaluations": teacher_evaluations
    }
    
    output_file = Path(output_dir) / f"eval_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return results


def evaluate_with_teacher(
    logs: list,
    teacher_model: str,
    verbose: bool = True
) -> list[dict]:
    """Use local Ollama model to evaluate game logs."""
    import requests
    
    evaluations = []
    
    for log in logs:
        if verbose:
            print(f"\nEvaluating Episode {log.episode_id}...")
        
        # Build prompt
        prompt = build_teacher_prompt(log)
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": teacher_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            evaluation_text = result.get("response", "")
            
            # Parse evaluation
            evaluation = parse_teacher_response(evaluation_text, log)
            evaluations.append(evaluation)
            
            if verbose:
                print(f"  Score: {evaluation.get('score', 'N/A'):.2f}")
                print(f"  Strategy: {evaluation.get('strategy_rating', 'N/A')}")
                
        except Exception as e:
            print(f"  ⚠️  Teacher evaluation failed: {e}")
            evaluations.append({
                "episode_id": log.episode_id,
                "error": str(e),
                "score": log.final_fidelity
            })
    
    return evaluations


def build_teacher_prompt(log) -> str:
    """Build evaluation prompt for teacher model."""
    
    steps_text = "\n".join([
        f"Step {s['step']}: {s['action_name']} -> fidelity {s['fidelity']:.4f}"
        for s in log.steps[:10]  # Limit to first 10 steps for shorter context
    ])
    
    if len(log.steps) > 10:
        steps_text += f"\n... and {len(log.steps) - 10} more steps"
    
    return f"""You are a quantum computing teacher evaluating a student's performance.

The student played a qubit manipulation game:
- Initial fidelity: {log.steps[0]['fidelity'] if log.steps else 0:.4f}
- Final fidelity: {log.final_fidelity:.4f}
- Solved: {log.solved}
- Steps taken: {len(log.steps)}

Actions taken:
{steps_text}

Rate the student's performance. Respond with JSON:
{{"score": <0.0-1.0>, "strategy_rating": "<excellent/good/fair/poor>", "feedback": "<brief feedback>"}}"""


def parse_teacher_response(response: str, log) -> dict:
    """Parse teacher model's JSON response."""
    try:
        # Try to extract JSON
        import re
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "episode_id": log.episode_id,
                "score": float(data.get("score", log.final_fidelity)),
                "strategy_rating": data.get("strategy_rating", "unknown"),
                "feedback": data.get("feedback", ""),
                "raw_response": response
            }
    except:
        pass
    
    # Fallback
    return {
        "episode_id": log.episode_id,
        "score": 1.0 if log.solved else log.final_fidelity * 0.8,
        "strategy_rating": "excellent" if log.solved else "fair",
        "feedback": "Auto-scored based on fidelity",
        "raw_response": response
    }


def generate_summary(logs: list, teacher_evaluations: list) -> dict:
    """Generate evaluation summary statistics."""
    
    solved_count = sum(1 for log in logs if log.solved)
    avg_fidelity = sum(log.final_fidelity for log in logs) / len(logs)
    avg_reward = sum(log.total_reward for log in logs) / len(logs)
    avg_steps = sum(len(log.steps) for log in logs) / len(logs)
    
    summary = {
        "total_episodes": len(logs),
        "solved_episodes": solved_count,
        "solve_rate": solved_count / len(logs),
        "avg_final_fidelity": avg_fidelity,
        "avg_total_reward": avg_reward,
        "avg_steps": avg_steps
    }
    
    if teacher_evaluations:
        valid_evals = [e for e in teacher_evaluations if "score" in e and "error" not in e]
        if valid_evals:
            summary["teacher_avg_score"] = sum(e["score"] for e in valid_evals) / len(valid_evals)
            
            ratings = [e.get("strategy_rating", "unknown") for e in valid_evals]
            summary["teacher_rating_distribution"] = {
                r: ratings.count(r) for r in set(ratings)
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes Played: {summary['total_episodes']}")
    print(f"Episodes Solved: {summary['solved_episodes']} ({summary['solve_rate']*100:.1f}%)")
    print(f"Avg Final Fidelity: {summary['avg_final_fidelity']:.4f}")
    print(f"Avg Total Reward: {summary['avg_total_reward']:.4f}")
    print(f"Avg Steps: {summary['avg_steps']:.1f}")
    
    if "teacher_avg_score" in summary:
        print(f"\nTeacher Model Scores:")
        print(f"  Avg Score: {summary['teacher_avg_score']:.2f}")
        print(f"  Ratings: {summary.get('teacher_rating_distribution', {})}")
    
    print("=" * 60)
    
    return summary


def run_inspect_evaluation(
    num_episodes: int = 5,
    bread_model: str | None = None,
    teacher_model: str = "ollama/qwen3:0.6b",
    use_teacher: bool = True,
    seed: int = 42,
    verbose: bool = False
):
    """Run evaluation using the full Inspect AI framework."""
    
    print("\n" + "=" * 60)
    print("🔍 INSPECT AI EVALUATION")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Bread Model: {bread_model or 'default'}")
    print(f"Teacher Model: {teacher_model}")
    print("=" * 60)
    
    try:
        from inspect_ai import eval as inspect_eval
        from inspect_qubit_eval import qubit_game_eval, qubit_game_eval_simple
        
        if use_teacher:
            task = qubit_game_eval(
                num_episodes=num_episodes,
                bread_model=bread_model,
                teacher_model=teacher_model,
                seed=seed,
                verbose=verbose
            )
        else:
            task = qubit_game_eval_simple(
                num_episodes=num_episodes,
                bread_model=bread_model,
                seed=seed
            )
        
        # Note: Running inspect eval requires specifying a model
        # This is typically done via CLI: inspect eval inspect_qubit_eval.py --model openai/gpt-4o
        print("\n⚠️  To run full Inspect evaluation, use the CLI:")
        print(f"   inspect eval inspect_qubit_eval.py --model openai/gpt-4o")
        print("\nOr start the Inspect viewer:")
        print("   inspect view")
        print("\nFor standalone evaluation (no Inspect), use: --standalone")
        
    except ImportError as e:
        print(f"❌ Inspect AI not available: {e}")
        print("Install with: pip install inspect-ai")


def main():
    parser = argparse.ArgumentParser(
        description="Run Qubit Game Evaluation with Bread Agent and Teacher Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick standalone test (no Inspect)
  python run_evaluation.py --standalone --episodes 3
  
  # With teacher model evaluation  
  python run_evaluation.py --standalone --teacher --episodes 5
  
  # Show Inspect CLI command
  python run_evaluation.py --inspect
  
  # View previous results
  inspect view
        """
    )
    
    parser.add_argument(
        "--episodes", "-n", type=int, default=5,
        help="Number of game episodes (default: 5)"
    )
    parser.add_argument(
        "--bread-model", "-b", type=str, default=None,
        help="Bread model path (e.g., user/repo/bake/checkpoint)"
    )
    parser.add_argument(
        "--teacher-model", "-t", type=str, default="qwen3:0.6b",
        help="Teacher model for evaluation (default: qwen3:0.6b)"
    )
    parser.add_argument(
        "--teacher", action="store_true",
        help="Enable teacher model evaluation"
    )
    parser.add_argument(
        "--standalone", action="store_true",
        help="Run standalone evaluation (without full Inspect)"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Show Inspect AI evaluation commands"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="./logs",
        help="Output directory for results (default: ./logs)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n" + "🍞" * 30)
    print("  BAKING AGENTS - QUBIT GAME EVALUATION")
    print("🍞" * 30)
    
    if args.inspect:
        run_inspect_evaluation(
            num_episodes=args.episodes,
            bread_model=args.bread_model,
            teacher_model=f"ollama/{args.teacher_model}",
            use_teacher=args.teacher,
            seed=args.seed,
            verbose=args.verbose
        )
    elif args.standalone:
        run_standalone_evaluation(
            num_episodes=args.episodes,
            bread_model=args.bread_model,
            teacher_model=args.teacher_model,
            use_teacher=args.teacher,
            seed=args.seed,
            verbose=args.verbose,
            output_dir=args.output_dir
        )
    else:
        # Default: show both options
        print("\nChoose an evaluation mode:\n")
        print("1. Standalone (quick, local):")
        print("   python run_evaluation.py --standalone --episodes 5")
        print("\n2. With Teacher Model:")
        print("   python run_evaluation.py --standalone --teacher --episodes 5")
        print("\n3. Full Inspect AI Framework:")
        print("   inspect eval inspect_qubit_eval.py --model openai/gpt-4o")
        print("\n4. View Results:")
        print("   inspect view")


if __name__ == "__main__":
    main()
