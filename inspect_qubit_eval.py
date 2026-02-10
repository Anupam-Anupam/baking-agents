"""
Inspect AI Evaluation for Qubit Game with Teacher Model

This module implements an Inspect AI evaluation where:
1. A Bread model (student) plays the qubit game
2. A small teacher model (Qwen3) evaluates the student's performance

Based on the Inspect framework: https://inspect.aisi.org.uk

Part of the baking-agents repo: https://github.com/Anupam-Anupam/baking-agents
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

import numpy as np
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from inspect_ai.model import Model, get_model
from inspect_ai.scorer import Score, Scorer, scorer, Target, accuracy
from inspect_ai.solver import Solver, solver, TaskState, Generate

from qubit_game_env import QubitGameEnv, QubitGameConfig
from bread_game_agent import BreadGameAgent, GameLog


# ============================================================================
# Dataset Generation - Create game scenarios for evaluation
# ============================================================================

def generate_game_samples(
    num_samples: int = 10,
    seed: int = 42
) -> list[Sample]:
    """
    Generate evaluation samples where each sample is a game scenario.
    
    Each sample contains:
        - input: The initial game state description
        - target: Expected behavior criteria
        - metadata: Full game configuration
    """
    rng = np.random.default_rng(seed)
    samples = []
    
    for i in range(num_samples):
        # Generate random initial state
        real = rng.normal(size=2)
        imag = rng.normal(size=2)
        psi = real + 1j * imag
        psi = psi / np.linalg.norm(psi)
        
        # Target is always |0> for now (can be extended)
        target_state = np.array([1.0 + 0.0j, 0.0 + 0.0j])
        
        initial_obs = [psi[0].real, psi[0].imag, psi[1].real, psi[1].imag]
        initial_fidelity = float(np.abs(np.vdot(target_state, psi)) ** 2)
        
        input_text = f"""Qubit Game Episode {i}

Initial State: |ψ⟩ = ({psi[0]:.4f})|0⟩ + ({psi[1]:.4f})|1⟩
Target State:  |0⟩
Initial Fidelity: {initial_fidelity:.4f}

The student model will attempt to transform the initial state to the target state
using quantum gates (I, X, Y, Z, H) within 20 steps.

Evaluate the student's:
1. Strategic gate selection
2. Understanding of quantum mechanics
3. Efficiency in reaching the target
4. Learning from feedback"""

        sample = Sample(
            input=input_text,
            target="Fidelity >= 0.99 within 20 steps",
            id=f"episode_{i}",
            metadata={
                "episode_id": i,
                "initial_state_real": [float(psi[0].real), float(psi[1].real)],
                "initial_state_imag": [float(psi[0].imag), float(psi[1].imag)],
                "initial_fidelity": initial_fidelity,
                "seed": seed + i
            }
        )
        samples.append(sample)
    
    return samples


def qubit_game_dataset(num_samples: int = 10, seed: int = 42) -> Dataset:
    """Create a dataset of qubit game scenarios."""
    samples = generate_game_samples(num_samples, seed)
    return MemoryDataset(samples=samples, name="qubit_game")


# ============================================================================
# Custom Solver - Runs the Bread agent on each game scenario
# ============================================================================

@solver
def bread_player_solver(
    model_name: str | None = None,
    verbose: bool = False
) -> Solver:
    """
    Solver that uses the Bread agent to play each game scenario.
    
    The solver runs the game and stores the full game log in the TaskState
    for later evaluation by the teacher model.
    """
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Extract game configuration from metadata
        metadata = state.sample.metadata or {}
        episode_id = metadata.get("episode_id", 0)
        seed = metadata.get("seed", 42)
        
        # Reconstruct initial state
        real = metadata.get("initial_state_real", [1.0, 0.0])
        imag = metadata.get("initial_state_imag", [0.0, 0.0])
        
        # Create environment and agent
        config = QubitGameConfig(max_steps=20)
        env = QubitGameEnv(config)
        
        try:
            agent = BreadGameAgent(model_name=model_name) if model_name else BreadGameAgent()
        except ValueError as e:
            # If no API key, simulate with random agent for testing
            state.metadata["game_log"] = {
                "error": str(e),
                "episode_id": episode_id,
                "solved": False,
                "final_fidelity": 0.0
            }
            state.output.completion = f"Error: {e}"
            return state
        
        # Play the game
        game_log = agent.play_episode(env, episode_id=episode_id, seed=seed, verbose=verbose)
        
        # Store game log in state for scorer
        state.metadata["game_log"] = asdict(game_log)
        
        # Format output for the scorer
        output_text = format_game_log_for_evaluation(game_log)
        state.output.completion = output_text
        
        return state
    
    return solve


def format_game_log_for_evaluation(log: GameLog) -> str:
    """Format the game log into a readable string for the teacher model."""
    
    lines = [
        f"=== Game Episode {log.episode_id} Results ===",
        f"",
        f"Outcome: {'SOLVED' if log.solved else 'NOT SOLVED'}",
        f"Final Fidelity: {log.final_fidelity:.4f}",
        f"Total Reward: {log.total_reward:.4f}",
        f"Steps Taken: {len(log.steps)}",
        f"",
        f"=== Step-by-Step Actions ===",
    ]
    
    for i, (step, reasoning) in enumerate(zip(log.steps, log.model_reasoning)):
        lines.append(
            f"Step {step['step']:2d}: {step['action_name']:12s} | "
            f"Fidelity: {step['fidelity']:.4f} | "
            f"Reward: {step['reward']:.4f}"
        )
        # Include truncated reasoning
        short_reasoning = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
        lines.append(f"         Reasoning: {short_reasoning}")
    
    lines.extend([
        f"",
        f"=== Analysis Required ===",
        f"Please evaluate the student's performance considering:",
        f"1. Did they make progress toward the target?",
        f"2. Were their gate choices sensible?",
        f"3. Did they show understanding of quantum mechanics?",
        f"4. How could they improve?",
    ])
    
    return "\n".join(lines)


# ============================================================================
# Teacher Model Scorer - Uses Qwen3 to evaluate student performance
# ============================================================================

TEACHER_EVALUATION_PROMPT = """You are a quantum computing teacher evaluating a student's performance on a qubit manipulation game.

The student was given a random initial qubit state and needed to transform it to the target state |0⟩ using quantum gates (Identity, Pauli-X, Pauli-Y, Pauli-Z, Hadamard).

Review the student's game log below and provide:
1. A score from 0.0 to 1.0 based on their performance
2. Detailed feedback on their strategy
3. Suggestions for improvement

Scoring Guidelines:
- 1.0: Solved efficiently with clear quantum understanding
- 0.8-0.9: Solved or very close, reasonable strategy
- 0.5-0.7: Made progress, some understanding shown
- 0.2-0.4: Limited progress, random or confused choices
- 0.0-0.1: No progress, no apparent strategy

Student's Game Log:
{game_log}

Respond in this exact JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "solved": <true/false>,
    "strategy_rating": "<excellent/good/fair/poor>",
    "quantum_understanding": "<strong/moderate/weak/none>",
    "feedback": "<detailed feedback string>",
    "suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
}}"""


@scorer(metrics=[accuracy()])
def teacher_model_scorer(
    teacher_model: str = "ollama/qwen3:0.6b",
    temperature: float = 0.1
) -> Scorer:
    """
    Scorer that uses a small teacher model to evaluate game performance.
    
    Default uses Qwen3 0.6B via Ollama for fast local inference.
    Can also use other small models like:
    - "ollama/qwen3:1.7b"
    - "ollama/qwen3:4b"
    - "ollama/phi3:mini"
    - "ollama/llama3.2:1b"
    
    Args:
        teacher_model: Model string for the teacher (Inspect format)
        temperature: Sampling temperature for teacher
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        # Get the game log from state
        game_log_text = state.output.completion
        game_log_data = state.metadata.get("game_log", {})
        
        # Check for errors
        if "error" in game_log_data:
            return Score(
                value=0.0,
                answer=game_log_text,
                explanation=f"Error during game: {game_log_data['error']}"
            )
        
        # Build prompt for teacher
        prompt = TEACHER_EVALUATION_PROMPT.format(game_log=game_log_text)
        
        try:
            # Get teacher model
            teacher = get_model(teacher_model)
            
            # Generate evaluation
            response = await teacher.generate(
                prompt,
                max_tokens=500,
                temperature=temperature
            )
            
            evaluation_text = response.completion
            
            # Parse JSON response
            try:
                # Extract JSON from response (handle potential markdown wrapping)
                json_str = evaluation_text
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                
                evaluation = json.loads(json_str.strip())
                score_value = float(evaluation.get("score", 0.5))
                score_value = max(0.0, min(1.0, score_value))  # Clamp to [0, 1]
                
                explanation = (
                    f"Strategy: {evaluation.get('strategy_rating', 'N/A')}\n"
                    f"Understanding: {evaluation.get('quantum_understanding', 'N/A')}\n"
                    f"Feedback: {evaluation.get('feedback', 'N/A')}"
                )
                
            except (json.JSONDecodeError, KeyError, ValueError):
                # Fallback: use game outcome for scoring
                solved = game_log_data.get("solved", False)
                fidelity = game_log_data.get("final_fidelity", 0.0)
                score_value = 1.0 if solved else fidelity
                explanation = f"Teacher parse error. Using fidelity score. Raw: {evaluation_text[:200]}"
            
            return Score(
                value=score_value,
                answer=game_log_text,
                explanation=explanation,
                metadata={
                    "teacher_response": evaluation_text,
                    "game_solved": game_log_data.get("solved", False),
                    "final_fidelity": game_log_data.get("final_fidelity", 0.0)
                }
            )
            
        except Exception as e:
            # Fallback scoring based on game outcome
            solved = game_log_data.get("solved", False)
            fidelity = game_log_data.get("final_fidelity", 0.0)
            
            return Score(
                value=1.0 if solved else fidelity * 0.8,
                answer=game_log_text,
                explanation=f"Teacher model error: {e}. Scored by fidelity.",
                metadata={"error": str(e)}
            )
    
    return score


# ============================================================================
# Main Task Definition
# ============================================================================

@task
def qubit_game_eval(
    num_episodes: int = 10,
    bread_model: str | None = None,
    teacher_model: str = "ollama/qwen3:0.6b",
    seed: int = 42,
    verbose: bool = False
) -> Task:
    """
    Main Inspect evaluation task for the qubit game.
    
    This task:
    1. Generates game scenarios as a dataset
    2. Uses the Bread model to play each game (solver)
    3. Uses a small teacher model to evaluate performance (scorer)
    
    Args:
        num_episodes: Number of game episodes to evaluate
        bread_model: Bread model path for the student agent
        teacher_model: Small model for teacher evaluation (default: Qwen3 0.6B)
        seed: Random seed for reproducibility
        verbose: Whether to print game progress
        
    Returns:
        Inspect Task ready for evaluation
        
    Usage:
        inspect eval inspect_qubit_eval.py --model openai/gpt-4o
        
        Or from Python:
        from inspect_qubit_eval import qubit_game_eval
        eval(qubit_game_eval(), model="openai/gpt-4o")
    """
    return Task(
        dataset=qubit_game_dataset(num_episodes, seed),
        solver=bread_player_solver(model_name=bread_model, verbose=verbose),
        scorer=teacher_model_scorer(teacher_model=teacher_model),
        metadata={
            "description": "Qubit game evaluation with teacher model feedback",
            "bread_model": bread_model or "default",
            "teacher_model": teacher_model,
            "num_episodes": num_episodes
        }
    )


# ============================================================================
# Alternative: Simple fidelity-based scorer (no teacher model)
# ============================================================================

@scorer(metrics=[accuracy()])  
def fidelity_scorer() -> Scorer:
    """Simple scorer based on final fidelity - no teacher model needed."""
    
    async def score(state: TaskState, target: Target) -> Score:
        game_log = state.metadata.get("game_log", {})
        
        solved = game_log.get("solved", False)
        fidelity = game_log.get("final_fidelity", 0.0)
        steps = len(game_log.get("steps", []))
        
        # Score: 1.0 if solved, otherwise scaled fidelity with efficiency bonus
        if solved:
            efficiency_bonus = max(0, (20 - steps) / 20) * 0.1
            score_value = min(1.0, 0.9 + efficiency_bonus)
        else:
            score_value = fidelity * 0.8
        
        return Score(
            value=score_value,
            answer=state.output.completion,
            explanation=f"Solved: {solved}, Fidelity: {fidelity:.4f}, Steps: {steps}"
        )
    
    return score


@task
def qubit_game_eval_simple(
    num_episodes: int = 10,
    bread_model: str | None = None,
    seed: int = 42
) -> Task:
    """Simplified task using fidelity scoring (no teacher model required)."""
    return Task(
        dataset=qubit_game_dataset(num_episodes, seed),
        solver=bread_player_solver(model_name=bread_model),
        scorer=fidelity_scorer()
    )


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Qubit Game Evaluation")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--bread-model", type=str, default=None, help="Bread model path")
    parser.add_argument("--teacher-model", type=str, default="ollama/qwen3:0.6b", 
                        help="Teacher model for evaluation")
    parser.add_argument("--simple", action="store_true", 
                        help="Use simple fidelity scorer instead of teacher")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎓 Qubit Game Evaluation with Inspect AI")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Bread Model: {args.bread_model or 'default'}")
    print(f"Teacher Model: {args.teacher_model}")
    print(f"Mode: {'Simple (fidelity)' if args.simple else 'Teacher evaluation'}")
    print("=" * 60)
    
    if args.simple:
        task_fn = qubit_game_eval_simple(
            num_episodes=args.episodes,
            bread_model=args.bread_model,
            seed=args.seed
        )
    else:
        task_fn = qubit_game_eval(
            num_episodes=args.episodes,
            bread_model=args.bread_model,
            teacher_model=args.teacher_model,
            seed=args.seed,
            verbose=args.verbose
        )
    
    # Run evaluation (requires a model to be specified for the task)
    # Use: inspect eval inspect_qubit_eval.py --model openai/gpt-4o
    print("\nTo run this evaluation, use:")
    print(f"  inspect eval inspect_qubit_eval.py --model openai/gpt-4o")
    print("\nOr start the Inspect viewer:")
    print("  inspect view")
