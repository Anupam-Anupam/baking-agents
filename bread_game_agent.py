"""
Bread Game Agent - Uses the Bread model to play the qubit game.

This module provides a wrapper that lets the Bread model interact with
the QubitGameEnv, making decisions based on the current state observations.

Part of the baking-agents repo: https://github.com/Anupam-Anupam/baking-agents
"""

from __future__ import annotations

import os
import json
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import dotenv
import numpy as np

from qubit_game_env import QubitGameEnv, QubitGameConfig

dotenv.load_dotenv()


@dataclass
class GameLog:
    """Container for a single game episode's logs."""
    episode_id: int
    initial_state: List[float]
    target_state: List[float]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    total_reward: float = 0.0
    solved: bool = False
    final_fidelity: float = 0.0
    model_reasoning: List[str] = field(default_factory=list)


class BreadGameAgent:
    """
    Agent that uses the Bread API to play the qubit game.
    
    The agent receives observations from the game environment and uses
    the Bread model to decide which quantum gate to apply.
    """
    
    GATE_NAMES = {
        0: "Identity (I)",
        1: "Pauli-X",
        2: "Pauli-Y", 
        3: "Pauli-Z",
        4: "Hadamard (H)"
    }
    
    def __init__(
        self,
        model_name: str = "johndoe/yoda_repo/yoda_bake/21",
        api_key: Optional[str] = None,
        base_url: str = "https://bapi.bread.com.ai/v1/chat/completions",
        temperature: float = 0.3,
        enable_thinking: bool = False
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("BREAD_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        
        if not self.api_key:
            raise ValueError(
                "BREAD_API_KEY not found. Set it via environment variable or pass api_key parameter."
            )
    
    def _build_game_prompt(
        self,
        observation: np.ndarray,
        target_state: np.ndarray,
        step_count: int,
        current_fidelity: float,
        history: List[Dict[str, Any]]
    ) -> str:
        """Build a prompt describing the current game state to the model."""
        
        # Convert observation back to complex state representation for clarity
        psi_0 = complex(observation[0], observation[1])
        psi_1 = complex(observation[2], observation[3])
        
        target_0 = complex(target_state[0].real, target_state[0].imag)
        target_1 = complex(target_state[1].real, target_state[1].imag)
        
        history_str = ""
        if history:
            history_str = "\nPrevious actions:\n"
            for h in history[-5:]:  # Show last 5 actions
                history_str += f"  Step {h['step']}: {self.GATE_NAMES[h['action']]} -> fidelity {h['fidelity']:.4f}\n"
        
        prompt = f"""You are playing a quantum gate game. Your goal is to transform the current qubit state to match the target state.

Current State: |ψ⟩ = ({psi_0:.4f})|0⟩ + ({psi_1:.4f})|1⟩
Target State:  |φ⟩ = ({target_0:.4f})|0⟩ + ({target_1:.4f})|1⟩

Current Fidelity: {current_fidelity:.4f} (need ≥ 0.99 to win)
Step: {step_count}/20
{history_str}

Available Actions:
0: Identity (I) - No change
1: Pauli-X - Bit flip (swaps |0⟩ and |1⟩)
2: Pauli-Y - Bit+phase flip
3: Pauli-Z - Phase flip (adds π phase to |1⟩)
4: Hadamard (H) - Creates superposition

Think about which gate will bring the current state closer to the target.
Respond with ONLY a single digit (0-4) representing your chosen action."""
        
        return prompt
    
    def choose_action(
        self,
        observation: np.ndarray,
        target_state: np.ndarray,
        step_count: int,
        current_fidelity: float,
        history: List[Dict[str, Any]]
    ) -> tuple[int, str]:
        """
        Use the Bread model to choose an action.
        
        Returns:
            action: int, the chosen gate index
            reasoning: str, the model's response/reasoning
        """
        prompt = self._build_game_prompt(
            observation, target_state, step_count, current_fidelity, history
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quantum computing expert playing a gate application game. "
                               "Respond concisely with your action choice."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 100,
            "stream": False
        }
        
        if not self.enable_thinking:
            payload["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            reasoning = result["choices"][0]["message"]["content"].strip()
            
            # Extract action from response
            action = self._parse_action(reasoning)
            return action, reasoning
            
        except requests.exceptions.RequestException as e:
            # Fallback to random action on API error
            print(f"API Error: {e}, using random action")
            return np.random.randint(5), f"[API Error: {e}]"
    
    def _parse_action(self, response: str) -> int:
        """Extract action number from model response."""
        # Try to find a digit 0-4 in the response
        for char in response:
            if char.isdigit() and int(char) in range(5):
                return int(char)
        
        # If no valid action found, use random
        return np.random.randint(5)
    
    def play_episode(
        self,
        env: QubitGameEnv,
        episode_id: int = 0,
        seed: Optional[int] = None,
        verbose: bool = True
    ) -> GameLog:
        """
        Play a complete episode of the game.
        
        Returns:
            GameLog containing the full episode history
        """
        obs, info = env.reset(seed=seed)
        
        log = GameLog(
            episode_id=episode_id,
            initial_state=obs.tolist(),
            target_state=[
                env.target_state[0].real, env.target_state[0].imag,
                env.target_state[1].real, env.target_state[1].imag
            ]
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode_id} - Initial fidelity: {info['fidelity']:.4f}")
            print(f"{'='*60}")
        
        done = False
        truncated = False
        history: List[Dict[str, Any]] = []
        
        while not (done or truncated):
            action, reasoning = self.choose_action(
                obs, env.target_state, info["step"], info["fidelity"], history
            )
            
            obs, reward, done, truncated, info = env.step(action)
            
            step_log = {
                "step": info["step"],
                "action": action,
                "action_name": self.GATE_NAMES[action],
                "fidelity": info["fidelity"],
                "reward": reward,
                "observation": obs.tolist()
            }
            
            log.steps.append(step_log)
            log.model_reasoning.append(reasoning)
            log.total_reward += reward
            history.append(step_log)
            
            if verbose:
                print(f"Step {info['step']:2d}: {self.GATE_NAMES[action]:12s} -> "
                      f"fidelity={info['fidelity']:.4f}, reward={reward:.4f}")
        
        log.solved = info.get("solved", False)
        log.final_fidelity = info["fidelity"]
        
        if verbose:
            status = "✓ SOLVED!" if log.solved else "✗ Not solved"
            print(f"\n{status} Final fidelity: {log.final_fidelity:.4f}, "
                  f"Total reward: {log.total_reward:.4f}")
        
        return log


def play_games(
    num_episodes: int = 5,
    model_name: Optional[str] = None,
    seed: Optional[int] = None
) -> List[GameLog]:
    """
    Play multiple games and return logs for all episodes.
    
    Args:
        num_episodes: Number of games to play
        model_name: Bread model path to use
        seed: Random seed for reproducibility
        
    Returns:
        List of GameLog objects for all episodes
    """
    config = QubitGameConfig(max_steps=20)
    env = QubitGameEnv(config)
    
    agent_kwargs = {}
    if model_name:
        agent_kwargs["model_name"] = model_name
    
    agent = BreadGameAgent(**agent_kwargs)
    
    logs = []
    for i in range(num_episodes):
        ep_seed = seed + i if seed is not None else None
        log = agent.play_episode(env, episode_id=i, seed=ep_seed)
        logs.append(log)
    
    return logs


if __name__ == "__main__":
    # Demo: play 3 episodes
    print("🎮 Bread Agent playing Qubit Game")
    print("=" * 60)
    
    logs = play_games(num_episodes=3)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    solved_count = sum(1 for log in logs if log.solved)
    avg_fidelity = sum(log.final_fidelity for log in logs) / len(logs)
    avg_reward = sum(log.total_reward for log in logs) / len(logs)
    
    print(f"Episodes played: {len(logs)}")
    print(f"Episodes solved: {solved_count}/{len(logs)}")
    print(f"Average final fidelity: {avg_fidelity:.4f}")
    print(f"Average total reward: {avg_reward:.4f}")
