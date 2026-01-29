"""
Simple single-qubit "game" environment for RL agents.

Part of the baking-agents repo: https://github.com/Anupam-Anupam/baking-agents

The environment exposes a Gym-style API so an agent can:
    - observe the current qubit state
    - apply a discrete set of quantum gates as actions
    - receive a reward based on how close the state is to a target

Key properties:
    - Single qubit, represented as a normalized 2D complex state vector
    - Random initial state at the start of each episode
    - Discrete action space of common single-qubit gates
    - Episode terminates after max_steps or once close enough to target
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np


@dataclass
class QubitGameConfig:
    """
    Configuration for the QubitGameEnv.
    """

    # Maximum number of steps per episode
    max_steps: int = 20

    # Target state to "reach". By default this is |0> = [1, 0]^T
    # You can override this with any normalized 2D complex vector.
    target_state: np.ndarray = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    # How close (in fidelity) we need to be to consider the episode "solved"
    solve_fidelity_threshold: float = 0.99

    # Reward shaping parameters
    reward_scale: float = 1.0
    step_penalty: float = 0.01  # small penalty per step to encourage shorter solutions


class QubitGameEnv:
    """
    Simple, self-contained single-qubit environment.

    The interface intentionally mirrors the core of a Gymnasium environment:

        env = QubitGameEnv()
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)  # or np.random.randint(env.n_actions)
            obs, reward, done, truncated, info = env.step(action)

    Observation:
        A 4D real-valued vector representing the real and imaginary parts
        of the 2D complex state vector:
            [Re(psi0), Im(psi0), Re(psi1), Im(psi1)]

    Actions (discrete):
        0: Identity (I)
        1: Pauli-X
        2: Pauli-Y
        3: Pauli-Z
        4: Hadamard (H)
    """

    def __init__(self, config: QubitGameConfig | None = None) -> None:
        self.config = config or QubitGameConfig()

        # Basic validation of target state
        self.target_state = self._normalize(self.config.target_state.astype(np.complex128))

        # Define gate matrices (2x2 complex)
        self.gates: Dict[int, np.ndarray] = {
            0: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),  # I
            1: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),  # X
            2: np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),  # Y
            3: np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),  # Z
            4: (1 / np.sqrt(2.0))
            * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128),  # H
        }

        self.n_actions: int = len(self.gates)

        # Internal state
        self._state: np.ndarray | None = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core environment API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to a random initial state.

        Returns:
            obs: np.ndarray of shape (4,) - real-valued observation
            info: dict with metadata (e.g., initial fidelity)
        """
        if seed is not None:
            # Seed numpy's RNG locally for reproducibility if desired
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Random normalized 2D complex vector
        real = rng.normal(size=2)
        imag = rng.normal(size=2)
        psi = real + 1j * imag
        self._state = self._normalize(psi)
        self._step_count = 0

        obs = self._state_to_obs(self._state)
        fidelity = self._fidelity(self._state, self.target_state)
        info = {"fidelity": fidelity, "step": self._step_count}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply a gate (action) to the current state.

        Returns:
            obs: np.ndarray, next observation
            reward: float, scalar reward
            terminated: bool, True if episode ended because of success/failure
            truncated: bool, True if episode ended because max_steps reached
            info: dict with diagnostic info
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if action not in self.gates:
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.n_actions - 1}].")

        self._step_count += 1

        # Evolve state
        gate = self.gates[action]
        new_state = gate @ self._state
        new_state = self._normalize(new_state)
        self._state = new_state

        fidelity = self._fidelity(self._state, self.target_state)

        # Reward: scaled fidelity minus small step penalty
        reward = self.config.reward_scale * float(fidelity) - self.config.step_penalty

        # Termination conditions
        solved = fidelity >= self.config.solve_fidelity_threshold
        truncated = self._step_count >= self.config.max_steps
        terminated = solved

        obs = self._state_to_obs(self._state)
        info = {
            "fidelity": fidelity,
            "step": self._step_count,
            "solved": solved,
            "action": action,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(psi: np.ndarray) -> np.ndarray:
        """Normalize a complex 2-vector."""
        norm = np.linalg.norm(psi)
        if norm == 0.0:
            # If we somehow get a zero vector, default to |0>
            return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        return psi / norm

    @staticmethod
    def _fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
        """
        Compute state fidelity |<phi|psi>|^2 between two normalized states.
        """
        overlap = np.vdot(phi, psi)  # <phi|psi>
        return float(np.abs(overlap) ** 2)

    @staticmethod
    def _state_to_obs(psi: np.ndarray) -> np.ndarray:
        """
        Convert complex 2-vector to 4D real vector observation.
        """
        return np.array(
            [psi[0].real, psi[0].imag, psi[1].real, psi[1].imag],
            dtype=np.float32,
        )


if __name__ == "__main__":
    # Minimal example of random interaction with the environment.
    env = QubitGameEnv()
    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Initial fidelity:", info["fidelity"])

    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        # Random policy just for demonstration
        action = np.random.randint(env.n_actions)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={info['step']}, action={info['action']}, "
            f"fidelity={info['fidelity']:.4f}, reward={reward:.4f}"
        )

    print("Episode finished. Total reward:", total_reward)

