"""
Adaptive opponent policy that can be optimized to achieve specific divergence targets.
Uses a simple neural network or parametric function that can be tuned via gradient descent.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
from .opponent_policies import OpponentPolicy, OpponentPolicyConfig


@dataclass  
class AdaptiveOpponentConfig(OpponentPolicyConfig):
    """Config for adaptive opponent that can be optimized for divergence targets."""
    family: str = "adaptive"
    target_kind: str = "state_action"  # "state_action" or "policy" 
    target_divergence: float = 0.12
    constraint_kinds: list = None  # e.g., ["policy"] for state_action shift
    constraint_level: float = 0.05
    # Network parameters
    hidden_dim: int = 64
    learning_rate: float = 0.01
    adaptation_steps: int = 50


class AdaptiveOpponent(OpponentPolicy):
    """
    Opponent policy with learnable parameters that can be optimized 
    to achieve specific distribution shift targets.
    """
    
    def __init__(self, cfg: AdaptiveOpponentConfig, agents: int, arena_size: Tuple[float, float], seed: int):
        super().__init__(cfg, agents, arena_size, seed)
        self.acfg = cfg
        
        # Initialize learnable parameters for a simple MLP
        # Input: flattened window state, Output: action offsets for all agents
        self.window_size = 6  # Should match generator window
        self.teams = 2
        self.input_dim = self.window_size * self.teams * agents * 2  # [W, teams, agents, 2] flattened
        self.output_dim = agents * 2  # [agents, 2] action deltas
        
        # Simple 2-layer MLP: input -> hidden -> output
        self.W1 = self.rng.normal(0, 0.1, size=(self.input_dim, cfg.hidden_dim))
        self.b1 = np.zeros(cfg.hidden_dim)
        self.W2 = self.rng.normal(0, 0.1, size=(cfg.hidden_dim, self.output_dim))
        self.b2 = np.zeros(self.output_dim)
        
        # Base random mapping for initialization
        self.base_prototypes = self.rng.normal(size=(1024, agents, 2))
        n = np.linalg.norm(self.base_prototypes, axis=-1, keepdims=True)
        self.base_prototypes = (self.base_prototypes / np.maximum(n, 1e-6)).astype(float)
        
    def _forward(self, win_pos: np.ndarray) -> np.ndarray:
        """Forward pass through the MLP to get action deltas."""
        # Flatten and normalize input
        x = win_pos.flatten()
        w, h = self.arena
        # Normalize positions to [0,1]
        x = x.reshape(-1, 2)
        x[:, 0] = x[:, 0] / max(1e-6, w)
        x[:, 1] = x[:, 1] / max(1e-6, h)
        x = x.flatten()
        
        # Pad or truncate to expected input dimension
        if len(x) > self.input_dim:
            x = x[:self.input_dim]
        elif len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
            
        # MLP forward pass
        h = np.tanh(x @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        
        # Reshape to [agents, 2] and scale
        return out.reshape(self.agents, 2) * 0.5  # Scale action deltas
        
    def step(self, snap: Dict[str, Any], selected_team: int, team_idx: int, t: int) -> np.ndarray:
        """Generate actions for this opponent team."""
        # Get base action from random mapping
        if "win_pos" in snap:
            win_pos = snap["win_pos"]
            # Simple hash for base action
            h = hash(tuple(win_pos.flatten()[:100])) % len(self.base_prototypes)
            base_action = self.base_prototypes[h]
        else:
            base_action = self.base_prototypes[0]
            
        # Add learnable delta
        if "win_pos" in snap:
            delta = self._forward(snap["win_pos"])
            action = base_action + delta
        else:
            action = base_action
            
        # Add noise if stochastic
        if self.cfg.stochastic and self.cfg.noise_sigma > 0:
            action = action + self.rng.normal(scale=self.cfg.noise_sigma, size=action.shape)
            
        return action
        
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get all learnable parameters."""
        return {
            "W1": self.W1,
            "b1": self.b1, 
            "W2": self.W2,
            "b2": self.b2
        }
        
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """Set learnable parameters."""
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()
        
    def describe(self) -> Dict[str, Any]:
        """Describe this opponent policy."""
        return {
            "id": self.id,
            "family": "adaptive",
            "target_kind": self.acfg.target_kind,
            "target_divergence": self.acfg.target_divergence,
            "constraint_level": self.acfg.constraint_level,
            "hidden_dim": self.acfg.hidden_dim,
            "learning_rate": self.acfg.learning_rate,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }


def build_adaptive_opponent(cfg: AdaptiveOpponentConfig, agents: int, arena_size: Tuple[float, float], seed: int) -> AdaptiveOpponent:
    """Factory function to build adaptive opponent."""
    return AdaptiveOpponent(cfg, agents, arena_size, seed)


