#!/usr/bin/env python3
"""
Gradient-based Opponent Optimization for State+Action Distribution Shifts

This module implements gradient-based optimization of opponent policies to achieve
target Wasserstein distances for state and state+action distribution shifts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass

from .world import World
from .policies import PolicyManager
from .opponent_policies import OpponentPolicyManager
from .arenas import Arena


@dataclass
class OptimizationTarget:
    """Target for opponent optimization."""
    shift_kind: str  # 'state_only', 'state_action', or 'policy'
    target_divergence: float
    tolerance: float = 0.02
    max_iters: int = 100
    lr: float = 0.01


class DifferentiableOpponent(nn.Module):
    """Differentiable opponent policy for gradient-based optimization."""
    
    def __init__(self, arena_width: float, arena_height: float, noise_sigma: float = 0.08, shift_kind: str = "state_only"):
        super().__init__()
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.noise_sigma = noise_sigma
        self.shift_kind = shift_kind
        
        # Learnable parameters for opponent behavior
        self.state_bias = nn.Parameter(torch.zeros(2))  # Bias in state space
        self.action_bias = nn.Parameter(torch.zeros(2))  # Bias in action space
        self.noise_scale = nn.Parameter(torch.tensor(noise_sigma))
        
        # Additional parameters for more complex behaviors
        self.velocity_scale = nn.Parameter(torch.ones(2))
        self.position_scale = nn.Parameter(torch.ones(2))
        
        # State-correlation parameters for state+action shifts
        if shift_kind == "state_action":
            self.state_correlation_matrix = nn.Parameter(torch.eye(2) * 0.1)  # Controls state-action correlation
            self.response_intensity = nn.Parameter(torch.tensor(0.5))  # How strongly opponent responds to ego state
        
        # Policy-targeting parameters for policy shifts
        if shift_kind == "policy":
            self.policy_preference = nn.Parameter(torch.zeros(2))  # Preference for different policy types
        
    def forward(self, ego_positions: torch.Tensor, ego_velocities: torch.Tensor, 
                ego_policy_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Generate opponent actions based on ego state.
        
        Args:
            ego_positions: [B, T, agents, 2] ego positions
            ego_velocities: [B, T, agents, 2] ego velocities
            ego_policy_ids: [B, T, agents] ego policy IDs (for policy shifts)
            
        Returns:
            opponent_actions: [B, T, agents, 2] opponent actions
        """
        B, T, A, D = ego_positions.shape
        
        if self.shift_kind == "state_only":
            # Only modify state distribution via position bias
            state_response = ego_positions * self.position_scale + ego_velocities * self.velocity_scale
            state_response = state_response + self.state_bias.view(1, 1, 1, 2)
            actions = state_response * 0.1  # Weak coupling to maintain state shift focus
            
        elif self.shift_kind == "state_action":
            # Create non-random correlation between state and action
            state_response = ego_positions * self.position_scale + ego_velocities * self.velocity_scale
            state_response = state_response + self.state_bias.view(1, 1, 1, 2)
            
            # Apply state-correlation matrix to create structured state-action dependency
            correlated_response = torch.matmul(state_response, self.state_correlation_matrix)
            actions = correlated_response * self.response_intensity + self.action_bias.view(1, 1, 1, 2)
            
        elif self.shift_kind == "policy":
            # Respond differently based on ego policy to create policy distribution shift
            state_response = ego_positions * self.position_scale + ego_velocities * self.velocity_scale
            
            if ego_policy_ids is not None:
                # Create policy-dependent response
                policy_response = torch.zeros_like(state_response)
                policy_prefs = torch.softmax(self.policy_preference, dim=0)
                
                for pid in range(len(policy_prefs)):
                    mask = (ego_policy_ids == pid).float().unsqueeze(-1)
                    policy_influence = policy_prefs[pid] * 2.0 - 1.0  # Map [0,1] to [-1,1]
                    policy_response += mask * policy_influence * state_response
                
                actions = policy_response + self.action_bias.view(1, 1, 1, 2)
            else:
                # Fallback when no policy IDs available
                actions = state_response + self.action_bias.view(1, 1, 1, 2)
        
        else:
            # Default behavior
            state_response = ego_positions * self.position_scale + ego_velocities * self.velocity_scale
            actions = state_response + self.action_bias.view(1, 1, 1, 2)
        
        # Add noise
        if self.training:
            noise = torch.randn_like(actions) * self.noise_scale
            actions = actions + noise
        
        # Keep actions within reasonable bounds
        actions = torch.tanh(actions)  # Bounded in [-1, 1]
        
        return actions


class GradientOpponentOptimizer:
    """Optimizer for opponent policies using gradient descent."""
    
    def __init__(self, world_config: Dict, device: str = "cpu"):
        self.world_config = world_config
        self.device = device
        self.arena_width = world_config.get("arena", {}).get("width", 20.0)
        self.arena_height = world_config.get("arena", {}).get("height", 14.0)
        
    def create_differentiable_opponent(self, noise_sigma: float = 0.08, shift_kind: str = "state_only") -> DifferentiableOpponent:
        """Create a differentiable opponent model."""
        return DifferentiableOpponent(
            arena_width=self.arena_width,
            arena_height=self.arena_height,
            noise_sigma=noise_sigma,
            shift_kind=shift_kind
        ).to(self.device)
    
    def compute_wasserstein_distance(self, samples1: torch.Tensor, samples2: torch.Tensor) -> torch.Tensor:
        """
        Compute 1-Wasserstein distance between two sets of samples.
        
        Args:
            samples1: [N, D] first set of samples
            samples2: [M, D] second set of samples
            
        Returns:
            wasserstein_distance: scalar tensor
        """
        # Simplified 1-Wasserstein using sorted quantiles
        samples1_sorted = torch.sort(samples1.flatten())[0]
        samples2_sorted = torch.sort(samples2.flatten())[0]
        
        # Interpolate to same length for comparison
        n1, n2 = len(samples1_sorted), len(samples2_sorted)
        if n1 != n2:
            if n1 > n2:
                indices = torch.linspace(0, n2-1, n1).long()
                samples2_sorted = samples2_sorted[indices]
            else:
                indices = torch.linspace(0, n1-1, n2).long()
                samples1_sorted = samples1_sorted[indices]
        
        return torch.mean(torch.abs(samples1_sorted - samples2_sorted))
    
    def compute_js_divergence(self, dist1: torch.Tensor, dist2: torch.Tensor) -> torch.Tensor:
        """
        Compute Jensen-Shannon divergence between two discrete distributions.
        
        Args:
            dist1: [K] first distribution (probabilities)
            dist2: [K] second distribution (probabilities)
            
        Returns:
            js_divergence: scalar tensor
        """
        # Ensure distributions are normalized
        dist1 = torch.softmax(dist1, dim=0)
        dist2 = torch.softmax(dist2, dim=0)
        
        # Compute Jensen-Shannon divergence
        m = 0.5 * (dist1 + dist2)
        kl1 = torch.sum(dist1 * torch.log(dist1 / (m + 1e-8) + 1e-8))
        kl2 = torch.sum(dist2 * torch.log(dist2 / (m + 1e-8) + 1e-8))
        return 0.5 * (kl1 + kl2)
    
    def generate_episode_data(self, opponent: DifferentiableOpponent, num_episodes: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate episode data using the differentiable opponent.
        
        Returns:
            states: [episodes, timesteps, agents, 4] state trajectories (pos + vel)
            actions: [episodes, timesteps, agents, 2] action trajectories
            policy_ids: [episodes, timesteps, agents] policy ID trajectories
        """
        episodes_states = []
        episodes_actions = []
        episodes_policy_ids = []
        
        # Generate episodes using the differentiable opponent
        for _ in range(num_episodes):
            # Initialize random starting positions and velocities
            timesteps = self.world_config.get("steps", 40)
            agents = self.world_config.get("agents_per_team", 3)
            
            # Random initial state (requires_grad for state modification)
            initial_positions = torch.randn(agents, 2, device=self.device, requires_grad=True) * 2.0
            initial_velocities = torch.randn(agents, 2, device=self.device, requires_grad=True) * 0.5
            
            # Generate policy IDs (simple 2-policy mixture) 
            policy_ids = torch.randint(0, 2, (timesteps, agents), device=self.device)
            
            # Create state sequences by integrating opponent influence
            pos_list = [initial_positions]
            vel_list = [initial_velocities]
            
            # Generate sequence with opponent influence
            for t in range(1, timesteps):
                # Get opponent action for current state
                current_pos = pos_list[t-1].unsqueeze(0)  # [1, agents, 2]
                current_vel = vel_list[t-1].unsqueeze(0)  # [1, agents, 2]
                current_policy = policy_ids[t-1:t]  # [1, agents]
                
                opponent_action = opponent(current_pos.unsqueeze(0), current_vel.unsqueeze(0), 
                                         current_policy.unsqueeze(0)).squeeze(0).squeeze(0)  # [agents, 2]
                
                # Simple dynamics update influenced by opponent
                new_pos = pos_list[t-1] + vel_list[t-1] * 0.25 + opponent_action * 0.1
                new_vel = vel_list[t-1] * 0.9 + opponent_action * 0.2
                
                pos_list.append(new_pos)
                vel_list.append(new_vel)
            
            # Stack to create sequences
            positions = torch.stack(pos_list)  # [T, agents, 2]
            velocities = torch.stack(vel_list)  # [T, agents, 2]
            
            # Generate final actions for the whole sequence
            actions = opponent(positions.unsqueeze(0), velocities.unsqueeze(0), 
                             policy_ids.unsqueeze(0)).squeeze(0)
            
            episodes_states.append(torch.cat([positions, velocities], dim=-1))  # [T, A, 4]
            episodes_actions.append(actions)  # [T, A, 2]
            episodes_policy_ids.append(policy_ids)  # [T, A]
        
        states = torch.stack(episodes_states)  # [episodes, T, A, 4]
        actions = torch.stack(episodes_actions)  # [episodes, T, A, 2]
        policy_ids = torch.stack(episodes_policy_ids)  # [episodes, T, A]
        
        return states, actions, policy_ids
    
    def optimize_opponent(self, target: OptimizationTarget, baseline_states: torch.Tensor, 
                         baseline_actions: torch.Tensor, baseline_policy_ids: torch.Tensor = None) -> Dict:
        """
        Optimize opponent policy to achieve target distribution shift.
        
        Args:
            target: Optimization target specification
            baseline_states: [episodes, timesteps, agents, state_dim] baseline state data
            baseline_actions: [episodes, timesteps, agents, action_dim] baseline action data
            baseline_policy_ids: [episodes, timesteps, agents] baseline policy IDs (for policy shifts)
            
        Returns:
            optimization_result: Dictionary with optimized opponent and metrics
        """
        print(f"🔧 Optimizing opponent for {target.shift_kind} shift (target: {target.target_divergence:.3f})")
        
        # Create differentiable opponent specific to shift type
        opponent = self.create_differentiable_opponent(shift_kind=target.shift_kind)
        optimizer = optim.Adam(opponent.parameters(), lr=target.lr)
        
        # Convert baseline data to tensors (no gradients needed for baseline data)
        baseline_states = torch.tensor(baseline_states, dtype=torch.float32, device=self.device).detach()
        baseline_actions = torch.tensor(baseline_actions, dtype=torch.float32, device=self.device).detach()
        if baseline_policy_ids is not None:
            baseline_policy_ids = torch.tensor(baseline_policy_ids, dtype=torch.long, device=self.device).detach()
        
        best_loss = float('inf')
        best_opponent_state = None
        convergence_history = []
        
        for iteration in range(target.max_iters):
            optimizer.zero_grad()
            
            # Generate new data with current opponent
            new_states, new_actions, new_policy_ids = self.generate_episode_data(opponent, num_episodes=20)
            
            # Compute divergences based on shift type
            if target.shift_kind == "state_only":
                # Only optimize state distribution
                state_positions = new_states[:, :, :, :2]  # Extract positions
                baseline_positions = baseline_states[:, :, :, :2]
                
                # Ensure we're computing gradients w.r.t. the generated states
                divergence = self.compute_wasserstein_distance(
                    state_positions.reshape(-1, 2),
                    baseline_positions.reshape(-1, 2).detach()  # Detach baseline for clarity
                )
                
            elif target.shift_kind == "state_action":
                # Optimize both state and action distributions with non-random correlation
                state_positions = new_states[:, :, :, :2]
                baseline_positions = baseline_states[:, :, :, :2]
                
                state_div = self.compute_wasserstein_distance(
                    state_positions.reshape(-1, 2),
                    baseline_positions.reshape(-1, 2).detach()
                )
                
                action_div = self.compute_wasserstein_distance(
                    new_actions.reshape(-1, 2),
                    baseline_actions.reshape(-1, 2).detach()
                )
                
                # Combined divergence (weighted average)
                divergence = (state_div + action_div) / 2.0
                
            elif target.shift_kind == "policy":
                # Optimize policy distribution using JS divergence
                if baseline_policy_ids is not None:
                    # Compute policy distributions
                    baseline_policy_dist = torch.bincount(baseline_policy_ids.flatten(), minlength=2).float()
                    new_policy_dist = torch.bincount(new_policy_ids.flatten(), minlength=2).float()
                    
                    # Normalize to probabilities
                    baseline_policy_dist = baseline_policy_dist / baseline_policy_dist.sum()
                    new_policy_dist = new_policy_dist / new_policy_dist.sum()
                    
                    divergence = self.compute_js_divergence(new_policy_dist, baseline_policy_dist)
                else:
                    # Fallback: use simple entropy-based divergence
                    policy_entropy = -torch.sum(torch.softmax(opponent.policy_preference, dim=0) * 
                                               torch.log_softmax(opponent.policy_preference, dim=0))
                    divergence = torch.abs(policy_entropy - 0.693)  # Target: log(2) for balanced distribution
            
            else:
                raise ValueError(f"Unsupported shift kind: {target.shift_kind}")
            
            # Loss: minimize distance to target divergence
            loss = (divergence - target.target_divergence) ** 2
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track progress
            current_loss = loss.item()
            current_div = divergence.item()
            convergence_history.append({
                "iteration": iteration,
                "loss": current_loss,
                "divergence": current_div,
                "target": target.target_divergence,
                "shift_kind": target.shift_kind
            })
            
            # Check for improvement
            if current_loss < best_loss:
                best_loss = current_loss
                best_opponent_state = opponent.state_dict().copy()
            
            # Check convergence
            if abs(current_div - target.target_divergence) < target.tolerance:
                print(f"  ✅ Converged at iteration {iteration}: {current_div:.4f} (target: {target.target_divergence:.3f})")
                break
            
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: divergence={current_div:.4f}, target={target.target_divergence:.3f}, loss={current_loss:.6f}")
        
        else:
            print(f"  ⚠️  Max iterations reached. Final divergence: {current_div:.4f} (target: {target.target_divergence:.3f})")
        
        # Restore best opponent
        if best_opponent_state is not None:
            opponent.load_state_dict(best_opponent_state)
        
        return {
            "opponent_model": opponent,
            "final_divergence": current_div,
            "target_divergence": target.target_divergence,
            "converged": abs(current_div - target.target_divergence) < target.tolerance,
            "convergence_history": convergence_history,
            "optimization_target": target
        }
    
    def save_optimized_opponent(self, result: Dict, save_path: str):
        """Save optimized opponent model and metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": result["opponent_model"].state_dict(),
            "final_divergence": result["final_divergence"],
            "target_divergence": result["target_divergence"],
            "converged": result["converged"],
            "optimization_target": result["optimization_target"].__dict__,
            "convergence_history": result["convergence_history"]
        }, save_path)
        
        print(f"📁 Optimized opponent saved to {save_path}")


def optimize_opponents_for_targets(world_config: Dict, baseline_data: Dict, 
                                  targets: List[OptimizationTarget], save_dir: str) -> Dict:
    """
    Optimize opponents for multiple targets.
    
    Args:
        world_config: World configuration
        baseline_data: Baseline episode data for comparison
        targets: List of optimization targets
        save_dir: Directory to save optimized opponents
        
    Returns:
        optimization_results: Dict mapping target names to results
    """
    optimizer = GradientOpponentOptimizer(world_config)
    results = {}
    
    for target in targets:
        target_name = f"{target.shift_kind}_{int(target.target_divergence * 1000):03d}"
        print(f"\n🎯 Optimizing for {target_name}")
        
        result = optimizer.optimize_opponent(
            target=target,
            baseline_states=baseline_data["states"],
            baseline_actions=baseline_data["actions"],
            baseline_policy_ids=baseline_data.get("policy_ids")
        )
        
        # Save result
        save_path = os.path.join(save_dir, f"opponent_{target_name}.pt")
        optimizer.save_optimized_opponent(result, save_path)
        
        results[target_name] = result
    
    return results


if __name__ == "__main__":
    # Example usage
    world_config = {
        "arena": {"width": 20.0, "height": 14.0},
        "steps": 40,
        "agents_per_team": 3
    }
    
    # Example baseline data (normally loaded from actual episodes)
    baseline_data = {
        "states": np.random.randn(50, 40, 3, 4),  # [episodes, timesteps, agents, state_dim]
        "actions": np.random.randn(50, 40, 3, 2)  # [episodes, timesteps, agents, action_dim]
    }
    
    # Define optimization targets
    targets = [
        OptimizationTarget("state_only", 0.05),
        OptimizationTarget("state_only", 0.10),
        OptimizationTarget("state_action", 0.05),
        OptimizationTarget("state_action", 0.10),
    ]
    
    # Optimize opponents
    results = optimize_opponents_for_targets(world_config, baseline_data, targets, "optimized_opponents/")
    
    print("\n🎉 Opponent optimization completed!")
