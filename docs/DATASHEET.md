# Data Card / Datasheet

- Motivation: Study state-conditional, lag-aware multi-agent policies under adversarial coordination with crafted confounding where action-only baselines fail.
- Composition: 2 teams, 3 agents/team (configurable). 2–3 intents (escort_screen, decoy_flank, convoy). State is previous W frames of all teams; action is next-frame velocities for the selected team.
- Simulation: Continuous 2D arena with optional obstacles. Observation–action lag is fixed/random/state-dependent. Latent intent evolves via a Markov prior. Opponents use simple stochastic guard or best-response heuristics.
- Splits: IID train/val/test; OOD: lag-shifted, arena-held-out.
- Metrics: Next-frame ADE/FDE; Lag accuracy/NLL; Intent accuracy/NLL and clustering metrics (optional). Calibration suggested (ECE/Brier).
- Risks: Synthetic dynamics may not match complex real-world domains; care is needed when transferring conclusions. Avoid encoding spurious cues (IDs, ordering); we randomize seeds and can enforce symmetry.
- License: Data CC BY 4.0. Code Apache-2.0.
- Versioning: Semantic versions; see CHANGELOG.

