# Walkthrough - Fully Distributed Async ADMM (Ring Topology)

I have implemented the **True Distributed P2P** version of the algorithm, moving away from the centralized simulator.

## Implementation Changes (Distributed Logic)
- **Topology**: Undirected Ring (Connection $i \leftrightarrow i+1$).
- **Local Optimization**: Agent $i$ optimizes not just $x_i$, but also *proposals* for neighbors $x_{i-1}, x_{i+1}$.
- **Message Passing**:
    - **L2G**: Agent $i \to$ Neighbors (Proposals $x_j^i$).
    - **G2L**: Agent $j \to$ Predecessors (Consensus $z_j$).
- **Consensus**: $z_j$ is computed locally by Agent $j$ by averaging received proposals.

## Results: Ring Topology (Stress Test)

6 Agents, Hexagon Formation, Max Delay 5, Prob 90%.

![Distributed ADMM Full Results](/home/sriv/.gemini/antigravity/brain/0a70ce2c-1475-4383-ad64-915702d87cf4/async_admm_distributed_full.png)

### Key Observations
1.  **Cost Efficiency (Middle Top)**:
    - **Proposed (Blue)**: Cost drops to near zero (~0.07).
    - **Standard (Red)**: Cost stays high (~14.5).
    - **Analysis**: This is the "Aha!" moment. In a distributed setting with delays, Standard ADMM forces agents to agree on "average" positions that might be energy-inefficient zig-zags. **Adaptive ADMM relaxes the constraints**, allowing agents to smooth out their trajectories (lower $u^2$ energy) instead of frantically chasing noisy neighbors.

2.  **Adaptive Rho (Bottom Left)**:
    - **Standard (Red)**: Fixed at 1.0.
    - **Proposed (Blue)**: Fluctuates between 0.6 and 0.8. This variation reflects the random delays in the network. By lowering $\rho$ when data is stale, the algorithm "trusts" the noisy neighbor data less, preventing the high-energy oscillations seen in the Standard method.

3.  **Residuals & Formation**:
    - Primal Residual converges similarly for both.
    - Dual Residual is more stable for Standard, but this rigidity is exactly what causes the high energy cost.
    - Perfect Hexagon formation achieved (Bottom Middle).

## Code Availability
- **Source Code**: [`async_admm_paper.py`](file:///home/sriv/async_admm_paper.py) (Rewrite)
- **Demo Script**: [`demo_async_paper.py`](file:///home/sriv/demo_async_paper.py)
