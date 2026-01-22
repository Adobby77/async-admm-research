# Implementation Plan: Distributed Async ADMM (Ring Topology)

## 1. Network Topology
- **6 Agents (Hexagon Formation)**
- **Topology**: Ring (Circular)
    - $\mathcal{N}_i = \{ (i-1)\%6, (i+1)\%6 \}$
    - $\mathcal{P}_i = \{ (i-1)\%6, (i+1)\%6 \}$ (Undirected Ring)
    - Agent $i$ optimizes its own trajectory $x_i$ AND proposes trajectories for neighbors $x_{i-1}, x_{i+1}$.

## 2. Class Structure Update

### `Agent` Class
- **State**:
    - `self.x`: My own trajectory plan ($x_i^i$).
    - `self.u`: My own control plan.
    - `self.neighbor_proposals`: Dictionary `{j: x_j^i}`. Trajectories I propose for my neighbors $j \in \mathcal{N}_i$.
    - `self.y`: Dictionary `{j: y_j^i}`. Dual variables for each neighbor $j$ (and maybe self?).
    - `self.z_received`: Dictionary `{j: z_j}`. The "Consensus" trajectory received from neighbor $j$ (where $j$ is the master of $z_j$).
    - `self.rho`: Adaptive penalties for each neighbor link.

- **Local Optimization**:
    - Variables: $x_i, u_i, \{x_j\}_{j \in \mathcal{N}_i}, \{u_j\}_{j \in \mathcal{N}_i}$.
    - Cost:
        - $J_i(x_i, u_i)$ (My local cost)
        - $+ \sum_{j \in \mathcal{N}_i \cup \{i\}} \frac{\rho}{2} || x_j^i - z_j + y_j^i/\rho ||^2$
    - Note: $z_j$ is the *delayed* consensus variable received from $j$. $z_i$ is my own previous consensus.

### `AsyncSimulator` Class
- **Message Queues**:
    - `L2G_queue`: Messages $i \to j$ containing proposals $(x_j^i)$. Delay $d^{LG}$.
    - `G2L_queue`: Messages $j \to i$ containing consensus $(z_j)$. Delay $d^{GL}$.
- **Consensus Step (Global Update)**:
    - Strictly local to each agent.
    - Agent $i$ collects proposals $\{ x_i^k \}_{k \in \mathcal{P}_i \cup \{i\}}$.
    - computes $z_i = \text{WeightedAvg}(x_i^k)$.

## 3. Visualization
- **Plot 1**: Primal Residual $\sum ||x_j^i - z_j||$.
- **Plot 2**: Dual Residual $\sum ||z_i^{new} - z_i^{old}||$.
- **Plot 3**: Objective Function Value $\sum J_i$.
- **Plot 4**: Adaptive $\rho$ evolution (Average $\rho$).
- **Plot 5**: Formation Snapshots or Final Trajectory.

## 4. Execution Steps
1. Rewrite `async_admm_paper.py`.
2. Rewrite `demo_async_paper.py`.
3. Run and Verify.
