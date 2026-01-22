# SR-ADMM Execution Trace

This document maps the mathematical steps of SR-ADMM to the specific code execution in `SR-ADMM.py` for the first two iterations.

## Problem Components
- **Input**: Worker $i$ has $(A_i, b_i)$.
- **Goal**: Find $x$ minimizing $\sum \frac{1}{2}||A_ix - b_i||^2$.
- **Variables**:
    - $x_i$: Worker's local solution.
    - $y_i$: Worker's Lagrange multiplier (penalty weight).
    - $z$: Global Consensus model.
    - $s_i$: Compressed message ($x_i + y_i/\rho$).

---

## Step 0: Initialization
**Code**: `main()` $\to$ `SimulationEnv.__init__` $\to$ `Worker.__init__`
- **Central**: `global_z` ($z^0$) = 0 vector. `global_k` = 0.
- **Workers**: $x_i^0 = 0$, $y_i^0 = 0$, `last_z` = 0.

---

## Step 1: Iteration 1 ($k=1$)

### 1.1 Central Starts Round
**Code**: `start_new_iteration()`
- Increments `global_k` to **1**.
- Broadcasts current `global_z` ($z^0=0$) to all workers.
- Effectively calls: `worker.local_step(z_global=0, k_global=1)`.

### 1.2 Worker Computation (Local Step)
**Code**: `Worker.local_step(z_global=0, k_global=1)`

1.  **Sync & Dual Update**:
    - Checks `k_global (1) > self.current_k (0)`.
    - Updates `last_z` $\leftarrow 0$.
    - **Math**: $y_i^0 \leftarrow y_i^{-1} + \rho(x_i^0 - z^0)$ (Conceptually).
    - **Code check**: `self.y` starts at 0, updates with $(x=0, z=0)$, remains **0**.

2.  **Primal Update ($x$-Update)**:
    - Solve Ridge Regression centered at $z^0 - y_i/\rho$.
    - **Math**: $x_i^1 = \text{argmin}_x \left( \frac{1}{2}||A_ix - b_i||^2 + \frac{\rho}{2}||x - z^0 + y_i^0/\rho||^2 \right)$
    - **Code**: `target = Atb + rho*last_z - y` (Here `last_z`=0, `y`=0)
    - `self.x` becomes the solution $x_i^1 \approx (A^TA + \rho I)^{-1} A^T b$.

3.  **Message Generation**:
    - **Math**: $s_i^1 = x_i^1 + y_i^1 / \rho$.
    - **Code**: `s = self.x + self.y/self.rho`. (Since $y=0$, $s_i^1 = x_i^1$).

4.  **Send**: Returns `(s, 1)`.

### 1.3 Communication & Delay
**Code**: `SimulationEnv.run()`
- Each worker pushes `WORKER_DONE` event with a random finish time $t_{now} + \text{delay}$.
- Stragglers have large delay (e.g., +0.5s), normal workers small (+0.05s).

### 1.4 Aggregation (Straggler Resilience)
**Code**: `SimulationEnv.run()` event loop
- Events pop in order of finish time.
- Central collects $s_i^1$ into `collected_s`.
- **Condition**: Checks `len(contributors) >= n_min` (e.g., 60 workers).
- **Action**: Once 60th worker arrives, **STOP waiting** for others. Proceed to update.

### 1.5 Global Update ($z$-Update)
**Code**: `update_global_model()`
- **Math**: $z^1 = \frac{1}{|S|} \sum_{j \in S} s_j^1$.
- **Code**: `np.mean(valid_s, axis=0)`.
- **Note**: The 40 stragglers' data is IGNORED for this step (or older stale values used if available in history).

---

## Step 2: Iteration 2 ($k=2$)

### 2.1 Central Broadcast
**Code**: `start_new_iteration()`
- Increments `global_k` to **2**.
- Broadcasts **new** `global_z` ($z^1$).

### 2.2 Worker Computation
**Code**: `Worker.local_step(z_global=z^1, k_global=2)`

1.  **Sync & Dual Update**:
    - **Crucial Step**: Now `last_z` updates from $0$ to **$z^1$**.
    - **Math**: $y_i^1 = y_i^0 + \rho(x_i^1 - z^1)$.
    - **Code**: `self.y += rho * (self.x - self.last_z)`.
    - **Interpretation**: If my previous $x_i^1$ was far from the consensus $z^1$, I increase penalty $y_i$. This forces $x_i$ closer to $z$ across iterations.

2.  **Primal Update**:
    - **Math**: $x_i^2 = \text{argmin}_x \dots$ using new $z^1$ and updated $y_i^1$.
    - **Code**: `target = ... + rho*self.last_z - self.y`.
    - Worker solves regression again, but now pulled towards the new average $z^1$.

3.  Compute $s_i^2$ and return.

---

## Summary of Flow
1. **Central** sets the target ($z$).
2. **Workers** try to reach target while fitting their data ($x$), then report back ($s$).
3. **Dual Variable ($y$)** accumulates the error over time to force consensus.
4. **SR-ADMM Magic**: Steps 1.4 & 1.5 don't wait for everyone. We update $z$ with partial information, allowing the system to move 2-3x faster.
