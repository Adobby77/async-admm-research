import numpy as np
import matplotlib.pyplot as plt
from async_admm_paper import AsyncSimulator

def run_simulation(adaptive=True, label="Adaptive"):
    print(f"\n--- Running Simulation: {label} ---")
    np.random.seed(42)
    # Ring Topology, 6 Agents
    sim = AsyncSimulator(n_agents=6, horizon=10, prob_delay=0.9, max_delay=5)
    
    hist = {'p': [], 'd': [], 'j': [], 'rho': []}
    max_steps = 100 
    
    for k in range(max_steps):
        p, d, j, rho = sim.step(adaptive=adaptive)
        hist['p'].append(p)
        hist['d'].append(d)
        hist['j'].append(j)
        hist['rho'].append(rho) # Placeholder 0 for now
        
        if k % 10 == 0:
            print(f"Step {k}: Primal={p:.4f}, Dual={d:.4f}, Cost={j:.4f}")
            
    return hist, sim

def main():
    h_adap, sim_adap = run_simulation(adaptive=True, label="Proposed")
    h_std, sim_std = run_simulation(adaptive=False, label="Standard")
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Primal Residual
    plt.subplot(2, 3, 1)
    plt.plot(h_std['p'], 'r--', label='Standard')
    plt.plot(h_adap['p'], 'b-', label='Proposed')
    plt.title('Primal Residual (Consent Mismatch)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # 2. Dual Residual
    plt.subplot(2, 3, 2)
    plt.plot(h_std['d'], 'r--', label='Standard')
    plt.plot(h_adap['d'], 'b-', label='Proposed')
    plt.title('Dual Residual (Stability)')
    plt.yscale('log')
    plt.grid(True)
    
    # 3. Objective Function
    plt.subplot(2, 3, 3)
    plt.plot(h_std['j'], 'r--', label='Standard')
    plt.plot(h_adap['j'], 'b-', label='Proposed')
    plt.title('Total Cost/Energy')
    plt.grid(True)
    
    # 4. Adaptive Penalty
    plt.subplot(2, 3, 4)
    plt.plot(h_std['rho'], 'r--', label='Standard')
    plt.plot(h_adap['rho'], 'b-', label='Proposed')
    plt.title('Average Penalty Parameter (Rho)')
    plt.xlabel('Iteration')
    plt.ylabel('Rho value')
    plt.ylim(0, 1.2) # Rho base is 1.0
    plt.grid(True)
    plt.legend()
    
    # 5. Formation (Final)
    plt.subplot(2, 3, 5)
    colors = plt.cm.rainbow(np.linspace(0, 1, 6))
    
    for i, agent in enumerate(sim_adap.agents):
        # Plot Agent's OWN trajectory (x_i^i)
        traj = agent.x_self
        c = colors[i]
        plt.plot(traj[:, 0], traj[:, 1], color=c, linestyle='-', label=f'Ag {i}')
        plt.plot(traj[0, 0], traj[0, 1], color=c, marker='s')
        plt.plot(traj[-1, 0], traj[-1, 1], color=c, marker='o')
        
    # Plot Formation Shape
    finals = [a.x_self[-1] for a in sim_adap.agents]
    finals.append(finals[0])
    fx = [p[0] for p in finals]
    fy = [p[1] for p in finals]
    plt.plot(fx, fy, 'k:', alpha=0.5, label='Formation')
    
    plt.title('Final Trajectories (Proposed)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('async_admm_distributed.png')
    print("Saved to async_admm_distributed.png")

if __name__ == "__main__":
    main()
