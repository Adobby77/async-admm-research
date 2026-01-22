import numpy as np
from scipy.optimize import minimize

class Agent:
    def __init__(self, agent_id, neighbors, predecessors, horizon, state_dim, control_dim, 
                 x_init, formation_offsets, rho_base=1.0, mu_base=1.0):
        """
        neighbors (N_i): Agents I propose trajectories FOR.
        predecessors (P_i): Agents who propose trajectories FOR ME.
        formation_offsets: Dict {id: offset_vector}. Target = z + offset_i.
        """
        self.id = agent_id
        self.N_i = neighbors         # IDs of agents I optimize for (j in N_i)
        self.P_i = predecessors      # IDs of agents who optimize for me (k in P_i)
        self.K = horizon
        self.Hx = state_dim
        self.Hu = control_dim
        self.rho_base = rho_base
        self.mu_base = mu_base
        self.offsets = formation_offsets

        # 1. Internal Variables (Proposals)
        # x_self: My plan for myself (x_i^i)
        self.x_self = np.tile(x_init, (self.K + 1, 1))
        self.u_self = np.zeros((self.K, self.Hu))
        
        # x_proposals: My plan for local neighbors (x_j^i for j in N_i)
        self.x_proposals = {j: np.tile(x_init, (self.K + 1, 1)) for j in self.N_i}
        self.u_proposals = {j: np.zeros((self.K, self.Hu)) for j in self.N_i} # Optional if we optimize u_j too
        
        # 2. Consensus Variables (The "Truth" I own)
        # z_self: My consensus trajectory (z_i). Aggregated from P_i.
        self.z_self = np.tile(x_init, (self.K + 1, 1))
        
        # 3. Dual Variables
        # y: Dual for mismatch between my proposals and their owners' z.
        # Key is 'j'. y_j^i corresponds to constraint x_j^i = z_j.
        # Includes j=self (x_i^i = z_i).
        self.y = {j: np.zeros((self.K + 1, self.Hx)) for j in self.N_i}
        self.y[self.id] = np.zeros((self.K + 1, self.Hx))
        
        # 4. Received Data (Delayed Views)
        # z_received: The z_j values I received from my neighbors j in N_i
        self.z_received = {j: np.tile(x_init, (self.K+1, 1)) for j in self.N_i}
        self.z_delays = {j: 0 for j in self.N_i}
        
        # proposals_received: The x_i^k values I received from predecessors k in P_i
        # Used to compute my z_self
        self.proposals_received = {k: np.tile(x_init, (self.K+1, 1)) for k in self.P_i}
        self.proposals_delays = {k: 0 for k in self.P_i}
        
    def local_optimization(self, adaptive=True, current_time=0):
        """
        Solve local optimization for x_i^i and {x_j^i}_{j in N_i}.
        Min J_i(x_i, u_i) + Sum_{j in N_i U {i}} ADMM_Penalty
        """
        # Variables to optimize: x_self, u_self, AND x_proposals, u_proposals
        # To simplify, we only assume cost J_i depends on x_i, u_i.
        # The variables x_j^i are only driven by the ADMM constraint x_j^i = z_j.
        # If J_i involves neighbors (collision avoidance), x_j^i matters for cost.
        # In this Formation task, J_i is just "Min Energy".
        # But we must optimize x_j^i explicitly to track z_j.
        
        # Construct decision vector
        # Order: [x_i, u_i, x_n1, u_n1, x_n2, u_n2, ...]
        targets = [self.id] + self.N_i
        
        # Flatten initial guess
        guess = []
        for agent_id in targets:
            if agent_id == self.id:
                guess.extend(self.x_self.flatten())
                guess.extend(self.u_self.flatten())
            else:
                guess.extend(self.x_proposals[agent_id].flatten())
                guess.extend(self.u_proposals[agent_id].flatten()) # We carry u for completeness
        guess = np.array(guess)
        
        # Helper indices
        dim_x = (self.K + 1) * self.Hx
        dim_u = (self.K) * self.Hu
        block_size = dim_x + dim_u
        
        def objective(vars_flat):
            cost = 0.0
            
            # Iterate over each optimized agent variable
            for idx, agent_id in enumerate(targets):
                offset = idx * block_size
                x_flat = vars_flat[offset : offset + dim_x]
                u_flat = vars_flat[offset + dim_x : offset + block_size]
                
                x_vals = x_flat.reshape(self.K + 1, self.Hx)
                u_vals = u_flat.reshape(self.K, self.Hu)
                
                # 1. Lagrangian / Energy Cost
                # Only "I" (agent_id == self.id) have a real cost J_i.
                # Neighbors' trajectories x_j^i don't have inherent cost in J_i unless coupled.
                # Here: Min Energy for everyone I propose?
                # Usually: J_i(x_i, u_i). Neighbors are auxiliary. 
                # Let's assign simple smoothing cost to proposals too.
                R = np.eye(self.Hu) * 0.1
                for k in range(self.K):
                    cost += 0.5 * u_vals[k] @ R @ u_vals[k]
                    
                # 2. ADMM Penalty
                # || x_target^i - z_target + y_target^i/rho ||^2
                # Note: "z_target" is what I received from 'agent_id'.
                # But wait, if agent_id == self.id, I use z_self? 
                # NO. Standard ADMM: x_i^i tracks z_i. z_i is "global" Consensus.
                # Here z_i is computed in STEP 3. So in STEP 1, we treat z_i as fixed parameter (from prev iter).
                
                if agent_id == self.id:
                    # Special handling: My own Z from last step
                    z_ref = self.z_self 
                    delay = 0 # My own memory
                else:
                     z_ref = self.z_received[agent_id]
                     delay = self.z_delays[agent_id]
                
                y_dul = self.y[agent_id]

                if adaptive:
                    rho = self.rho_base / (1.0 + delay)
                else:
                    rho = self.rho_base
                
                # Formation Logic:
                # The "z" we agree on is the *Center*.
                # So x_agent should equal z + offset_agent.
                # Relation: x - offset = z.
                
                target_pos = z_ref + self.offsets[agent_id]
                
                diff = x_vals - target_pos
                norm_term = diff + y_dul / rho
                cost += 0.5 * rho * np.sum(norm_term**2)
                
            return cost

        # Constraints (Dynamics for each proposed trajectory)
        constraints = []
        dt = 0.1
        
        for idx, agent_id in enumerate(targets):
             offset = idx * block_size
             
             # Start Pos Constraint
             # If it's ME, I must start at my true start.
             # If it's neighbour, I should assume they start at THEIR start? 
             # I might not know their start.
             # Assumption: I know their start_pos or I use my previous estimate.
             # Let's fix start pos to "current x[0]" to avoid jumps.
             
             start_val = self.x_self[0] if agent_id == self.id else self.x_proposals[agent_id][0]
             
             def init_constraint(vars_flat, idx=idx, s_val=start_val):
                 off = idx * block_size
                 x_v = vars_flat[off : off + dim_x].reshape(self.K+1, self.Hx)
                 return x_v[0] - s_val
             constraints.append({'type': 'eq', 'fun': init_constraint})
             
             # Dynamics
             for k in range(self.K):
                 def dyn(vars_flat, idx=idx, k=k):
                     off = idx * block_size
                     x_v = vars_flat[off : off + dim_x].reshape(self.K+1, self.Hx)
                     u_v = vars_flat[off + dim_x : off + block_size].reshape(self.K, self.Hu)
                     return x_v[k+1] - (x_v[k] + u_v[k] * dt)
                 constraints.append({'type': 'eq', 'fun': dyn})

        # SOLVE
        res = minimize(objective, guess, constraints=constraints, method='SLSQP', options={'disp': False, 'maxiter': 20})
        
        # Unpack results
        for idx, agent_id in enumerate(targets):
            offset = idx * block_size
            x_res = res.x[offset : offset + dim_x].reshape(self.K+1, self.Hx)
            u_res = res.x[offset + dim_x : offset + block_size].reshape(self.K, self.Hu)
            
            if agent_id == self.id:
                self.x_self = x_res
                self.u_self = u_res
            else:
                self.x_proposals[agent_id] = x_res
                self.u_proposals[agent_id] = u_res
        
        # Return my J_i for metrics
        energy = 0.5 * np.sum(self.u_self**2)
        return energy

    def update_consensus(self, adaptive=True):
        """
        Step 3: Calculate z_i (My consensus)
        z_i = WeightedAvg( [x_i^i, x_i^j, x_i^k ...] )
        """
        num = np.zeros_like(self.z_self)
        den = 0.0
        
        # 1. My own proposal x_i^i
        # rho for self is base (delay 0)
        rho_self = self.rho_base
        # My proposal checks out against z_i. 
        # Standard ADMM Z-update: z = avg(x + y/rho).
        # We need to include y_i^i in the average!
        num += rho_self * (self.x_self - self.offsets[self.id]) + self.y[self.id]
        den += rho_self
        
        # 2. Predecessors' inputs x_i^j (stored in self.proposals_received)
        # They don't send y. The "y" is local to them?
        # Wait, in Consensus ADMM, Z-update aggregates x. 
        # The dual variable y is associated with the constraint x-z=0.
        # Who performs the dual update? The *proposer*.
        # So "I" (the consensus owner) just average the "x"s described by others.
        # But wait, to be exact equivalent to solving Min Sum L_i,
        # Z minimizes: Sum_j { rho/2 || x_i^j - z_i + y_i^j/rho ||^2 }
        # -> z_i = (Sum rho(x+y/rho)) / Sum rho
        # But I don't know y_i^j (stored at j).
        # Standard Distributed ADMM split:
        # Node i updates x_i.
        # Node i updates y_i.
        # Consensus handled by... exchange?
        
        # In this Formation paper:
        # Eq 23: z_l^(k+1) = sum_{j in P_l} ...
        # The paper specifically says z is updated using received (x, u).
        # It does NOT mention receiving y.
        # So it's likely just averaging the PROPOSALS.
        # The dual term "y" is handled in Local Optimization (Step 1) and Dual Update (Step 5).
        # The z-step is typically just a projection/averaging.
        
        # Let's assume z is just weighted average of proposals x.
        for pid in self.P_i:
            delay = self.proposals_delays[pid]
            if adaptive:
                rho = self.rho_base / (1.0 + delay)
            else:
                rho = self.rho_base
            
            # Proposal for ME (pid thinks I should be at...)
            prop_x = self.proposals_received[pid]
            # Adjust for formation if they sent raw x?
            # They propose x_i. I want z. x_i ~ z + offset_i.
            # So z contribution = x_i - offset_i.
            contrib = prop_x - self.offsets[self.id]
            
            num += rho * contrib
            den += rho
            
        z_new = num / den
        
        # Dual Residual Metric (z change)
        z_change = np.linalg.norm(z_new - self.z_self)
        self.z_self = z_new
        return z_change

    def update_duals(self, adaptive=True):
        """
        Step 5: Dual Update
        For each j in N_i U {i}:
        y_j^i += rho * (x_j^i - z_j)
        """
        rho_sum = 0
        count = 0
        
        # 1. Self
        target_pos = self.z_self + self.offsets[self.id]
        resid = self.x_self - target_pos
        self.y[self.id] += self.rho_base * resid
        rho_sum += self.rho_base
        count += 1
        
        # 2. Neighbors
        for nid in self.N_i:
            delay = self.z_delays[nid]
            if adaptive:
                rho = self.rho_base / (1.0 + delay)
            else:
                rho = self.rho_base
            
            # z_received[nid] is z_j
            z_j = self.z_received[nid]
            target_pos = z_j + self.offsets[nid]
            
            resid = self.x_proposals[nid] - target_pos
            self.y[nid] += rho * resid
            rho_sum += rho
            count += 1
            
        self.last_avg_rho = rho_sum / max(count, 1)
        return

class AsyncSimulator:
    def __init__(self, n_agents=6, horizon=10, prob_delay=0.9, max_delay=5):
        self.N = n_agents
        self.K = horizon
        self.prob_delay = prob_delay
        self.max_delay = max_delay
        
        # Topology: Ring (i connects to i-1, i+1)
        # N_i: Who I talk to (and optimize for). 
        # P_i: Who talks to me. In undirected ring, N_i == P_i.
        self.N_map = {}
        self.P_map = {}
        for i in range(self.N):
            prev_i = (i - 1) % self.N
            next_i = (i + 1) % self.N
            neighbors = [prev_i, next_i]
            self.N_map[i] = neighbors
            self.P_map[i] = neighbors # Symmetric
            
        # Formation: Hexagon
        self.offsets = {}
        radius = 5.0
        angles = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        for i in range(self.N):
            self.offsets[i] = np.array([radius * np.cos(angles[i]), radius * np.sin(angles[i])])
            
        # Agents
        self.agents = []
        for i in range(self.N):
            start_pos = np.random.rand(2) * 5 + self.offsets[i]
            a = Agent(i, self.N_map[i], self.P_map[i], self.K, 2, 2, 
                      start_pos, self.offsets)
            self.agents.append(a)
            
        # Queues
        # queue_L2G: (sender_i, receiver_j, data_x_j, arrival_time) (Proposals)
        # queue_G2L: (sender_j, receiver_i, data_z_j, arrival_time) (Consensus)
        self.queue_L2G = []
        self.queue_G2L = []
        
        self.time_step = 0

    def step(self, adaptive=True):
        # 1. Local Optimization
        total_energy = 0
        for ag in self.agents:
            # We assume agent also has its own z_self accessible (no delay for self)
            # Z from neighbors is in ag.z_received (updated by G2L)
            e = ag.local_optimization(adaptive=adaptive)
            total_energy += e
            
        # 2. Local-to-Global Comm (Send Proposals)
        # i sends x_j^i to j
        for ag in self.agents:
            # Send to self (instant)
            # (Implicitly handled, Agent has access to x_proposals and x_self)
            
            # Send to neighbors j
            for nid in ag.N_i:
                delay = 0
                if np.random.rand() < self.prob_delay:
                    delay = np.random.randint(1, self.max_delay + 1)
                
                # Payload: The trajectory I propose for j
                payload = ag.x_proposals[nid].copy()
                
                self.queue_L2G.append({
                    'sender': ag.id,
                    'receiver': nid,
                    'data': payload,
                    'arrival': self.time_step + delay,
                    'sent_at': self.time_step
                })
                
        # Process L2G Queue (Deliver to Agents)
        # Agents are "Predecessors" from receiver's POV
        # If j receives from i, i is in P_j.
        for msg in self.queue_L2G:
            if msg['arrival'] <= self.time_step:
                rec = self.agents[msg['receiver']]
                sen_id = msg['sender']
                
                # Update if new
                # We need to track timestamp per link to know if it's new?
                # Simplified: Just overwrite.
                # Delay calculation: current_time - sent_at
                actual_delay = self.time_step - msg['sent_at']
                rec.proposals_received[sen_id] = msg['data']
                rec.proposals_delays[sen_id] = actual_delay
                
        # 3. Global Update (Consensus)
        total_dual_res = 0
        for ag in self.agents:
            dz = ag.update_consensus(adaptive=adaptive)
            total_dual_res += dz
            
        # 4. Global-to-Local Comm (Send Consensus)
        # i sends z_i to predecessors (who optimized for i)
        # i.e. send to k in P_i
        for ag in self.agents:
            # Send current z_i
            payload = ag.z_self.copy()
            
            for pid in ag.P_i:
                delay = 0
                if np.random.rand() < self.prob_delay:
                    delay = np.random.randint(1, self.max_delay + 1)
                    
                self.queue_G2L.append({
                    'sender': ag.id,
                    'receiver': pid,
                    'data': payload,
                    'arrival': self.time_step + delay,
                    'sent_at': self.time_step
                })
                
        # Process G2L Queue
        for msg in self.queue_G2L:
            if msg['arrival'] <= self.time_step:
                rec = self.agents[msg['receiver']] # The agent who proposed
                sen_id = msg['sender'] # The consensus owner
                
                # Update
                actual_delay = self.time_step - msg['sent_at']
                rec.z_received[sen_id] = msg['data']
                rec.z_delays[sen_id] = actual_delay
                
        # 5. Dual Update
        # Primal Residual calc while we are at it
        total_primal_res = 0
        sum_rho = 0
        count_rho = 0
        
        for ag in self.agents:
            # We need to peek inside update_duals or do it here to track rho
            # Let's just replicate the rho logic for metric tracking or modify update_duals.
            # Easiest: modify agent to store 'last_used_rho_avg'.
            ag.update_duals(adaptive=adaptive)
            sum_rho += getattr(ag, 'last_avg_rho', 1.0) # We will add this to agent
            count_rho += 1
            
            # Primal Res
            total_primal_res += np.linalg.norm((ag.x_self - ag.offsets[ag.id]) - ag.z_self)
            for nid in ag.N_i:
                z_j = ag.z_received[nid]
                total_primal_res += np.linalg.norm((ag.x_proposals[nid] - ag.offsets[nid]) - z_j)
                
        avg_rho = sum_rho / max(count_rho, 1)
        self.time_step += 1
        return total_primal_res, total_dual_res, total_energy, avg_rho
