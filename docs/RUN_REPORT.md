# GridWorld MDP — Explained Run

**What you’re seeing:**  
- **States (s)** are grid cells; **Actions (a)** move the agent; **Rewards (r)** are −0.04 per step and +1.0 at the terminal; **Return (G)** accumulates rewards (discounted by **γ=0.95**).  
- **Value Iteration (VI)** maximizes over actions each iteration; **Policy Iteration (PI)** alternates evaluation/improvement until policy stabilizes.

## Convergence & Policies
- ![VI Δ](value_iteration_convergence.png)  
  *Δ measures the max change in any state value each iteration; Δ→0 means we’ve solved the Bellman optimality equations.*
- ![PI changes](policy_iteration_changes.png)  
  *Number of states whose chosen action changed; zero means policy is stable (optimal under this model).*

## Episode Behavior
- **Logged traces** (CSV):
  - `trace_episode_vi.csv` — per-step (s, a, s', r, G) under VI policy  
  - `trace_episode_pi.csv` — per-step under PI policy
- **Returns**:
  - ![VI returns](episode_returns_value_iter.png)
  - ![PI returns](episode_returns_policy_iter.png)

### Key MDP takeaways
- *Markov property:* next state and reward depend only on current (s, a).  
- *Objective:* maximize expected **return** \(sum of discounted rewards\).  
- *γ (discount):* balances short vs long-term rewards; here γ=0.95.

