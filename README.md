# Markov Decision Process (MDP)

A **Markov Decision Process (MDP)** formalizes sequential decision-making under uncertainty. At each step $t$ an **agent** sees a **state** $s_t$, picks an **action** $a_t$, the **environment** moves to a new state $s_{t+1}$ and emits a **reward** $r_{t+1}$. The goal is to choose actions that maximize cumulative reward. It is a **discreet stochastic control process**.

---

## Core pieces (super short)

* **Environment**: The world the agent interacts with (e.g., a grid maze).
* **State (s)**: Information the agent uses to decide; a snapshot of the environment. *Markov* means the next state/reward depend only on the current $s,a$, not on the past.
* **Action (a)**: A choice the agent can take in a state (discrete or continuous).
* **Agent**: The decision-maker; follows a **policy** $\pi(a\mid s)$ mapping states to action probabilities.
* **Reward (r)**: Immediate scalar feedback from the environment after an action.

---

## The Markov Property

The defining property of a Markov process is **memorylessness**:  
the future depends only on the present state, not on the past.

$P[S_{t+1} \mid S_t = s_t] \;=\; P[S_{t+1} \mid S_t = s_t, S_{t-1} = s_{t-1}, \dots, S_0 = s_0]$

* The next state depends only on the current state, not on previous ones.  
* If a process satisfies this property, it is called a **Markov process**.

---

## Formal MDP

$\mathcal{M} = (\mathcal{S},\mathcal{A}, P, R, \gamma)$


* $\mathcal{S}$: set of possible states  
* $\mathcal{A}$: set of actions that can be taken  
* $P(s'\mid s,a)$: transition probabilities of moving from state $s$ to $s'$ under action $a$  
* $R(s,a)$: expected immediate reward for state–action pair  
* $\gamma \in [0,1)$: discount factor  

**Objective**: maximize expected return

$G_0 = \sum_{t=0}^{\infty} \gamma^t r_{t+1}.$

---

## Markov Chain vs Markov Decision Process

| Aspect | Markov Chain | Markov Decision Process |
|--------|--------------|--------------------------|
| **Actions** | ❌ none | ✅ agent chooses action |
| **Rewards** | ❌ none | ✅ rewards linked to $(s,a)$ |
| **Transition** | $P(s' \mid s)$ | $P(s' \mid s,a)$ |
| **Objective** | Just describes dynamics | Optimize decisions for long-term reward |

👉 **Markov chain = passive system** (just state transitions).  
👉 **MDP = interactive system** (agent makes choices + gets feedback).  

---

## Types of Markov Decision Processes

1. **Finite vs Infinite**  
   * **Finite MDP** – The state space $\mathcal{S}$ and action space $\mathcal{A}$ are finite (countable).  
     > Example: A grid-world maze where states are grid cells and actions are moves up/down/left/right.  
   * **Infinite MDP** – Either the state or action space is infinite (often continuous).  
     > Example: A robot navigating in continuous 2D space with continuous velocity choices.  

   👉 Many real-world problems are infinite MDPs, but finite cases are simpler and used for theoretical analysis.

2. **Episodic vs Continuing**  
   * **Episodic tasks** – Interaction breaks into episodes with a start and a terminal state.  
     > Example: Playing a game of chess or reaching the exit of a maze.  
   * **Continuing tasks** – No natural terminal state; the process continues indefinitely.  
     > Example: An automated stock trading agent that interacts with the market continuously.  

   👉 Episodic MDPs are solved per episode, while continuing MDPs rely on discounting ($\gamma$) to ensure long-term rewards remain finite.

3. **Trajectory vs Episode**  
   * **Trajectory** – A sequence of states, actions, and rewards over time:  
     $$(s_0, a_0, r_1, s_1, a_1, r_2, \dots)$$  
   * **Episode** – A trajectory that **terminates** when a terminal state is reached.  

   👉 All episodes are trajectories, but not all trajectories are complete episodes.

4. **Reward vs Return**  
   * **Reward ($r_t$)** – The immediate scalar feedback at time step $t$.  
     > Example: −1 for each step in a maze.  
   * **Return ($G_t$)** – The cumulative discounted reward starting from time $t$:  
     $$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$$  

   👉 Rewards are **short-term signals**, while returns capture **long-term objectives**.

5. **Discount Factor ($\gamma$)**  
   * A number in the range $[0,1)$.  
   * Determines how much future rewards are valued compared to immediate rewards.  
   * Appears in the return definition:  
     $$G_t = \sum_{k=0}^\infty \gamma^k r_{t+k+1}$$  

   **Interpretation**:  
   * $\gamma \approx 0$ → the agent is **myopic**, focusing almost only on immediate rewards.  
   * $\gamma \approx 1$ → the agent is **far-sighted**, strongly considering long-term rewards.  

   👉 Choosing $\gamma$ balances short-term vs long-term decision-making.

6. **Policy ($\pi$)**  
   * A **policy** defines the agent’s behavior: it maps states to actions.  
   * Can be **deterministic**:  
     $$\pi(s) = a$$  
     (always picks the same action in a given state)  
   * Or **stochastic**:  
     $$\pi(a \mid s) = P[A_t = a \mid S_t = s]$$  
     (gives a probability distribution over actions for each state)  

   **Goal**:  
   Find an **optimal policy** π* that maximizes the expected return.  

   👉 The policy is the agent’s “strategy” for decision-making.

   The optimal policy is denoted by π*. One compact way to write it is:
   <p><strong>π*</strong> = argmax<sub>π</sub> V<sup>π</sup></p>
   Equivalently, per-state in terms of action-value:
   <p><strong>π*</strong>(s) = argmax<sub>a</sub> q<sup>*</sup>(s,a)</p>

7. **Value Functions**

Value functions estimate how good it is to be in a state or to take an action, under a given policy $\pi$.

* **State-value function** ($v_\pi(s)$):  
  Expected return starting from state $s$, following policy π:

  <p>v<sub>π</sub>(s) = E<sub>π</sub>[ G<sub>t</sub> | S<sub>t</sub> = s ]</p>  
  

  👉 Answers: *“How good is it to be in this state?”*

* **Action-value function** ($q_\pi(s,a)$):  
  Expected return starting from state $s$, taking action $a$, and then following policy π:
  
  <p>q<sub>π</sub>(s,a) = E<sub>π</sub>[ G<sub>t</sub> | S<sub>t</sub> = s, A<sub>t</sub> = a ]</p>  

  👉 Answers: *“How good is it to take this action in this state?”*

---

### Bellman Equations

These express the recursive relationship between values of states and state–action pairs.

* **State-value (Bellman expectation equation):**  
  $$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s',r} P(s',r \mid s,a) \big[ r + \gamma v_\pi(s') \big]$$  

* **Action-value (Bellman expectation equation):**  
  $$q_\pi(s,a) = \sum_{s',r} P(s',r \mid s,a) \big[ r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s',a') \big]$$  

---

**Key difference**:  
- $v(s)$ evaluates **states**.  
- $q(s,a)$ evaluates **state–action pairs**.  

Both are central for learning and improving policies (e.g., Q-learning, value iteration).
