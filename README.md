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