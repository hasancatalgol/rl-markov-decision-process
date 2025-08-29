# Markov Decision Process (MDP)

A **Markov Decision Process (MDP)** formalizes sequential decision‑making under uncertainty. At each step $t$ an **agent** sees a **state** $s_t$, picks an **action** $a_t$, the **environment** moves to a new state $s_{t+1}$ and emits a **reward** $r_{t+1}$. The goal is to choose actions that maximize cumulative reward. It is a **discreet stochastic control process**

---

## Core pieces (super short)

* **Environment**: The world the agent interacts with (e.g., a grid maze).
* **State (s)**: Information the agent uses to decide; a snapshot of the environment. *Markov* means the next state/reward depend only on the current $s,a$, not on the past.
* **Action (a)**: A choice the agent can take in a state (discrete or continuous).
* **Agent**: The decision‑maker; follows a **policy** $\pi(a\mid s)$ mapping states to action probabilities.
* **Reward (r)**: Immediate scalar feedback from the environment after an action.

---

## Formal MDP

$\mathcal{M} = (\mathcal{S},\mathcal{A}, P, R, \gamma)$

* $P(s'\mid s,a)$: transition dynamics
* $R(s,a)$: expected immediate reward
* $\gamma \in [0,1)$: discount factor

**Objective**: maximize expected return
$G_0 = \sum_{t=0}^{\infty} \gamma^t r_{t+1}.$

---

## Tiny loop (pseudocode)

```text
initialize s ~ start_state
while s not terminal:
    a ~ π(·|s)           # pick an action
    s, r ← environment.step(a)
    update agent with (s, a, r)
```

> Example: In a maze, **state** is the agent’s cell, **actions** are up/right/down/left, **reward** is −1 per step until the exit (terminal state).
