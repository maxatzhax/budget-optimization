# Mathematical formulation

## Notation

| Symbol | Meaning |
|---|---|
| $I$ | set of budget items, $i \in I$ |
| $D$ | set of departments; $I_d \subseteq I$ — items of department $d$ |
| $S$ | set of economic scenarios: recession, stable, growth |
| $B$ | total available budget (million KZT) |
| $l_i, u_i$ | minimum and maximum allocation to item $i$ ($l_i = u_i$ for mandatory items) |
| $a_d, b_d$ | minimum and maximum share of department $d$ in the allocated budget |
| $r_i^s$ | effect of 1 unit spent on item $i$ under scenario $s$ |
| $p_s$ | probability (weight) of scenario $s$, $\sum_s p_s = 1$ |
| $x_i$ | **decision variable** — amount allocated to item $i$ |

## Feasible set $X$

$$
\begin{aligned}
& \textstyle\sum_{i \in I} x_i \le B && \text{(total budget)} \\
& l_i \le x_i \le u_i, \quad i \in I && \text{(item bounds)} \\
& a_d \textstyle\sum_{i \in I} x_i \;\le\; \sum_{i \in I_d} x_i \;\le\; b_d \sum_{i \in I} x_i, \quad d \in D && \text{(department shares)}
\end{aligned}
$$

Department shares are defined relative to the amount actually allocated, so a
larger budget never makes the problem infeasible — surplus funds simply stay
unallocated once every useful item hits its upper bound.

## Decision criteria

**1. Single scenario** (`solve_scenario`) — deterministic plan for a given forecast:

$$\max_{x \in X} \sum_i r_i^s x_i$$

**2. Expected value / Bayes criterion** (`solve_expected`):

$$\max_{x \in X} \sum_{s \in S} p_s \sum_i r_i^s x_i$$

Default weights: $p_\text{recession}=0.3,\ p_\text{stable}=0.5,\ p_\text{growth}=0.2$.

**3. Robust / Wald max-min criterion** (`solve_robust`) — the plan with the best
guaranteed result. The non-linear $\max_x \min_s$ is linearised with an
auxiliary variable $t$:

$$
\max_{x \in X,\ t} \; t \quad \text{s.t.} \quad t \le \sum_i r_i^s x_i \quad \forall s \in S
$$

## Sensitivity analysis

The dual value $\lambda_B$ of the total-budget constraint (returned by HiGHS) is
the **shadow price of the budget**: the increase of the objective from one extra
unit of budget. `budget_sensitivity` re-solves the model over a grid of $B$
values; $\lambda_B$ is piecewise constant and drops to 0 once additional money
can no longer be used under the item and department limits.

## Solver

All models are linear programmes solved with HiGHS through
`scipy.optimize.linprog(method="highs")`.
