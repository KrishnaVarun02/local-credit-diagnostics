# Diagnostic protocol, version 1.0

Written before pilot and final evaluation on 2026-10-01. This is a local protocol, not a public preregistration. Theory informed the design; the result is an intentionally constructed counterexample, not a hypothesis discovered from held-out tasks.

## Question and claim boundary

Can small uniform absolute reconstruction MSE certify a useful greedy decision in a cooperative one-step game? We investigate a transparent family and separately show how a normalized metric exposes its sparse reward. This is a diagnostic of least-squares reward projection, not a reproduction or performance comparison of LOMAQ, VDN, QMIX, QTRAN, or DCG. Weighted QMIX already establishes the central conceptual failure; no claim of first discovery or algorithmic novelty is made.

## Fixed tasks and estimators

For n binary agents, every joint action is enumerated. Reward is R(a)=1[a=1^n]−κ 2^(1−n)Σa_i. The primary setting is κ=1.5. Each projected model includes a constant and unary Walsh features x_i=2a_i−1; graph models additionally include x_i x_j for edges in a chain, ring, star, or complete graph. Uniform projections are computed exactly as empirical inner products over the complete population. Greedy decisions for graph models are centrally enumerated (not decentralized message passing); additive decisions are decentralizable. Exact oracle return is an evaluator reference.

The oracle-anchor ablation assigns known optimal action 1^n weight w in full-population additive least squares, all others weight 1. This is deliberately privileged, not a proposed deployable mitigation or Weighted QMIX implementation. It diagnoses the tradeoff between projection weighting and decision quality. Weights 2, 16, and 256 are fixed before execution, not tuned.

## Experiment matrix and controls

1. Scaling: n in {4,6,8,10,12,14,16}, five projections and three oracle-anchor weights, κ=1.5.
2. Strength: n=8, κ in {0,0.5,0.9,1,1.1,1.5,2,3}, five projections. κ=1 is a tie case, reported separately.
3. Finite coverage: n in {8,12}; m in {64,256,1024,4096}; 100 independent uniform-with-replacement datasets per cell. Fit additive least squares using only sampled action/reward pairs, without reward formulas, oracle actions, or evaluation rewards passed into the fitting function. Seed = 20261001 + 100000*n + 1000*m + replicate. Evaluate on every joint action; training/evaluation overlap is intentional because this isolates coverage and projection, not task generalization. No held-out task claims.
4. Controls: exactly representable additive and chain-pairwise games; the published 3×3 game [[8,−12,−12],[−12,0,0],[−12,0,0]] from QTRAN, also reproduced as Table 2 in Weighted QMIX. Our additive fit is not either published algorithm.

## Metrics and analysis

Primary metrics: uniform mean squared error and coordination regret max_a R(a)−R(argmax_a Rhat(a)). Also report MSE/Var(R), root MSE/reward range, maximum absolute error, reward at selected action, optimal-action success, synergy-action observations, rank, and wall runtime. Lexicographically first maximizer breaks ties (numeric tolerance 1e−12); this affects the κ=1 stratum and is disclosed. Raw actions use least-significant-bit agent ordering.

Exact experiments have no statistical error bars. For finite coverage, datasets are the independent unit; report sample mean regret with a descriptive 95% normal interval, and Wilson 95% intervals for success/observation rates. All 100 replicates are retained. These intervals characterize only the sampling process on the fixed synthetic tasks; no p-values or broad significance claims. One hundred seeds give worst-case binomial standard error 0.05, appropriate to the diagnostic aim, not a power justification for small effects.

## Execution and stopping

Run tests and a 3-seed smoke pilot first; retain its separate manifest. Run the fixed full matrix once if pilot runtime is tractable. No paid services, GPU, or external datasets. Only invalid numerical runs would be excluded, with reason retained; rank-deficient fits are retained. Fixes require logging and rerunning affected results. Record code/config/protocol SHA-256, dependency versions, platform, CPU model, timestamps, and run times. Freeze reported artifacts; reproduction uses a new output directory. Exact theory and independent least-squares tests cross-check every central deterministic claim.

## External validation gaps

No temporal tasks, learned observations, neural optimization, environment shift, SMAC, PettingZoo, or LOMAQ training. The topology manipulation changes the approximation class, not a time-varying physical environment. Generalization to those settings remains untested. The matrix-game control is an established diagnostic, not a substantial external MARL benchmark.
