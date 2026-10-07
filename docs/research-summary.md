# Research summary

**Question.** Can low average reward-reconstruction error alone certify good cooperative decisions?

**Scoped contribution.** A transparent, tested diagnostic package for an established issue. For the binary sparse-synergy family, exact additive projection has MSE `2^-n − (n+1)4^-n` and regret `1 − 3n·2^-n`. A proof and exhaustive runs show that any pairwise graph lowers error according to its edge count but retains the wrong maximizing action. This is not a new MARL training algorithm or a novelty claim.

**Strongest evidence.** At 16 agents, additive MSE is 0.0000152548 while regret is 0.999268. Variance-normalized MSE is 0.999939: the apparent raw-error accuracy hides an almost entirely unexplained rare reward. The fixed matrix contains 96 exact evaluations, three controls, and 800 sampled fits. At eight agents, all 100 datasets of size 4096 observe the synergistic action, but only one fitted policy chooses it. More coverage alone does not repair the limiting objective.

**Ablation.** A twofold weight on the known optimal joint action gives zero regret with worse uniform MSE. This intervention is privileged and analytical. It shows the evaluation tradeoff; it does not provide an exploration algorithm.

**What is missing.** A distinct novelty case beyond prior projection analyses, faithful LOMAQ/QMIX/VDN baselines, sequential benchmark validation, observation dynamics, noisy rewards, and independent human scientific review. The supplied manuscript is a completed diagnostic draft, not a claim of submission readiness or publication.

**Next research decision.** Use the diagnostic to audit reward-decomposition evaluation in a properly reproduced sequential MARL system. Continue only if that system exposes a specific unresolved failure and a fair, testable mitigation. Preserve this synthetic result as a sanity check rather than presenting it as evidence of broad algorithm failure.
