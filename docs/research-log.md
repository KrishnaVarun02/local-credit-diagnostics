# Research log

## 2026-10-01 — scope and prior-art check

The proposed broad investigation was narrowed to an exact, one-step diagnostic. The available historic MARL archive did not provide executable experiment code or verifiable raw runs, so it could not support an original LOMAQ reproduction. No old report, its coauthor names, its figures, or its claimed metrics are included here.

Primary-source review identified Weighted QMIX as direct conceptual prior work for the projection/decision mismatch, and LOMAQ already discusses locality misspecification. The package therefore makes no novelty or new-algorithm claim. It is an independent implementation of a small controlled diagnostic, with a full draft manuscript and explicit broader-validation gaps.

## Design correction before protocol freeze

An initial sketch used a cost coefficient that would have produced a tied additive action preference. Algebra checking corrected the main reward to `1[all ones] − 3·2^-n Σa_i`, with n≥4, before implementation, protocol freeze, or experiments. This gives strict negative singleton coefficients and a strict wrong maximizing action. No discarded empirical runs were involved.

## Frozen protocol and execution

`docs/protocol.md` and `configs/*.json` were written before the first run. The protocol is locally timestamped through manifests/hashes, not publicly preregistered. The theory deliberately predicts the outcome; these tasks are not held-out discovery evidence.

- 33 meaningful tests passed, including 20 independent complete-design least-squares checks of the theorem and nine independent weighted-fit checks.
- `results/pilot/`: 29 exact evaluations, 6 sampled fits, 3 controls; 1.55448 seconds in-process runtime.
- `results/full/`: 96 exact evaluations, 800 sampled fits, 3 controls; 0.271593 seconds in-process runtime.
- Every requested final run was retained. There were no failed runs, exclusions, or tuning changes.
- First-time figure rendering separately built a font cache; rendering time is not counted as experiment time.
- The principal figure was visually inspected. The manifest records the available generic ARM CPU identification; an exact marketed processor model was not available through the sandboxed query.

## Independent review and corrections

A separate reviewing agent reran all 33 tests, checked every recorded source/result hash, independently regenerated all 800 training datasets, and refit them. The reviewer reported exact agreement of coefficients and metrics, and verified the proposition and weighted-anchor derivation. This was an independent computational check within the same AI-assisted session, not an external human peer review.

The reviewer caught an incorrect Python-version statement in early documentation. It was corrected to the manifest's **Python 3.12.13**, and a complete direct/transitive package lock was added. This did not change the experiment code or results. A normalization sentence was clarified to specify κ=1.5. The manuscript subtitle now explicitly names additive and pairwise projection.

**Frozen-protocol wording erratum:** the protocol says “lexicographically first” but the implemented and tested rule is the **lowest integer action index in least-significant-bit-first enumeration**. This can differ from lexicographic ordering of the displayed agent tuple. The paper describes the implemented rule. The primary additive/pairwise maximizers are unique; in the κ=1 tie stratum, both rules pick all zeros. The frozen protocol is retained unchanged so its pre-run hash remains verifiable. No result changes or reruns were needed for this wording correction.

The scope remains strictly additive and pairwise. Odd higher-order factors can change the preferred action; the paper does not claim that all restricted-order models fail.

## Reproduction check

The complete matrix was run again in `reproduction/full/` (ignored by git) using the same scientific environment. `python -m src.verify --reference results/full --reproduction reproduction/full` passed: all 899 result rows matched within the declared tolerance and every training-index hash was checked. A repeat test run passed all 33 tests.

A second, fresh Python 3.12.13 virtual environment then installed only the locked
dependencies and reran all 33 tests plus the full protocol. All 899 rows and all
800 dataset hashes matched the retained evidence; the record is
`results/clean-verification.json`. PDF compilation used Tectonic 0.17.0, and every
page was visually inspected. Compilation and verification are complete.

## Status and exact continuation

Implementation, fixed scoped experiments, generated evidence, manuscript text,
PDF compilation, and visual QA are complete. GitHub delivery is recorded separately
in the task's publication manifest. No external academic submission has occurred.
Before submission, the author must review the actual code, proof, sources,
authorship/disclosure, and scientific contribution. A substantial MARL paper
additionally requires faithful baselines and sequential validation.

Recheck original evidence: `python -m src.verify`. Reproduce: `python -m src.run --config configs/full.json --output reproduction/new-run`. Generate assets from an alternate run: `python -m src.analyze --results reproduction/new-run --figures reproduction/new-figures --tables reproduction/new-tables`. Compile manuscript from `paper/`: `tectonic main.tex`.
