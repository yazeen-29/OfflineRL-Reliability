# P1 Counterfactual Unit-Test Validation

Status: PASS

Environment:
Hopper-v5

Policy:
Frozen IQL seed 0

Verified properties:

1. Identical MuJoCo qpos/qvel state can be restored in independent
   simulator branches.

2. Clean and shifted policy observations differ.

3. The observation perturbation does not modify the underlying physical
   simulator state.

4. The frozen IQL policy produces a different action for the tested
   shifted observation.

5. Both branches execute the prescribed H=10 decision-consequence
   horizon.

Diagnostic result for the fixed test state:

Clean action:
[-0.08500707, 0.19303781, -0.29235882]

Shifted action:
[-0.10374147, 0.18491298, -0.28015417]

Action disagreement:
0.0137349168

Clean 10-step return:
9.0217896793

Shifted 10-step return:
9.0312177201

Diagnostic return difference:
-0.0094280407

Interpretation:

The negative diagnostic return difference is not treated as evidence
against the research hypotheses. This is a single implementation test
state and is used only to verify the counterfactual execution mechanism.

Primary P1 statistical inference is performed only after the
prespecified multi-state, multi-seed experiment.

Protocol:
experiments/reliability/P1_DECISION_CONSEQUENCE_PROTOCOL.md
