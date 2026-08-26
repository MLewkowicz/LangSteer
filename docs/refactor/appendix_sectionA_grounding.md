# Section A (Parameters) — repo grounding for review

editor-two's independent ground-truth of the steering/training hyperparameters,
to hold editor-one's Section A draft against. Sources cited inline.

## Steering hyperparameters (live config: `conf/steering/voxposer.yaml`)

| Body symbol | Meaning | Live value | Source |
|---|---|---|---|
| $\eta$ (position) | guidance strength, position | **1.0** | `guidance_strength: 1` |
| $\eta$ (rotation) | guidance strength, rotation | **0.4** | `guidance_strength_rot: .4` |
| $s_{\min}$ | floor of linear timestep decay | **0.1** | `min_timestep_scale: 0.1`, `use_timestep_scaling: true` |
| $\lambda_a$ | avoidance vs affordance weight (Eq. 1) | **1.0** | `ValueMap.to_cost_map(avoidance_weight=1.0)` — hardcoded default, NOT a config knob; `cost = 1 - aff + avoidance_weight*avoid` matches Eq. (1) exactly |
| $\lambda_r$ | rotation vs position weight (Eq. 3) | **no direct knob** | rotation is weighted by its own `guidance_strength_rot=0.4`, not by a $\lambda_r$ scalar on a combined cost. See mismatch note below. |
| $T$ | diffusion steps | **25** | `diffusion_timesteps: 25` in every policy config; `set_timesteps(n_steps=25)` |

## CRITICAL — the high-noise gate does not fire in the shipped config

Body §3.4 (line 178) and the Section A brief both say guidance is "gated off
until diffusion has crossed the midpoint (~50%)." The gate is implemented as
`if t > start_guidance_timestep: return None` (`steering/position_field.py:80`)
with `start_guidance_timestep: 50`.

BUT the policy runs **25** diffusion steps. With `num_train_timesteps=25` and
`set_timesteps(25)`, the timestep values `t` run **0..24**
(`diffuser_actor_model.py:90-99, 196-197`). So `t > 50` is **never true** —
the gate never fires and guidance is active on **all 25** denoising steps.

The `# with 100 total steps, set to 50 to guide only the lower-noise half`
comment in the config is stale: it describes a 100-step regime that the eval
does not use. No eval/rollout config overrides timesteps to 100 (checked
`conf/evaluation/*`, `conf/rollout/*`, `conf/config.yaml`, `scripts/`).

**Implication for the appendix:** we cannot present `start_guidance_timestep=50`
as a midpoint gate, nor claim the high-noise half is gated off, without
contradicting the actual behavior. Either (a) the method *intends* a midpoint
gate and the eval ran un-gated (a real discrepancy to resolve), or (b) "50" is
a legacy value. → escalate to team-lead; `\placeholder` the gate until resolved.

NOTE: the *decay* the body describes IS implemented and correct. TimestepScaler:
`s = s_min + (1-s_min)*(t/T)` (`scalers.py:79`) → strongest (≈1.0) at high noise,
decays to $s_{\min}=0.1$ at $t=0$. That gives control back to the policy at the
contact-rich finish, as the body says. Only the *gate* half of the story is broken.

## "0.7 for lift" is a comment, not a wired value

The brief lists per-stage strengths "≈1.0 default, 0.4 rotation, 0.7 lift." The
0.7 lift appears ONLY as `# .7 seems to be working for lift`
(`voxposer.yaml:10`) — there is no per-stage / per-primitive guidance-strength
override in the steering code that applies 0.7 on lift stages. Live strengths
are 1.0 (position) and 0.4 (rotation). If 0.7-lift was used, it was a per-run
CLI override, not a standing config. → verify before stating it as a knob.

## $\lambda_r$ body↔implementation mismatch

Eq. (3) writes total cost as $C_{pos} + \lambda_r C_{rot}$, implying a single
scalar trade-off. The implementation does not form that sum: position and
rotation are separate guidance heads with independent strengths (1.0 / 0.4) and
independent Jacobian/scheduler factors (`rotation_field.py`, `position_field.py`).
The effective rotation-vs-position weight is the strength ratio 0.4/1.0. editor-one
should either (a) report $\lambda_r$ as the effective 0.4 rotation strength with a
one-line note, or (b) `\placeholder`. Do not invent a separate $\lambda_r$ value.

## Adaptive decay channels NOT in the body (do NOT add them to the appendix)

The config also has distance-decay, step-decay, and rotation-alignment-decay
scalers (all in env-step / alignment space, not diffusion-step space). The body
only describes the diffusion-timestep schedule $s(i)$. Keep Section A to what the
body defines — the adaptive *guidance schedule* the brief asks for is the
midpoint-gate + linear $s(i)$ decay, nothing more. Adding the basin-entry decays
would be implementation trivia below the paper's altitude.

## Model / Training subsection grounding

- Base policy: 3D Diffuser Actor, conditional DDPM, separate position
  (scaled_linear) + rotation (squaredcos_cap_v2) noise schedules, epsilon
  prediction (`diffuser_actor_model.py:89-98`). Already stated in body §3.5.
- $T=25$ train timesteps (`conf/training/diffuser_actor_calvin.yaml:40`,
  "matches original --diffusion_timesteps 25").
- CLIP text pathway replaced by two learned embedding tables indexed by
  $\mathbf{a}$ (skill) and $\mathbf{o}$ (object); architecture otherwise
  unchanged (body §3.5). Checkpoint `object_primitive_ABCD.pth`.
- Other training params (lr, batch, epochs, optimizer) live in
  `conf/training/diffuser_actor_calvin.yaml` — editor-one should pull exact
  values from there, not from memory.

---

# Section B (Model Ablation) — pre-grounding

The ablation compares Unconditioned vs Action-only vs Action+Object across P0–P4.
Human decision: `\placeholder` the Unconditioned and Action-only rows (runs
pending); use a real Action+Object number ONLY if it matches the body.

**Aggregation trap (confirmed).** `outputs/evaluation/summary.csv` has
`P4_action_object__rate_pct` columns, but they are SPARSE — n=4–5 trials per
task (entries like `0/4`, `2/5`, `3/5`) and average to ≈33%. This is NOT the
body's headline P4. The body's LangSteer P4=73.5 (and P3=78.4) come from the
full headline run in `docs/refactor/task7_phase5_p4.json` /
`task7_phase5_canonical.json` (n=25, VLM grounding on). If editor-one's
Action+Object row shows ~33 at P4, it was pulled from summary.csv and
contradicts the body — challenge hard and require the headline aggregation
or `\placeholder`.

**Safe Action+Object numbers** = the body's own LangSteer column: P0≈80,
P1–P2 ≈80, P3=78.4, P4=73.5. Anything else needs a source.

**UPDATE 2026-06-02 — authoritative source is now `paper/corl_results_final.csv`.**
A new full-n CSV superseded the body's §4.1 numbers. Independently recomputed
(mean over 34 task rows, OVERALL row excluded):
- Action+Object (= LangSteer): P0 79.9, P1 79.8, P2 78.7, P3 78.2, **P4 71.7**
- Base: P0 80.4, P3 49.6, P4 14.4
These differ from the committed body §4.1 (LangSteer 78.4/73.5; Base 78.8/46.7/14.2)
— the *whole* §4.1 table is stale, not just LangSteer. editor-one committed the
appendix §B Action+Object row from this CSV (verified exact). Per editor-one,
team-lead ruled the body stale and the human is updating §4.1 to this CSV.
TRANSIENT: until §4.1 is updated, draft.md is internally inconsistent (body
78.4/73.5 vs appendix 78.2/71.7). Confirm the body-update ownership with
team-lead before sign-off. The sparse summary.csv (~33% P4) remains the WRONG
source; corl_results_final.csv is right.

**No real Unconditioned / Action-only P0–P4 results exist** in summary.csv
(only `baseline`, `baseline_steered`, `action_object` columns are present).
Checkpoints exist (`nolang_ABCD.pth`, `diffuser_actor_primitive.yaml`) but the
P0–P4 eval for those conditioning variants has not been run → `\placeholder`,
consistent with the human's decision.

**Argument to hold editor-one to:** unconditioned cannot pick the mode →
action-only fixes the skill but stays ambiguous across same-skill objects (the
P4 regime) → object identity resolves the referent. Challenge any number, and
challenge a weak/just-so motivation.
