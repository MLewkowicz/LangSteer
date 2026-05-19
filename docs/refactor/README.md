# LangSteer Refactor — Task Index

Branch: `refactoring` | Started: 2026-05-18

## Tasks

| # | Name | File | Status |
|---|------|------|--------|
| 1 | Steering module separation (voxposer_steering.py) | [task1_steering_separation.md](task1_steering_separation.md) | ✅ done — open_drawer 1/1 passed |
| 2 | Policy variant split (diffuser actor variants) | [task2_policy_variants.md](task2_policy_variants.md) | ✅ done — open_drawer 1/1 passed |
| 3a | VoxPoser prompts cleanup + vocab linter + dead-helper removal | [task3a_voxposer_prompts.md](task3a_voxposer_prompts.md) | ✅ done — 36.9% → 55.7% (+50% relative) |
| 3b | Value-map redesign + loop-back fix + perturbation robustness | [task3b_value_map_redesign.md](task3b_value_map_redesign.md) | ✅ done — 52.1% → 54.3% (+2.1pp, gpt-5.4-mini) |
| 4 | Loop-back dwell tuning + per-task diagnostics | [task4_diagnostics.md](task4_diagnostics.md) | ✅ done — dwell 3→4 (close_drawer 3/3, no cut-short; loop-back fires correctly); 28 tasks classified, 5 recommended fixes flagged |
| 5 | Visualization cleanup | [task5_log.md](task5_log.md) | ✅ done — 5 renderers → 3; ~−1700 LoC; Renderer Protocol; OBJECT label; HTML dedup; headless tk; `run_evaluation` wired |
| 7 | VLM scene-image ingestion for value-map construction | *(doc pending)* | ⬜ available — unblocked by 3b |

## Status legend
- ✅ done
- 🟡 in-progress / planning
- ⬜ blocked (waiting on upstream task)
- ❌ cancelled

## Scope summary

**Task 1** splits `steering/voxposer_steering.py` (1583 lines) into focused modules: trajectory-space guidance, rotation-space guidance, adaptive scaling adapters, and a separate stage-manager. The refactor must not break `uv run python scripts/run_evaluation.py policy=diffuser_actor_primitive_object steering=voxposer`.

**Tasks 2–5** are downstream and will be unblocked after Task 1 lands.
