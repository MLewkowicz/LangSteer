# Cavity Fixtures Plan: drawer_interior + slider_interior

You are extending the CALVIN LMP interface with two new fixture entries. The user has measured the bounding boxes for the two cavity interiors using their own annotator script (in `tmp/`) and will hand them to you. Your job is to plug those coordinates into the right config files so the rest of the codebase can resolve the new fixture names.

This plan is self-contained. You don't need any project-history or refactoring-orchestration context to execute it.

---

## Background

The codebase has a fixture dictionary at `voxposer/calvin_interface.py::CALVIN_FIXTURES` (around lines 80–126). It currently contains entries like:

- `drawer` (the drawer body) and `drawer_handle` (the pull bar on the drawer's front face)
- `slider` (the sliding door panel) and `slider_handle` (the grasp groove on the door's front)
- `lightbulb`, `switch`, `light_switch`, `led`, `button`

Each entry has the shape:
```python
'fixture_name': {
    'position': np.array([x, y, z]),    # world-frame center, meters
    'size':     np.array([sx, sy, sz]), # axis-aligned extents, meters
},
```

What's missing: the **interior cavities** of the drawer and the slider cabinet. The handles sit on the *front faces* (operator side), not inside. Tasks that need to *place things inside* (e.g. dropping a block into the drawer or slider) currently have no fixture target for the actual cavity, and composer-generated affordance queries fall back to handle positions — which point at the wrong place.

Adding `drawer_interior` and `slider_interior` as new fixture entries lets the rest of the LMP pipeline (composer prompt → affordance LMP → `parse_query_obj` → `detect`) resolve those names to the right world-frame target.

---

## Inputs (from the user)

The user will provide two axis-aligned bounding boxes:

- `drawer_interior`: position `(x, y, z)` + size `(sx, sy, sz)` in meters, world frame
- `slider_interior`: position `(x, y, z)` + size `(sx, sy, sz)` in meters, world frame

Match the format of the existing entries exactly (NumPy arrays, units of meters).

---

## What to change

### 1. `voxposer/calvin_interface.py::CALVIN_FIXTURES`

Add two new entries to the dict, alongside the existing fixtures:

```python
'drawer_interior': {
    'position': np.array([<x>, <y>, <z>]),
    'size':     np.array([<sx>, <sy>, <sz>]),
},
'slider_interior': {
    'position': np.array([<x>, <y>, <z>]),
    'size':     np.array([<sx>, <sy>, <sz>]),
},
```

Group them next to their related fixtures (`drawer_interior` near `drawer`/`drawer_handle`, `slider_interior` near `slider`/`slider_handle`) so the file stays readable.

### 2. Verify name resolution

After you add the entries, confirm that `CalvinLMPInterface.detect('drawer_interior')` and `detect('slider_interior')` both return valid Observation dicts. If the `detect()` method has an explicit whitelist of resolvable names, you may need to extend it to include the new entries. If it iterates over `CALVIN_FIXTURES` directly, no further change is needed.

A quick check:
```bash
uv run python -c "
from voxposer.calvin_interface import CalvinLMPInterface, CALVIN_FIXTURES
print('drawer_interior:', CALVIN_FIXTURES['drawer_interior'])
print('slider_interior:', CALVIN_FIXTURES['slider_interior'])
iface = CalvinLMPInterface({'map_size': 100})
print('detect drawer_interior:', iface.detect('drawer_interior'))
print('detect slider_interior:', iface.detect('slider_interior'))
"
```

If `detect()` errors on either name, fix the resolution path so it falls through to `CALVIN_FIXTURES` lookup. Do not add a new lookup table — extend the existing one.

### 3. Do NOT touch `OBJECT_VOCAB`

There is a strict 8-entry vocabulary at `steering/stage_spec.py:53-62` called `OBJECT_VOCAB`. It must match the input vocabulary of a trained policy — reordering or extending it would require retraining the policy. The cavity fixtures should NOT go in `OBJECT_VOCAB`.

The resolution path is:

```
composer prompt           (text)
  → get_affordance_map('... slider interior ...')
    → affordance LMP
      → parse_query_obj('slider_interior')
        → CalvinLMPInterface.detect('slider_interior')
          → CALVIN_FIXTURES['slider_interior']      ← this is all you need
```

The stage's `object` slot (which the policy attends to) stays at `drawer_handle` or `slider_handle`. The new fixtures are referenced only as affordance-query *string* targets.

### 4. Composer prompt — needs to USE the new fixtures

The composer prompt at `voxposer/prompts/calvin/composer_prompt.txt` has in-context examples that currently use handle positions as placement destinations (e.g. `'5cm above the slider handle'` for `place_in_slider`, `'5cm above the drawer handle'` for `push_into_drawer`). Those affordance-query strings need to reference the new cavity fixtures instead — something like `'a point at the slider interior'` / `'a point at the drawer interior'`.

Locate the affected in-context examples by grepping the file for `slider handle` and `drawer handle`. Update only the *affordance-query strings* in those examples — do not restructure the prompt or change unrelated examples. Keep your edits minimal and localized.

**Ask the user before editing the prompt.** They may want to handle this themselves or pair with you. If they ask you to do it, keep the change scoped to swapping handle references for interior references in the affordance queries of the place/push/close-style examples; leave everything else alone.

### 5. Smoke check

After (1)–(3) are done and (4) is decided, run a 1-episode smoke per affected task:

```bash
# place_in_slider
uv run python scripts/run_evaluation.py \
    --evaluation langsteer_primitive_object \
    --num-episodes 1 --tasks place_in_slider

# push_into_drawer
uv run python scripts/run_evaluation.py \
    --evaluation langsteer_primitive_object \
    --num-episodes 1 --tasks push_into_drawer
```

In the log, verify:
- The composer's emitted code references the new fixture name (e.g. `get_affordance_map('a point at the drawer interior')`) — only after step (4) has landed.
- The `Activated stage 1: ... target=[x, y, z]` line for stage 2 shows a target inside the cavity's bounding box (within ~5 cm of the centroid you put in `CALVIN_FIXTURES`).
- The episode runs to completion (success/fail doesn't matter — this smoke checks plumbing).

If step (4) was deferred (user handling prompt themselves), the composer will still emit the old handle-based query and the smoke just confirms resolution doesn't break.

---

## Files touched

- `voxposer/calvin_interface.py` — two new entries in `CALVIN_FIXTURES` (plus a `detect()` adjustment only if needed).
- `voxposer/prompts/calvin/composer_prompt.txt` — only if step (4) is greenlit by the user.

## Files NOT touched

- `steering/stage_spec.py` — `OBJECT_VOCAB` is policy-trained; don't modify.
- `voxposer/prompts/calvin/get_affordance_map_prompt.txt` — no change needed; the affordance LMP resolves names through the existing path.
- Anything in `policies/`, `envs/`, `scripts/`.

---

## Sequence

1. Read this plan.
2. Read `voxposer/calvin_interface.py` around lines 80–126 to see the existing `CALVIN_FIXTURES` format and the `detect()` method below it.
3. Wait for the user to provide the two bounding boxes.
4. Paste them into `CALVIN_FIXTURES` (step 1).
5. Run the Python sanity check (step 2) and confirm both names resolve.
6. Ask the user whether you should also update the composer prompt (step 4), or whether they'll handle that themselves.
7. Run the two smoke episodes (step 5) and report the stage-2 target coordinates from the log.

Stop and ask the user if anything is ambiguous — particularly if `detect()` doesn't resolve the new names after step (1), or if the composer prompt has examples in unexpected shapes.
