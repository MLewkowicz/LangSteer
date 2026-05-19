"""DiffuserActor variant dispatch + back-compat alias.

The runner calls `build_diffuser_actor_policy(cfg_dict)`; this function reads
the three conditioning flags (`use_instruction`, `use_primitive_id`,
`use_object_id`), validates their pairwise constraints, and returns an
instance of one of four per-variant classes:

  * `LanguageDiffuserActorPolicy`         — CLIP text embeddings (default).
  * `NolangDiffuserActorPolicy`           — no instruction.
  * `PrimitiveDiffuserActorPolicy`        — (B, 1) primitive ids.
  * `PrimitiveObjectDiffuserActorPolicy`  — (B, 2) (primitive, object) ids.

A back-compat alias `DiffuserActorPolicy = build_diffuser_actor_policy` keeps
`from policies.diffuser_actor import DiffuserActorPolicy; DiffuserActorPolicy(cfg)`
working unchanged for any out-of-tree caller. The runners are encouraged to
use `build_diffuser_actor_policy` directly.
"""

from __future__ import annotations

from typing import Any

from policies.diffuser_actor_base import DiffuserActorBasePolicy


def build_diffuser_actor_policy(cfg: Any) -> DiffuserActorBasePolicy:
    """Factory: pick the right DiffuserActor variant from `cfg` flags."""
    use_primitive = cfg.get("use_primitive_id", False)
    use_object = cfg.get("use_object_id", False)
    use_instruction = cfg.get("use_instruction", True)

    # Cross-validation — surface yaml-level errors before model construction.
    if use_primitive and not use_instruction:
        raise ValueError(
            "use_primitive_id=True requires use_instruction=True "
            "(primitive mode reuses the instruction cross-attention pipeline)."
        )
    if use_object and not use_primitive:
        raise ValueError("use_object_id=True requires use_primitive_id=True.")

    if use_object:
        from policies.diffuser_actor_primitive_object import (
            PrimitiveObjectDiffuserActorPolicy,
        )

        return PrimitiveObjectDiffuserActorPolicy(cfg)
    if use_primitive:
        from policies.diffuser_actor_primitive import (
            PrimitiveDiffuserActorPolicy,
        )

        return PrimitiveDiffuserActorPolicy(cfg)
    if use_instruction:
        from policies.diffuser_actor_language import (
            LanguageDiffuserActorPolicy,
        )

        return LanguageDiffuserActorPolicy(cfg)

    from policies.diffuser_actor_nolang import NolangDiffuserActorPolicy

    return NolangDiffuserActorPolicy(cfg)


# Back-compat shim — `DiffuserActorPolicy(cfg)` keeps working as if it were a class.
DiffuserActorPolicy = build_diffuser_actor_policy
