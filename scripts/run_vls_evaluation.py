"""VLS evaluation on CALVIN (34 tasks).

Runs Vision-Language Steering end-to-end on the same 34 CALVIN tasks as the
LangSteer baseline, using identical deterministically pre-sampled starting
conditions for a fair comparison. Results are saved in LangSteer's JSON schema.

Usage:
    uv run python scripts/run_vls_evaluation.py --evaluation vls
    uv run python scripts/run_vls_evaluation.py --evaluation vls --num-episodes 25
    uv run python scripts/run_vls_evaluation.py --evaluation vls \\
        --output-dir outputs/evaluation/2026-05-01_12-00-00
    # Skip VLM queries by reusing cached guidance functions:
    uv run python scripts/run_vls_evaluation.py --evaluation vls \\
        --cached-functions-dir outputs/vls_guidance_cache/open_drawer

Resume: pass --output-dir pointing to an existing run directory to pick up where
the run left off (same semantics as run_evaluation.py).
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
VLS_ROOT = REPO_ROOT / "third_party" / "vls"
for _p in [str(REPO_ROOT), str(VLS_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Monkey-patch calvin_env Light before any CALVIN imports so scene state dict
# matches what LangSteerCalvinAdapter expects (same patch VLS main.py applies).
from patches.light import Light as _PatchedLight, LightState as _PatchedLightState
import types as _types
_light_mod = _types.ModuleType("calvin_env.scene.objects.light")
_light_mod.Light = _PatchedLight
_light_mod.LightState = _PatchedLightState
sys.modules["calvin_env.scene.objects.light"] = _light_mod

from omegaconf import OmegaConf

from scripts.run_evaluation import (
    presample_starting_conditions,
    load_results,
    save_results,
    get_completed_episodes,
    deterministic_hash,
    set_seed,
    print_summary,
)
from envs.calvin_utils.gym_wrapper import CalvinGymWrapper, _find_calvin_data_dir
from envs.calvin_utils.language_ann import load_language_annotations, get_instruction_for_task
from envs.vls_calvin_adapter import LangSteerCalvinAdapter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLS evaluation on CALVIN")
    p.add_argument("--evaluation", required=True,
                   help="Evaluation config name from conf/evaluation/ (e.g. vls)")
    p.add_argument("--num-episodes", type=int, default=25, help="Episodes per task")
    p.add_argument("--max-steps", type=int, default=360, help="Max env steps per episode")
    p.add_argument("--seed", type=int, default=42, help="Random seed for starting conditions")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory; if it contains existing results the run resumes")
    p.add_argument("--cached-functions-dir", type=str, default=None,
                   help="Directory of pre-generated VLS guidance functions (skips VLM queries)")
    p.add_argument("--no-guidance", action="store_true",
                   help="Disable VLS guidance (run base policy only, no VLM queries)")
    return p.parse_args()


def _load_task_oracle():
    from calvin_env.envs.tasks import Tasks
    tasks_cfg_path = _find_calvin_data_dir().parent / "conf" / "tasks" / "new_playtable_tasks.yaml"
    tasks_cfg = OmegaConf.load(str(tasks_cfg_path))
    return Tasks(tasks_cfg.tasks)


def _load_vls_policy(vls_cfg: dict, adapter: LangSteerCalvinAdapter):
    """Load DiffusionPolicySteer and wire up pre/post-processors."""
    sys.path.insert(0, str(VLS_ROOT))
    from core.diffusion_policy_steer import DiffusionPolicySteer
    from lerobot.policies.factory import make_pre_post_processors

    pretrained = vls_cfg.get("policy_pretrained", "Vision-Language-Steering/vls_calvin_base")
    logger.info(f"Loading VLS policy from: {pretrained}")
    policy = DiffusionPolicySteer.from_pretrained(pretrained)
    policy.to("cuda")
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=pretrained,
        preprocessor_overrides=preprocessor_overrides,
    )
    policy.post_init(
        adapter=adapter,
        postprocessor=postprocessor,
        sample_batch_size=vls_cfg.get("sample_batch_size", 20),
        policy_config=None,
    )
    logger.info("VLS policy loaded and wired")
    return policy, preprocessor


def _load_perception_components(vls_cfg: dict, adapter: LangSteerCalvinAdapter):
    """Load KeypointDetector and VLMAgent (heavy models loaded once)."""
    sys.path.insert(0, str(VLS_ROOT))
    from core.keypoint_detector import KeypointDetector
    from core.keypoint_tracker import KeypointTracker
    from vlm_query.vlm_agent import VLMAgent

    kp_cfg = dict(vls_cfg.get("keypoint_detector", {}))
    kp_cfg.setdefault("device", "cuda")
    kp_cfg.setdefault("seed", 42)
    keypoint_detector = KeypointDetector(config=kp_cfg)

    template_dir = str(VLS_ROOT / "vlm_query")
    vlm_config = {
        "api_key": vls_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        "model": vls_cfg.get("model", "gpt-4o"),
        "temperature": vls_cfg.get("temperature", 1.0),
        "max_completion_tokens": vls_cfg.get("max_completion_tokens", 2000),
        "query_template_dir": template_dir,
    }
    vlm_agent = VLMAgent(config=vlm_config, base_dir="outputs/vls_eval", env_type="calvin")

    keypoint_tracker = KeypointTracker(adapter)
    return keypoint_detector, vlm_agent, keypoint_tracker


def _prepare_episode_guidance(
    adapter: LangSteerCalvinAdapter,
    keypoint_detector,
    keypoint_tracker,
    vlm_agent,
    instruction: str,
    episode_dir: Path,
    cached_functions_dir: Optional[str],
) -> Optional[Dict]:
    """Run keypoint detection and VLM query; return guidance_fns dict or None on failure."""
    sys.path.insert(0, str(VLS_ROOT))
    from utils.guidance_utils import load_functions_from_txt
    import json

    try:
        rgb, depth, points, segmentation, segment_id_to_name = (
            adapter.get_keypoint_detection_inputs()
        )
    except Exception as e:
        logger.error(f"Keypoint detection input failed: {e}")
        return None

    try:
        key_points, projected_img, mask_ids = keypoint_detector.get_keypoints(
            rgb=rgb, points=points,
            segmentation=segmentation, segment_id_to_name=segment_id_to_name,
        )
    except Exception as e:
        logger.error(f"Keypoint detection failed: {e}")
        return None

    key_points_objects_map = keypoint_tracker.register_keypoints(
        key_points, mask_ids=mask_ids, segment_id_to_name=segment_id_to_name
    )

    if cached_functions_dir is not None:
        guidance_functions_dir = cached_functions_dir
        logger.info(f"Using cached guidance functions from: {guidance_functions_dir}")
    else:
        vlm_agent.task_dir = str(episode_dir / "vlm_agent")
        os.makedirs(vlm_agent.task_dir, exist_ok=True)
        metadata = {
            "init_keypoint_positions": key_points,
            "num_keypoints": len(key_points),
            "key_points_objects_map": key_points_objects_map,
        }
        try:
            guidance_functions_dir = vlm_agent.generate_guidance(
                projected_img, instruction, metadata
            )
        except Exception as e:
            logger.error(f"VLM guidance generation failed: {e}")
            return None

    meta_path = Path(guidance_functions_dir) / "metadata.json"
    if not meta_path.exists():
        logger.error(f"metadata.json not found at {guidance_functions_dir}")
        return None

    with open(meta_path) as f:
        program_info = json.load(f)

    guidance_fns: Dict[int, List] = {}
    for stage in range(1, program_info["num_stages"] + 1):
        load_path = Path(guidance_functions_dir) / f"stage{stage}_guidance.txt"
        guidance_fns[stage] = load_functions_from_txt(str(load_path)) if load_path.exists() else []

    logger.info(f"Loaded {len(guidance_fns)} guidance stage(s)")
    return guidance_fns


def _run_one_episode(
    adapter: LangSteerCalvinAdapter,
    policy,
    preprocessor,
    keypoint_tracker,
    guidance_fns: Optional[Dict],
    vls_cfg: dict,
    max_steps: int,
    use_guidance: bool,
) -> Tuple[bool, int, float]:
    """Inner step loop — mirrors VLS Main._run_episode() without video/plotting."""
    action_horizon = policy._action_chunk_horizon
    action_executed = 0
    action_chunk = None
    global_steps = 0
    done = False
    success = False
    total_reward = 0.0
    current_stage = 1

    guide_scale = vls_cfg.get("guide_scale", 80.0)
    sigmoid_k = vls_cfg.get("sigmoid_k", 12.0)
    sigmoid_x0 = vls_cfg.get("sigmoid_x0", 0.7)
    start_ratio = vls_cfg.get("start_ratio", None)
    use_diversity = vls_cfg.get("use_diversity", True)
    diversity_scale = vls_cfg.get("diversity_scale", 10.0)
    mcmc_steps = vls_cfg.get("MCMC_steps", 4)
    use_fkd = vls_cfg.get("use_fkd", False)

    current_guidance_fns = (guidance_fns or {}).get(current_stage, []) if use_guidance else None

    while not done and global_steps < max_steps:
        generate_new_chunk = (action_executed == 0)

        if generate_new_chunk and use_guidance and guidance_fns:
            keypoints = keypoint_tracker.get_keypoint_positions()
        else:
            keypoints = None

        observation = preprocessor(adapter.get_policy_observation(
            sample_num=vls_cfg.get("sample_batch_size", 20)
        ))

        action_chunk = policy.select_action(
            observation,
            generate_new_chunk=generate_new_chunk,
            use_guidance=use_guidance and bool(current_guidance_fns),
            keypoints=keypoints,
            guidance_fns=current_guidance_fns,
            guide_scale=guide_scale,
            sigmoid_k=sigmoid_k,
            sigmoid_x0=sigmoid_x0,
            start_ratio=start_ratio,
            use_diversity=use_diversity,
            diversity_scale=diversity_scale,
            MCMC_steps=mcmc_steps,
            use_fkd=use_fkd,
            global_step=global_steps,
            current_stage=current_stage,
        )

        obs, reward, terminated, truncated, info = adapter.step(
            action_chunk[0][action_executed]
        )
        total_reward += float(reward)
        global_steps += 1
        action_executed = (action_executed + 1) % action_horizon

        if terminated or truncated:
            success = info.get("success", False)
            done = True

    return success, global_steps, total_reward


def run_vls_condition(
    eval_cfg: dict,
    hydra_cfg,
    starting_conditions: Dict,
    sampled_episode_ids: Dict,
    num_episodes: int,
    max_steps: int,
    base_seed: int,
    results_path: Path,
    use_guidance: bool,
    cached_functions_dir: Optional[str],
    output_dir: Path,
):
    vls_cfg = eval_cfg.get("vls", {})
    task_instructions = load_language_annotations(hydra_cfg.env.lang_ann_path)

    # --- One-time initialization of expensive components ---
    logger.info("Initializing CALVIN environment...")
    gym_wrapper = CalvinGymWrapper(
        dataset_path=str(hydra_cfg.env.dataset_path),
        split=hydra_cfg.env.split,
        use_gui=False,
    )

    task_oracle = _load_task_oracle()
    env_config = {
        "vlm_camera": "static",
        "max_episode_steps": max_steps,
    }
    adapter = LangSteerCalvinAdapter(
        raw_calvin_env=gym_wrapper._env,
        env_config=env_config,
        task_oracle=task_oracle,
        device="cuda",
    )

    logger.info("Loading VLS policy...")
    policy, preprocessor = _load_vls_policy(vls_cfg, adapter)

    if use_guidance:
        logger.info("Loading perception components (DINOv3, VLM agent)...")
        keypoint_detector, vlm_agent, keypoint_tracker = _load_perception_components(
            vls_cfg, adapter
        )
    else:
        keypoint_detector = vlm_agent = keypoint_tracker = None

    # --- Result file setup ---
    results = load_results(results_path)
    results.setdefault("condition_name", eval_cfg.get("condition_name", "VLS"))
    results.setdefault("seed", base_seed)
    results.setdefault("max_steps", max_steps)
    results.setdefault("perturbation_axis", None)
    results.setdefault("tasks", {})

    # --- Task / episode loop ---
    for task_name in eval_cfg.get("tasks", []):
        instruction = get_instruction_for_task(task_name, task_instructions)
        conditions = starting_conditions.get(task_name, [])
        ep_ids = sampled_episode_ids.get(task_name, [])
        completed = get_completed_episodes(results, task_name)

        results["tasks"].setdefault(task_name, {"instruction": instruction, "episodes": []})

        if not conditions:
            logger.warning(f"No starting conditions for task '{task_name}', skipping")
            continue

        logger.info(f"\nTask: {task_name}  ({instruction})")

        for ep_idx in range(completed, num_episodes):
            set_seed(base_seed + deterministic_hash(task_name) + ep_idx)

            robot_obs, scene_obs = conditions[ep_idx]
            adapter.stage_starting_condition(robot_obs, scene_obs, task_name, instruction)
            policy.reset()

            episode_dir = output_dir / "episodes" / task_name / f"ep{ep_idx:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Keypoint detection + VLM guidance (once per episode; VLM response is cached
            # to disk so re-runs skip the API call when cached_functions_dir is provided)
            guidance_fns = None
            if use_guidance:
                # Reset happens inside _prepare_episode_guidance via adapter.reset()
                adapter.reset()
                keypoint_tracker.keypoints = {}
                guidance_fns = _prepare_episode_guidance(
                    adapter, keypoint_detector, keypoint_tracker, vlm_agent,
                    instruction, episode_dir,
                    cached_functions_dir or vls_cfg.get("cached_functions_dir"),
                )
                if guidance_fns is None:
                    logger.warning(f"  Episode {ep_idx}: guidance prep failed, running without guidance")
                    use_guidance_this_ep = False
                else:
                    use_guidance_this_ep = True
                # Re-stage so _run_one_episode's first action uses correct start state
                adapter.stage_starting_condition(robot_obs, scene_obs, task_name, instruction)
                adapter.reset()
            else:
                adapter.reset()
                use_guidance_this_ep = False

            success, steps, reward = _run_one_episode(
                adapter, policy, preprocessor, keypoint_tracker,
                guidance_fns, vls_cfg, max_steps, use_guidance_this_ep,
            )

            results["tasks"][task_name]["episodes"].append({
                "episode_idx": ep_idx,
                "source_episode_id": int(ep_ids[ep_idx]) if ep_idx < len(ep_ids) else -1,
                "success": success,
                "steps": steps,
                "reward": float(reward),
            })
            save_results(results_path, results)

            status = "SUCCEEDED" if success else "FAILED"
            logger.info(f"  Episode {ep_idx+1}/{num_episodes} {status} (steps={steps})")

        eps = results["tasks"][task_name]["episodes"]
        n_success = sum(1 for e in eps if e["success"])
        logger.info(f"  -> {task_name}: {n_success}/{len(eps)} succeeded")

    print_summary(results)
    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs/evaluation") / f"vls_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation config
    conf_dir = REPO_ROOT / "conf"
    eval_cfg_path = conf_dir / "evaluation" / f"{args.evaluation}.yaml"
    if not eval_cfg_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {eval_cfg_path}")
    eval_cfg = OmegaConf.to_container(OmegaConf.load(eval_cfg_path), resolve=True)

    # Load base hydra config to get env settings (dataset_path, split, etc.)
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        hydra_cfg = compose(config_name="config")

    # Pre-sample starting conditions (deterministic — same as run_evaluation.py)
    logger.info("Pre-sampling starting conditions...")
    starting_conditions, sampled_episode_ids = presample_starting_conditions(
        dataset_path=Path(hydra_cfg.env.dataset_path),
        split=hydra_cfg.env.split,
        task_list=eval_cfg["tasks"],
        num_episodes=args.num_episodes,
        seed=args.seed,
    )

    condition_name = eval_cfg.get("condition_name", "VLS")
    safe_name = condition_name.replace(" ", "_").lower()
    results_path = output_dir / f"{safe_name}.json"

    use_guidance = not args.no_guidance and eval_cfg.get("vls", {}).get("use_guidance", True)

    run_vls_condition(
        eval_cfg=eval_cfg,
        hydra_cfg=hydra_cfg,
        starting_conditions=starting_conditions,
        sampled_episode_ids=sampled_episode_ids,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        base_seed=args.seed,
        results_path=results_path,
        use_guidance=use_guidance,
        cached_functions_dir=args.cached_functions_dir,
        output_dir=output_dir,
    )

    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
