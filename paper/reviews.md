Strong assumptions on perception and external VLM

R1: The method assumes strong perception and VLM infrastructure without evaluating their failures. The composer receives labeled 3D bounding boxes and generates executable programs; the authors acknowledge that perception errors and hallucinated code directly affect mode selection. The evaluation should perturb bounding boxes, object labels, and generated programs, and report composer accuracy separately from control success.

R2: Clarifying whether the semantic bounding boxes come from simulator ground truth or a learned, and evaluating sensitivity to localization errors, would strengthen the evaluation.

R3: LangSteer assumes that the VLM composer receives an RGB snapshot and a set of labeled 3D bounding boxes for objects in the workspace. The composer then emits programs that construct value maps by referencing these bounding boxes. This is a strong assumption for unstructured real-world environments. The paper does not sufficiently evaluate robustness to missing objects, incorrect labels, noisy bounding boxes, occlusions, or hallucinated program outputs.

Strong indications to use a real perception pipeline at least somewhere. 

Limited Baselines and novelty of language-to-value-map component

R1: The novelty of the language-to-value-map component is limited. The paper explicitly states that it “adopts this value-map representation” from prior work, while VoxPoser and related methods already generate scene-grounded spatial programs and constraints from language. The primary novelty is therefore the injection of their gradients into diffusion denoising. The paper should more clearly narrow its contribution and experimentally isolate why this integration is superior to existing ways of consuming the same maps.

Have the authors compared the proposed method with recent LLM-based planners or VLA foundation models on language perturbation benchmarks? 

R2: Have the authors compared the proposed method with recent LLM-based planners or VLA foundation models on language perturbation benchmarks? If not, such a comparison would better position the proposed approach relative to current methods.

R3: hard to attribute the gains. For example, the strong CALVIN results may largely come from replacing a brittle CLIP text embedding with a structured symbolic conditioning pair, rather than from value-map gradient steering itself. The paper includes a “+Steering” baseline, but this does not fully answer whether the full improvement comes from the VLM composer, the structured conditioning, or the dense value-map guidance. A more careful ablation is needed.

R3: A critical missing baseline is sample-and-rerank: sample multiple trajectories from the same frozen diffusion policy and select the trajectory with the lowest value-map cost. This would test whether injecting gradients into the denoising process is necessary, or whether the gains mainly come from the VLM-derived value maps.

R3: Another missing baseline is a stronger VLM/VLA-based planner or re-ranking method using the same detected objects and language input. Since the method relies heavily on VLM reasoning and structured scene grounding, the fair comparison should include methods that use the same information but do not use classifier guidance.

Arguing for some re-ranking baseline + VLA + code-as-policies style baseline

Tasks do not warrant policy steering

R1: The tasks do not convincingly require nontrivial diffusion-policy steering. The custom tasks mainly ask for placements “behind,” “left of,” or on the “right side” of an object in controlled scenes. Once the VLM identifies the desired free-space region, guidance may largely amount to pulling the trajectory toward that region. Comparisons against extracting a target pose, waypoint conditioning, trajectory reranking, or conventional geometric planning are needed to show that gradient injection into denoising is necessary.

R1: The paper does not quantify the diversity of its language variants. It reports task success but not the number of distinct paraphrases per task, lexical or syntactic diversity, human naturalness ratings, or semantic-equivalence judgments. Repeating environment trials for one constructed instruction does not establish generalization across human language. The authors should evaluate multiple independently generated or human-written instructions for each semantic category.


Physical evaluation is largely qualitative and Isaac experiments don't cover enough task variety and not robust enough (base policy performance).

R1: The claim of broad instruction robustness exceeds the tested language variation. P1–P3 consist of synonym substitution, syntactic restructuring, and added irrelevant detail; P4 removes identifying information so that the scene resolves the referent. These tests cover surface variation and visual disambiguation, but not fundamentally different specification forms such as behavior-level commands, desired outcomes, functional requirements, constraints, preferences, or latent intent. The wine-glass example is closer to intent inference, but it is a single anecdotal demonstration.

R2: The paper does not clearly describe how the regions of interest are obtained in the Isaac Extensions experiments.

R3: The real-world wine-glass experiment is visually compelling but appears to be mainly qualitative. The paper describes one scenario where the unsteered policy defaults to placing the glass upright in the cabinet, while LangSteer steers it toward the inverted rack mode using an affordance map and rotation target. However, the paper does not report repeated real-world trials, success rates, variance across initial conditions, comparison with VLS, or failure cases. For a CoRL paper making claims about real-world human-centered manipulation, a single qualitative demonstration is not enough to support the broader claim.

R3: the custom simulation evaluation uses only two Isaac scenes and ten instructions per scene. Table 1 reports small-count results such as 6/10, 5/10, 7/10, and 4/10 for LangSteer, which are promising but not strong enough to establish generality. Moreover, even LangSteer succeeds on only around half of these preference-satisfaction instructions, which suggests that the method is still far from robust. The paper should discuss these modest absolute success rates more directly.


Need to have improved base policy and larger task selection for Isaac. Potentially need different steering objectives in the Isaac experiments (i.e. waypoint steering?). Need real-world evaluation on ~3 or more tasks.

Need to show that this can be an inference time module

R1: The method is not directly applied to an unchanged pretrained checkpoint. The authors “retrain the policy once,” replacing the original CLIP language pathway with fixed skill and object embeddings. The policy is frozen only after this restructuring. The paper should avoid implying that LangSteer is a plug-in inference-time method for arbitrary existing language-conditioned policies.

Should I have some baseline or alternative architecture that first compresses the language input but keeps the conditioning mechanism?

Multi-goal disambiguation?

R2: The proposed method uses language to disambiguate task goals. How does the baseline resolve ambiguity when multiple valid goal states exist for the same manipulation task (e.g., placing a cup in one of several possible locations)?

I think the type of tasks in Isaac can admit more complicated goal regions than I have currently.

Avoidance maps + Failure modes 

R3: Because LangSteer injects gradients into a robot policy at inference time, incorrect value maps could actively steer the robot toward unsafe regions or fragile objects. The paper’s motivating example includes “Watch out for the vase,” but there is no quantitative evaluation of avoidance reliability, safety margins, or behavior under wrong avoidance maps.



Main questions:
What should the task set be for Isaac so that we can demonstrate the usefulness of a dense costmap over over other steering signals
Edge of table
Wiping a table or whiteboard?
Baselines on the physical system
version of LangSteer where the initial language input is distilled into some simpler form via a VLM but still use the same language conditioning pathway for the base model
