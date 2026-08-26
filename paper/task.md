You will be coordinating a team of two agents to polish a paper on the method constructed in this repository in latex format in @draft.md. YOU DO NOT HAVE PERMISSION TO CHANGE THE TEMPLATE OR ANY STYLING FEATURES (THESE MUST ALL BE APPROVED BY ME) but you may freely edit content within sections / add sections/  Some details about how to get started are in @vision.md. One agent will have editing permission and edits to draft.md should only be able to go through if a second reviewer agent approves and believes it improves the quality of the report. If there is room to make certain sections more succint, please do so. 

THIS IS A CONFERENCE PAPER, THIS IS NOT A REPORT OF HOW THIS CODEBASE EXACTLY WORKS. WE NEED TO PULL BACK ON THE LEVEL OF SPECIFICITY AND HAVE A HIGHER LEVEL FRAMING FOR THE PAPER.

We should not be saying things like "language conditioned models tend to use CLIP embedding in their backbone." 3D Diffuser actor is a specific implementation of a language conditioning architecture but this method should be agnostic to the different types of language conditioning. We should not be provided extremely specific evaluation conditions in our abstract or introduction. 

ALWAYS CONSIDER THE HIGHER LEVEL FRAMING, THIS SHOULD NOT READ LIKE A STEP BY STEP RECAP OF EVERYTHING IN THE REPOSITORY. 

I will be steering this editing process.

We should always make sure that we are not overloading the reader with repository specific information. Let's run through this refinement section by section.

abstract
introduction
related works - You should independently try to identify the relevant literature that should be mentioned although a few are already brought up in the @related_works.md
method - we need to iterate on the succinctness of this section
results - we should present our simulation results (even if the results are pending) and motivate them for base language + language perturbations. We will be evaluating the 
- baseline (just language conditioned) 
- language conditioned + steering
- action + object primitive + steering
The idea here is that the language conditioned model might have competing objectives with the steering module and that it might be better for the LLM to control your stage generation and your conditioning for the base model. We are also considering adding VLS to the CALVIN evals

We should then move on to the Isaac + real world experiments which are meant to demonstrate the different types of preference elicitation that can be done with this steering module and then the real world experiment which shows that this can be done with a real world perception pipeline that runs ICP on 3D meshes extracted by SAM3D and assigns semantic labels that the LLM can use for value map generation.

Create a team of two agents. 

## Agent 1: Planner and Reviewer
The planning agent's goal is to organize the structure of the paper and provide a good skeleton for the writing agent to follow. The planning agent will not only plan, but also be in a constant feedback loop with the writing agent, reviewing its edits to make sure they are in line with the plan. This agent should also provide a succinct set of bullet points for each paragraph to guide the writing process of the writing agent. These succinct bullet points should highlight the meta idea that the paragraph is aiming to convey. An example of this is above the introduction in blue. (THIS SHOULD BE VERY SUCCINCT, MAIN IDEA / PURPOSE OF THE PARAGRAPH)
Agent type: Opus 4.7 in max thinking mode

## Agent 2: Writer
Subtype: writing-agent
This agent will be responsible for the majority of the writing. It will be receiving high level guidance from the planner and reviewer agent although it can push back and request a different plan / different framing for a particular section.
Agent type: Opus 4.7 in max thinking mode
We need to be aware of the page limit of the paper. We cannot provide every single detail to explain the method.

