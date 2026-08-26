Create a team of 2 agents using Opus 4.8 [1M] in max thinking mode

## Agent 1: editor-one
This writing agent should be in a constant feedback loop with the other writing agent.
Agent Subtype: writing-agent

## Agent 2: editor-two
Agent Subtype: writing-agent
This agent will be suggesting various revisions. It should push back on the other writing agent and request / suggest a different framing for a particular section.
Agent type: Opus 4.8 in max thinking mode

Goal: You will be coordinating a team of two agents to write the appendix of the paper in @draft.md. YOU DO NOT HAVE PERMISSION TO CHANGE THE TEMPLATE OR ANY STYLING FEATURES (THESE MUST ALL BE APPROVED BY ME) and also you do not have permission to change anything in the paper. The prose in the paper is somewhat polished, so we should try to imitate this when writing the appendix.

THIS IS A CONFERENCE PAPER, THIS IS NOT A REPORT OF HOW THE CODE FOR THIS METHOD EXACTLY WORKS. WE NEED TO PULL BACK ON THE LEVEL OF SPECIFICITY.

Create a team of two agents. Both agents will be suggesting edits to the paper in order to wrte up the appendix sections. They should reach a consensus on how to write a section / sentence. The idea is to first have a hyperparameter section that will first document all the necessary components that are needed to understand the base policy, the parameters used for steering, how it varies over task (I gave some pointers in the comments within draft.md). In this section we jshould talk about adaptive guidance, that is we shrink the guidance as the task progressing according to some schedule. Then we will effectively perform a model ablation to motivate our structured conditioning. We are running p0-p4 on the action conditioned and the unconditioned model to motivate why we needed the action + object conditioning. We will also have to have a section describing the prompts used (although we should only extract the high level prompts (NOT ALL THE IN CONTEXT EXAMPLES OR THE SPECIFIC HACKS THAT WERE USED FOR EACH TASK)).