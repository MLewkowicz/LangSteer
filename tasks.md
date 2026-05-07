
# Integrating RLBench into @envs/
- The goal of integrating RLBench is to facilitate similar 

# Integrating IsaacSim into @envs/
- The goal of integrating IsaacSim is to create several (2 for now) custom environments in which I can collect sufficient demonstrations of a Franka Emika Panda arm completing a task so that I can train both a language conditioned diffuser actor and an action primitive or potentially unconditioned diffuser actor and evaluate their performance.

- Stage I: once I configure the ZED camera the Isaac 

- Stage II: we have to decompose the environment into semantically labeled 3D bboxes. voxposer steering + visualization