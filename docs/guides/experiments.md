# Quick Experimentation Guide

## Understanding Robot State and Running Custom Experiments

### 🤖 Robot State Structure (15 dimensions)

```python
robot_obs = [
    # Dimensions 0-5: TCP (Tool Center Point) - Computed via Forward Kinematics
    tcp_x, tcp_y, tcp_z,                    # [0-2] End-effector position (meters)
    tcp_roll, tcp_pitch, tcp_yaw,           # [3-5] Euler angles (radians)

    # Dimension 6: Gripper
    gripper_opening_width,                   # [6] Opening width (meters, 0-0.04)

    # Dimensions 7-13: Joint Angles - THE VALUES YOU SHOULD MODIFY
    joint_0, joint_1, joint_2,              # [7-9] Shoulder joints
    joint_3, joint_4, joint_5, joint_6,     # [10-13] Elbow and wrist joints

    # Dimension 14: Gripper Command
    gripper_action                           # [14] Binary: 1 (open) or -1 (close)
]
```

### ✅ Key Insight: Forward Kinematics

**When you modify joint angles (dimensions 7-13):**
- The TCP position/orientation (dimensions 0-5) is **automatically recomputed** via forward kinematics
- You don't need to manually calculate the TCP position
- PyBullet handles this using the robot's URDF kinematic chain

**Example:**
```python
# Original joint 1: 1.04 rad
# Modify to: 1.34 rad (+0.3 rad = +17 degrees)
robot_obs[8] = robot_obs[8] + 0.3

# After reset:
# - Joint 1 will be at 1.34 rad
# - TCP position will be automatically updated via forward kinematics!
```

---

## 🧪 Experiment 1: Test Custom Starting Joint Positions

**What it does:** Modifies specific joint angles and shows how the TCP position changes

```bash
cd /home/mlewkowicz/LangSteer
python scripts/test_custom_start_state.py

# Note: Set use_gui=True in the script to see the robot visualization
# The dataset is at: /home/mlewkowicz/calvin/dataset/task_D_D
```

**How to customize:**
1. Open `scripts/test_custom_start_state.py`
2. Modify the `joint_deltas` dictionary (line 51):
   ```python
   joint_deltas = {
       1: +0.3,  # Joint 1: shoulder lift (+17 degrees)
       3: -0.2,  # Joint 3: elbow (-11 degrees)
       5: +0.5,  # Joint 5: wrist pitch (+29 degrees)
   }
   ```
3. Set `use_gui: True` (line 64) to see visual rendering
4. Run the script

**Output:**
- Prints original vs modified joint angles
- Shows TCP position before/after (computed via FK)
- Displays the environment with the new starting pose

---

## 📹 Experiment 2: Visualize Camera Views

**What it does:** Shows RGB images from overhead static camera and gripper camera

```bash
python scripts/visualize_cameras.py
```

**Output:**
- Saves images to `./camera_views/`
  - `step_0000_static.png` - Overhead camera view
  - `step_0000_gripper.png` - Gripper camera view (if available)
- Displays cameras side-by-side using matplotlib

**Available cameras in CALVIN:**
- `rgb_static`: Overhead camera (200x200 RGB, fixed position)
- `rgb_gripper`: Wrist-mounted camera (200x200 RGB, moves with gripper)

**Tip:** Run actions and watch how the gripper camera view changes as the robot moves!

---

## 🎯 Experiment 3: Full Model Rollout with Custom Start State

**What it does:** Runs your trained DP3 model from a custom starting state and visualizes the trajectory

### Step 1: Modify the starting position

Edit `envs/calvin_utils/task_configs.py` line 55-73:

```python
robot_obs = np.array([
    # TCP will be recomputed - these values are placeholders
    0.02586889, -0.2313129, 0.5712808,      # [0-2] TCP position
    3.09045411, -0.02908596, 1.50013585,    # [3-5] TCP orientation
    0.07999963,                              # [6] Gripper width

    # MODIFY THESE JOINT ANGLES:
    -1.21779124 + 0.2,  # [7] Joint 0: +0.2 rad
    1.03987629,         # [8] Joint 1: unchanged
    2.11978254,         # [9] Joint 2: unchanged
    -2.34205014 - 0.3,  # [10] Joint 3: -0.3 rad
    -0.87015899,        # [11] Joint 4: unchanged
    1.64119093,         # [12] Joint 5: unchanged
    0.55344928,         # [13] Joint 6: unchanged

    1.0,  # [14] Gripper action
])
```

### Step 2: Run trajectory visualization

```bash
# Make sure you have a trained model checkpoint
python scripts/visualize_trajectories.py \
    visualization.num_rollouts=5 \
    visualization.max_steps=100 \
    visualization.save_snapshot=true \
    policy.checkpoint_path=path/to/your/checkpoint.pt
```

**Output:**
- Interactive 3D Plotly visualizations (HTML files)
- Shows end-effector trajectories from the modified start state
- Compares multiple rollouts from the same initial state
- Success/failure analysis

**Visualization files:**
- `outputs/trajectory_viz/scatter_plot.html` - 3D scatter plot
- `outputs/trajectory_viz/lines_plot.html` - 3D line trajectories
- `outputs/trajectory_viz/combined_plot.html` - Both combined

---

## 🔧 Advanced: Create Custom Snapshots

**Use case:** Save a specific environment state and replay it multiple times

```python
from utils.state_management.env_snapshots import EnvSnapshot, EnvSnapshotManager

# After modifying task_configs.py
env = CalvinEnvironment(cfg)
obs = env.reset()

# Capture the state
manager = EnvSnapshotManager()
snapshot = manager.capture_state(env)

# Save it
snapshot.save("snapshots/my_custom_start_state.pkl")

# Later: Load and restore
snapshot = EnvSnapshot.load("snapshots/my_custom_start_state.pkl")
manager.restore_state(env, snapshot)
```

---

## 📊 Useful Joint Angle Ranges

Franka Panda joint limits (radians):

| Joint | Name | Min | Max | Middle | Description |
|-------|------|-----|-----|--------|-------------|
| 0 | Shoulder pan | -2.90 | +2.90 | 0.0 | Rotates base left/right |
| 1 | Shoulder lift | -1.76 | +1.76 | 0.0 | Lifts arm up/down |
| 2 | Shoulder roll | -2.90 | +2.90 | 0.0 | Rolls shoulder |
| 3 | Elbow | -3.07 | -0.07 | -1.57 | Elbow bend (always negative!) |
| 4 | Wrist yaw | -2.90 | +2.90 | 0.0 | Rotates wrist |
| 5 | Wrist pitch | -0.02 | +3.75 | 1.87 | Pitches wrist |
| 6 | Wrist roll | -2.90 | +2.90 | 0.0 | Rolls gripper |

**Note:** Joint 3 (elbow) must always be negative! This is a physical constraint of the Franka arm.

---

## 💡 Common Modifications

### Reach Forward
```python
joint_deltas = {1: -0.3, 3: -0.2}  # Lower shoulder, extend elbow
```

### Reach Left
```python
joint_deltas = {0: +0.8}  # Rotate base left
```

### Reach Right
```python
joint_deltas = {0: -0.8}  # Rotate base right
```

### High Reach
```python
joint_deltas = {1: +0.4, 3: +0.3}  # Lift shoulder, straighten elbow
```

### Gripper Down (for pushing)
```python
joint_deltas = {5: +1.0}  # Pitch wrist down
```

---

## 🐛 Troubleshooting

**IK solver fails:**
- Your joint configuration might be at a singularity
- Try smaller joint angle modifications
- Check joint limits above

**TCP doesn't move as expected:**
- Remember: Joint angles → FK → TCP position
- Small joint changes can cause large TCP movements
- Test with `use_gui=True` to visualize

**Camera images are blank:**
- Make sure CALVIN dataset is downloaded
- Check that `dataset_path` points to correct location
- Verify rendering is enabled (don't use headless mode)

---

## 📚 Next Steps

1. **Start simple:** Modify 1-2 joints at a time
2. **Visualize:** Use `use_gui=True` to see changes live
3. **Test model:** Run trajectory visualization from custom states
4. **Analyze:** Compare success rates across different start poses
5. **Iterate:** Find challenging configurations to improve model robustness

Happy experimenting! 🚀
