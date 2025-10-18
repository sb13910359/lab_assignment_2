import os
import numpy as np
from spatialmath import SE3
import spatialgeometry as geometry
import swift


class Human:
    """
    Represents a human composed of multiple STL body parts that move together.
    Includes movement controls and button interface for manual motion.
    """

    def __init__(self, env, current_dir, start_pose=SE3(8.7, 5.7, 0), scale=(0.4, 0.4, 0.4)):
        self.env = env
        self.start_pose = start_pose
        self.scale = scale
        self.current_dir = current_dir

        # --- Load STL meshes ---
        self.human = geometry.Mesh(
            os.path.join(current_dir, "human.stl"),
            pose=start_pose, color=[0.95, 0.80, 0.63, 1], scale=scale)
        self.human1 = geometry.Mesh(
            os.path.join(current_dir, "human5.stl"),
            pose=start_pose, color=[0.29, 0.18, 0.02, 1], scale=scale)
        self.human2 = geometry.Mesh(
            os.path.join(current_dir, "human_hair.stl"),
            pose=start_pose, color=[0.44, 0.31, 0.22, 1], scale=scale)
        self.human3 = geometry.Mesh(
            os.path.join(current_dir, "human_shirt.stl"),
            pose=start_pose, color=[1.0, 0.95, 0.0, 1], scale=scale)

        # --- Add all parts to environment ---
        env.add(self.human)
        env.add(self.human1)
        env.add(self.human2)
        env.add(self.human3)

        # --- Create control buttons ---
        self.btn_forward = swift.Button(desc="↑ Forward", cb=lambda _: self.move(0, -0.1, 0))
        self.btn_turnL = swift.Button(desc="⟲ Turn Left", cb=lambda _: self.move(0, 0, np.pi/8))
        self.btn_turnR = swift.Button(desc="⟳ Turn Right", cb=lambda _: self.move(0, 0, -np.pi/8))

        env.add(self.btn_forward)
        env.add(self.btn_turnL)
        env.add(self.btn_turnR)

    # ------------------------------------------------------------
    # 🔧 Core motion updates
    # ------------------------------------------------------------
    def sync_body(self):
        """Synchronize all body parts to the main mesh position."""
        T = self.human.T
        self.human1.T = T
        self.human2.T = T
        self.human3.T = T

    def move(self, delta_x=0, delta_y=0, delta_yaw=0):
        """Move or rotate the human in the environment (always responsive)."""
        T = self.human.T
        T = T * SE3(delta_x, delta_y, 0) * SE3.Rz(delta_yaw)
        self.human.T = T
        self.sync_body()
        try:
            # Force environment step even if other threads halted
            self.env.step(0.02)
        except Exception:
            pass
