import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import time
import os
from math import pi
from ir_support import CylindricalDHRobotPlot

# -----------------------------------------------------------------------------------#
class UR3_Scaled(DHRobot3D):
    def __init__(self, scale=1.5):
        """
        UR3 robot scaled up by a factor of 1.5 
        """
        self.scale = scale
        links = self._create_DH(scale)

        #assign gripper
        self.gripper = UR3Gripper()

        # Link file names
        link3D_names = dict(
            link0='base_ur3_scaled',
            link1='shoulder_ur3_scaled',
            link2='upperarm_ur3_scaled',
            link3='forearm_ur3_scaled',
            link4='wrist1_ur3_scaled',
            link5='wrist2_ur3_scaled',
            link6='wrist3_ur3_scaled'
        )

        # Reference joint configuration
        qtest = [0, -pi/2, 0, 0, 0, 0]

        # Adjusted 3D transforms — each link scaled and correctly chained
        s = scale
        qtest_transforms = [
            spb.transl(0, 0, 0),  # base
            spb.transl(0, 0, 0.15239 * s) @ spb.trotz(pi),
            spb.transl(0, 0, 0.15239 * s)  # attach shoulder
            @ spb.transl(0, -0.12 * s, 0) @ spb.trotz(pi),
            spb.transl(0, -0.12 * s, 0.1524 * s)
            @ spb.transl(0, -0.027115 * s, 0.24343 * s) @ spb.trotz(pi),
            spb.transl(0, -0.027316 * s, 0.60903 * s)
            @ spb.rpy2tr(0, -pi/2, pi, order='xyz'),
            spb.transl(0.000389 * s, -0.11253 * s, 0.60902 * s)
            @ spb.rpy2tr(0, -pi/2, pi, order='xyz'),
            spb.transl(-0.083765 * s, -0.11333 * s, 0.61096 * s)
            @ spb.trotz(pi)
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "meshes", "robot2")
        super().__init__(
            links,
            link3D_names,
            name=f'UR3_{scale:.1f}x',
            link3d_dir=mesh_dir,
            qtest=qtest,
            qtest_transforms=qtest_transforms,
        )

        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self, scale):
        """Scaled DH model"""
        s = scale
        a = [0, -0.24365 * s, -0.21325 * s, 0, 0, 0]
        d = [0.1519 * s, 0, 0, 0.11235 * s, 0.08535 * s, 0.0819 * s]
        alpha = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
        qlim = [[-2 * pi, 2 * pi] for _ in range(6)]

        links = [
            rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i])
            for i in range(6)
        ]
        return links

    # -----------------------------------------------------------------------------------#
    def test(self):
        
        env = swift.Swift()
        env.launch(realtime=True)
        self.q = self._qtest
        self.base = SE3(0.5, 0.5, 0.0)
        self.add_to_env(env)

        q_goal = [self.q[i] - pi / 3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q

        for q in qtraj:
            self.q = q
            env.step(0.02)
        time.sleep(2)
        env.hold()

class UR3Gripper:
    def __init__(self):
        # DH Links for both jaws
        l2_1 = rtb.DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
        l2_2 = rtb.DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
        self.left = rtb.DHRobot([l2_1, l2_2], name="gripper_left")

        r2_1 = rtb.DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
        r2_2 = rtb.DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
        self.right = rtb.DHRobot([r2_1, r2_2], name="gripper_right")

        # Geometries
        self.g_left = CylindricalDHRobotPlot(self.left, cylinder_radius=0.01, color="#1E324D")
        self.g_right = CylindricalDHRobotPlot(self.right, cylinder_radius=0.01, color="#1E324D")

        self.geom_left = self.g_left.create_cylinders()
        self.geom_right = self.g_right.create_cylinders()

        # Define open/close poses
        self.q1_open = [-pi / 2.5, pi / 3.5]
        self.q2_open = [pi / 2.5, -pi / 3.5]
        self.q1_close = [-pi / 3, pi / 4]
        self.q2_close = [pi / 3, -pi / 4]

        # Precompute trajectories
        self.traj_open_L = rtb.jtraj(self.q1_close, self.q1_open, 50).q
        self.traj_open_R = rtb.jtraj(self.q2_close, self.q2_open, 50).q
        self.traj_close_L = rtb.jtraj(self.q1_open, self.q1_close, 50).q
        self.traj_close_R = rtb.jtraj(self.q2_open, self.q2_close, 50).q

        # Initialize state
        self.geom_left.q = self.q1_open
        self.geom_right.q = self.q2_open

    def attach_to_robot(self, robot: DHRobot3D):
        """Attach gripper to UR3’s end-effector."""
        arm_T = robot.fkine(robot.q) * SE3(0.03, 0, 0)
        adjust = SE3.Ry(-pi / 2) * SE3(0, 0, 0.03) * SE3.Rx(-pi / 2)
        self.geom_left.base = arm_T * adjust
        self.geom_right.base = arm_T * adjust

    
    # ----------------------------------------------------------
    # Add to Swift environment
    # ----------------------------------------------------------
    def add_to_env(self, env):
        env.add(self.geom_left)
        env.add(self.geom_right)

    # ----------------------------------------------------------
    # Control functions
    # ----------------------------------------------------------
    def open(self,i=50):
        self.geom_left.q = self.traj_open_L[i]
        self.geom_right.q = self.traj_open_R[i]


    def close(self,i=50):
        self.geom_left.q = self.traj_close_L[i]
        self.geom_right.q = self.traj_close_R[i]



# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = UR3_Scaled(scale=1.5)
    r.test()
