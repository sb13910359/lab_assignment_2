import os
from math import pi
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialgeometry import Cylinder
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
from ir_support import CylindricalDHRobotPlot



class Gen3Lite(DHRobot3D):
    def __init__(self, scale=1.5):
        """
        Kinova Gen3 Lite Robot using DH parameters and STL visualization
        scale: 模型放大倍數 (預設 1.5)
        """
        self.scale = scale

        #assign gripper
        self.gripper = Gen3LiteGripper()

        # DH links
        links = self._create_DH()

        # STL link names
        link3D_names = dict(
            link0='base_link',
            link1='shoulder_link',
            link2='arm7',
            link3='fm3',
            link4='lower_wrist',
            link5='upper_wrist',
            link6='base_link'   # ⚠️ 這個會被透明小球覆蓋
        )

        # 測試姿態 (qtest)
        qtest = [0, 0, 0, 0, 0, 0]
        s = self.scale   # 縮放倍數
        qtest_transforms = [
            spb.transl(0, 0, -0.112 * s),      
            spb.transl(0, 0, 0.013 * s),       
            spb.transl(0, -0.02 * s, 0.128 * s),   
            spb.transl(0, -0.0205 * s, 0.408 * s), 
            spb.transl(0.138 * s, 0, 0.408 * s),   
            spb.transl(0.2428 * s, 0, 0.38 * s),   
            spb.transl(0.39 * s, 0, 0.35 * s) @ spb.rpy2tr(0, pi/2, 0, order='xyz')
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_dir = os.path.join(current_path, "meshes", "robot1")
        super().__init__(links, link3D_names, name='Gen3Lite',
                         link3d_dir=mesh_dir,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        # 顏色設定
        colors = [
            (0.45, 0.42, 0.40, 1),   
            (0.55, 0.55, 0.55, 1),   
            (0.45, 0.42, 0.40, 1),   
            (0.45, 0.42, 0.40, 1),   
            (0.45, 0.42, 0.40, 0.8), 
            (0.55, 0.55, 0.55, 0.9), 
            (0.55, 0.55, 0.55, 1)    
        ]
        for i, mesh in enumerate(self.links_3d):
            mesh.color = tuple(float(c) for c in colors[i])
            mesh.scale = [s, s, s]   # ⚡ Mesh 放大

        # dummy end-effector
        dummy_ee = Cylinder(
            radius=0.03 * s,     # 半徑隨比例縮放
            length=0.02 * s,     # 高度隨比例縮放
            color=(0.45, 0.30, 0.20, 0.5),
        )
        self.links_3d[-1] = dummy_ee

        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create DH model for Gen3 Lite (from URDF approx)
        """
        s = self.scale
        a = [0, 0.28 * s, 0, 0, 0, 0]
        d = [0.128 * s, 0, 0, 0.2428 * s, -0.055 * s, 0.15 * s]
        alpha = [pi/2, 0, pi/2, pi/2, -pi/2, 0]
        offset = [0, pi/2, 0, pi/2, 0, 0]

        qlim = [
            [-2.7, 2.7],
            [-2.7, 2.7],
            [-2.7, 2.7],
            [-2.6, 2.6],
            [-2.6, 2.6],
            [-2.6, 2.6]
        ]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(
                d=float(d[i]), a=float(a[i]),
                alpha=float(alpha[i]),
                offset=float(offset[i]),
                qlim=[float(qlim[i][0]), float(qlim[i][1])]
            )
            links.append(link)
        return links


class Gen3LiteGripper:
    def __init__(self):
        # DH Links for both jaws
        l1_1 = rtb.DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi])
        l1_2 = rtb.DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi])
        self.left = rtb.DHRobot([l1_1, l1_2], name="gripper1")

        r1_1 = rtb.DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi])
        r1_2 = rtb.DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi])
        self.right = rtb.DHRobot([r1_1, r1_2], name="gripper2")

        # Visual geometries
        self.g1 = CylindricalDHRobotPlot(self.left, cylinder_radius=0.01, color="#7D7060")
        self.g2 = CylindricalDHRobotPlot(self.right, cylinder_radius=0.01, color="#5C5247")
        self.geom_left = self.g1.create_cylinders()
        self.geom_right = self.g2.create_cylinders()

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

    # ----------------------------------------------------------
    # Attach gripper base to arm end
    # ----------------------------------------------------------
    def attach_to_robot(self, robot: DHRobot3D):
        """Attach gripper to Gen3Lite’s end-effector."""
        arm_T = robot.fkine(robot.q) * SE3(0.03, 0, 0)
        adjust = SE3.Ry(-pi/2) * SE3(0, 0, 0.03) * SE3.Rx(-pi/2)
        self.geom_left.base = arm_T * adjust
        self.geom_right.base = arm_T * adjust

    # ----------------------------------------------------------
    # Add to Swift environment
    # ----------------------------------------------------------
    def add_to_env(self, env):
        env.add(self.geom_left)
        env.add(self.geom_right)

    # ----------------------------------------------------------
    # Control functions (add into loop in main)
    # ----------------------------------------------------------
    def open(self,i):
        self.geom_left.q = self.traj_open_L[i]
        self.geom_right.q = self.traj_open_R[i]


    def close(self,i):
        self.geom_left.q = self.traj_close_L[i]
        self.geom_right.q = self.traj_close_R[i]
