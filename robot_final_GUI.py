# gui_gen3lite_test.py — Minimal GUI-only test (with transparent brown sphere as EE)

import time
import numpy as np
from math import pi

import swift
import roboticstoolbox as rtb
from spatialmath import SE3
import spatialmath.base as spb
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialgeometry import Sphere,Cuboid ,Cylinder  # ✅ 加這個
import os


class Gen3Lite(DHRobot3D):
    def __init__(self):
    
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
            link6='base_link'   # 這個會被透明小球覆蓋
        )

        # 測試姿態 (qtest)
        qtest = [0, 0, 0, 0, 0, 0]
        qtest_transforms = [
            spb.transl(0, 0, -0.112),      # base
            spb.transl(0, 0, 0.013),       # shoulder
            spb.transl(0, -0.02, 0.128),   # arm
            spb.transl(0, -0.0205, 0.408), # forearm
            spb.transl(0.138, 0, 0.408),   # lower wrist
            spb.transl(0.2428, 0, 0.38),   # upper wrist
            spb.transl(0.39, 0, 0.35)@ spb.rpy2tr(0,pi/2,0, order='xyz')       # end-effector
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        super().__init__(links, link3D_names, name='Gen3Lite',
                         link3d_dir=current_path,
                         qtest=qtest, qtest_transforms=qtest_transforms)

        # ✅ 設定顏色 (RGBA)
        colors = [
            (0.45, 0.42, 0.40, 1),   # base
            (0.55, 0.55, 0.55, 1),   # shoulder
            (0.45, 0.42, 0.40, 1),   # arm
            (0.45, 0.42, 0.40, 1),   # forearm
            (0.45, 0.42, 0.40, 0.8), # lower wrist
            (0.55, 0.55, 0.55, 0.9), # upper wrist
            (0.55, 0.55, 0.55, 1)    # dummy end effector
        ]
        for i, mesh in enumerate(self.links_3d):
            mesh.color = tuple(float(c) for c in colors[i])

        # ✅ 把末端換成咖啡色透明小球
        #dummy_ee = Sphere(radius=0.02, color=(0.45, 0.30, 0.20, 0.5))
        #self.links_3d[-1] = dummy_ee
        dummy_ee = Cylinder(
        radius=0.03,              # 半徑 2cm
        length=0.02,              # 高度 5cm
        color=(0.45, 0.30, 0.20, 0.5),  # 咖啡色 + 透明度 0.5
        
)
        self.links_3d[-1] = dummy_ee

        self.q = qtest

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        """
        Create DH model for Gen3 Lite (from URDF approx)
        """
        a = [0, 0.28, 0, 0, 0, 0]
        d = [0.128, 0, 0, 0.2428, -0.055, 0]   # ✅ 最後一節沿 z 偏移 0.0285
        alpha = [pi/2, 0, pi/2, pi/2, -pi/2, 0]  # ✅ end-effector α = 0
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
            link = rtb.RevoluteDH(d=float(d[i]), a=float(a[i]),
                                  alpha=float(alpha[i]),
                                  offset=float(offset[i]),
                                  qlim=[float(qlim[i][0]), float(qlim[i][1])])
            links.append(link)
        return links


# ------------------------------ Swift env ------------------------------
env = swift.Swift()
env.launch(realtime=True)
env.set_camera_pose([3, 3, 2], [0, 0, 0])

robot = Gen3Lite()
robot.add_to_env(env)

robot.base = SE3(0, 0, 0.05)

# ------------------------------ GUI ------------------------------
e_stop = False

def slider_callback(value_deg, joint_index):
    global e_stop
    if e_stop:
        return
    q = robot.q.copy()
    q[joint_index] = float(np.deg2rad(float(value_deg)))
    robot.q = q
    env.step(0.02)

for i in range(robot.n):
    env.add(
        swift.Slider(
            cb=lambda v, j=i: slider_callback(v, j),
            min=-180, max=180, step=1,
            value=float(np.rad2deg(float(robot.q[i]))),
            desc=f'Joint {i+1}', unit='°'
        )
    )

# ------------------------------ Main loop ------------------------------
while True:
    env.step(0.03)
    time.sleep(0.03)
