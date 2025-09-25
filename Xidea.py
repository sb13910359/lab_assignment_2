# -*- coding: utf-8 -*-
# ✅ 重點：
# 1) E-STOP/Resume：用 e_stop 控制，停車但不重置狀態
# 2) 防重抓：if not holding: 才執行接近＋關夾；holding=True 代表已抓到、Resume 後直接接後續
# 3) 球跟隨：holding=True 時，凡是 robot.q 在迴圈內移動，都同步更新 target_ball.T

from itertools import combinations  # 你要求整合進來；目前未使用，但保留
import time
import numpy as np
from math import pi

import swift
import roboticstoolbox as rtb
from roboticstoolbox import DHLink, DHRobot, trapezoidal
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Sphere
import spatialgeometry as geometry

from ir_support import UR5, CylindricalDHRobotPlot, line_plane_intersection


# --------------------------------------------------
# 工具：牆壁檢查
# --------------------------------------------------
WALLS = [
    {"x": [2, 2.05], "y": [-1e6, 1e6]},      # 例：x≈2 有牆
    {"x": [-1e6, 1e6], "y": [-0.05, 0.05]},  # 例：y≈0 有牆
]

def base_step_with_walls(base_geom, step_size=0.05, walls=None):
    if walls is None:
        walls = WALLS
    T_now = base_geom.T
    p_next = (T_now * SE3(step_size, 0, 0))[0:3, 3]
    for wall in walls:
        if wall["x"][0] <= p_next[0] <= wall["x"][1] and wall["y"][0] <= p_next[1] <= wall["y"][1]:
            angle = np.random.choice([np.pi/4, -np.pi/4, np.pi/2, -np.pi/2])
            base_geom.T = T_now * SE3.Rz(angle)
            return False
    base_geom.T = T_now * SE3(step_size, 0, 0)
    return True

def move_base_towards(base_geom, target_xy, step_size=0.05,
                      yaw_step=np.deg2rad(15), walls=WALLS, max_iters=800):
    def _yaw_of(T: SE3):
        R = T[:3, :3]
        return np.arctan2(R[1, 0], R[0, 0])
    it = 0
    while it < max_iters:
        it += 1
        p = base_geom.T[0:3, 3]
        dx, dy = target_xy[0] - p[0], target_xy[1] - p[1]
        if np.hypot(dx, dy) < step_size:
            break
        desired_yaw = np.arctan2(dy, dx)
        cur_yaw = _yaw_of(base_geom.T)
        yaw_err = (desired_yaw - cur_yaw + np.pi) % (2*np.pi) - np.pi
        dpsi = np.clip(yaw_err, -yaw_step, yaw_step)
        base_geom.T = base_geom.T * SE3.Rz(dpsi)
        moved = base_step_with_walls(base_geom, step_size, walls)
        if not moved:
            base_geom.T = base_geom.T * SE3.Rz(np.sign(yaw_err) * yaw_step)

        # 若正抓著球，行走時目標球也跟著末端走
        if holding and target_ball is not None:
            target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)

        gripper_stick_arm()
        robot_stick_base()
        env.step(0.03)
        time.sleep(0.03)


# --------------------------------------------------
# 碰撞檢查（線段-平面）
# --------------------------------------------------
def check_collision(robot, q):
    tr = robot.fkine_all(q).A
    planes = {"z=0": {"n": [0, 0, 1], "P": [0, 0, 0],
                      "location_x": [0, 5], "location_y": [0, 5]}}
    for i in range(6):
        p0 = tr[i][:3, 3]
        p1 = tr[i+1][:3, 3]
        for plane in planes.values():
            n, P = plane["n"], plane["P"]
            intersect_p, check = line_plane_intersection(n, P, p0, p1)
            if check == 1:
                xmin, xmax = plane["location_x"]
                ymin, ymax = plane["location_y"]
                if xmin <= intersect_p[0] <= xmax and ymin <= intersect_p[1] <= ymax:
                    return True
    return False


# --------------------------------------------------
# RRT（直線撞就插入隨機點）
# --------------------------------------------------
def safe_rrt_path(q1, q2, max_iters=300):
    robot.q = q1
    env.step()
    time.sleep(0.01)

    q_waypoints = np.array([q1, q2])
    checked_till_waypoint = 0
    q_matrix = []

    iters = 0
    while iters < max_iters:
        iters += 1
        start_waypoint = checked_till_waypoint
        progressed = False

        for i in range(start_waypoint, len(q_waypoints)-1):
            q_traj = rtb.jtraj(q_waypoints[i], q_waypoints[i+1], 50).q
            is_collision_check = any(check_collision(robot, q) for q in q_traj)

            if not is_collision_check:
                q_matrix.extend(q_traj.tolist())
                checked_till_waypoint = i+1
                progressed = True

                q_traj2 = rtb.jtraj(q_matrix[-1], q2, 50).q
                if not any(check_collision(robot, q) for q in q_traj2):
                    q_matrix.extend(q_traj2.tolist())
                    return np.array(q_matrix)
            else:
                q_rand = (2 * np.random.rand(robot.n) - 1) * pi
                while check_collision(robot, q_rand):
                    q_rand = (2 * np.random.rand(robot.n) - 1) * pi
                q_waypoints = np.concatenate(
                    (q_waypoints[:i+1], [q_rand], q_waypoints[i+1:]),
                    axis=0
                )
                progressed = True
                break

        if not progressed:
            return rtb.jtraj(q1, q2, 50).q

    return rtb.jtraj(q1, q2, 50).q


# --------------------------------------------------
# 機器人 / 夾爪
# --------------------------------------------------
def robot_stick_base():
    robot.base = base_geom.T * SE3(0, 0, 0.05)

def gripper_stick_arm():
    arm_T = robot.fkine(robot.q) * SE3(0.03, 0, 0)
    adjust = SE3.Ry(-pi/2) * SE3(0, 0, 0.03) * SE3.Rx(-pi/2)
    gripper_1.base = arm_T * adjust
    gripper_2.base = arm_T * adjust

def RMRC_lift():
    steps = 60
    delta_t = 0.02
    lift_h = 0.50

    T0 = robot.fkine(robot.q).A
    z0 = T0[2, 3]
    z1 = z0 + lift_h
    s = trapezoidal(0, 1, steps).q
    z = (1 - s) * z0 + s * z1

    q_matrix = np.zeros((steps, robot.n))
    q_matrix[0, :] = robot.q.copy()

    for i in range(steps - 1):
        zdot = (z[i + 1] - z[i]) / delta_t
        xdot = np.array([0.0, 0.0, zdot])
        J = robot.jacob0(q_matrix[i, :])
        Jv = J[:3, :]
        qdot = np.linalg.pinv(Jv) @ xdot
        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot

    for q in q_matrix:
        robot.q = q
        # ★ 只要 holding，就讓球跟著末端
        if holding and target_ball is not None:
            target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)
        gripper_stick_arm()
        env.step(delta_t)
        time.sleep(delta_t)

def go_to_home():
    move_base_towards(base_geom, target_xy=(3.2, 2), step_size=0.05, walls=WALLS)
    if holding and target_ball is not None:
        target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)
    gripper_stick_arm()
    env.step(0.03)
    time.sleep(0.03)


# --------------------------------------------------
# 初始化環境
# --------------------------------------------------
env = swift.Swift()
env.launch(realtime=True)
env.set_camera_pose([3, 3, 2], [0, 0, 0])

# --------- 初始化背景 ----------
wall1 = geometry.Mesh('3d-model.stl', pose=SE3(5.7,-0.1, 0)* SE3.Rz(-pi),color = [0.80, 0.78, 0.70, 1],scale=[0.255,0.05,0.052])
wall2 = geometry.Mesh('3d-model.stl', pose=SE3(0.8, 4.5, 0)* SE3.Rz(pi/2),color = [0.7, 0.83, 0.75, 1],scale=[0.24,0.05,0.052])
wall3 = geometry.Mesh('3d-model.stl', pose=SE3(5.8, 9.3, 0),color = [0.7, 0.83, 0.75, 1],scale=[0.255,0.05,0.052])
table = geometry.Mesh('table.stl', pose=SE3(2, 2, 0),color=[0.25, 0.25, 0.25, 1], scale=[1, 1 ,1])
table2 = geometry.Mesh('neutra_table.stl', pose=SE3(5.5, -8, 0),color=[0.25, 0.25, 0.25, 1], scale=[0.007, 0.015, 0.0078])
floor = Cuboid(scale=[10.3, 9.65, 0.01], color=[0.78, 0.86, 0.73, 1])  # 只用 scale 定義大小
floor.T = SE3(5.8, 4.6, 0)  # 位置
Workingzone = Cuboid(scale=[3, 9.5, 0.02], color=[0.78, 0.086, 0.073, 0.8])  # 只用 scale 定義大小
Workingzone.T = SE3(2.2, 4.5, 0)
env.add(floor)
env.add(Workingzone)

env.add(table)
env.add(table2)
env.add(wall1)
env.add(wall2)
env.add(wall3)

# 隨機小球
balls = []
for _ in range(30):
    x = np.random.uniform(4, 9)
    y = np.random.uniform(0, 9)
    z = 0.05
    s = Sphere(radius=0.05, color=[0.0, 0.5, 1.0, 0.9])
    s.T = SE3(x, y, z)
    env.add(s)
    balls.append(s)

# 夾爪（可視化用 DH 兩節）
l1 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
l2 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
gripper1 = DHRobot([l1, l2], name="gripper1")
l11 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
l22 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi])
gripper2 = DHRobot([l11, l22], name="gripper2")

g1 = CylindricalDHRobotPlot(gripper1, cylinder_radius=0.01, color="#36454F")
gripper_1 = g1.create_cylinders()
g2 = CylindricalDHRobotPlot(gripper2, cylinder_radius=0.01, color="#36454F")
gripper_2 = g2.create_cylinders()
env.add(gripper_1)
env.add(gripper_2)

# 夾爪姿態（開/關＋軌跡）
q1_open = [-pi / 2.5,  pi / 3.5]
q2_open = [ pi / 2.5, -pi / 3.5]
q1_close = [-pi / 4,   pi / 5]
q2_close = [ pi / 4,  -pi / 5]
gripper_1.q = q1_open
gripper_2.q = q2_open
traj3 = rtb.jtraj(q1_open, q1_close, 50).q
traj4 = rtb.jtraj(q2_open, q2_close, 50).q

# 機器人與基座
base_geom = Cylinder(radius=0.25, length=0.2, color=[0.3, 0.3, 0.3, 1])
base_geom.T = SE3(5, 5, 0.05)
env.add(base_geom)

area = Cuboid(scale=[0.6, 0.6, 0.01], color=[1, 0.6, 0, 1])
area.T = SE3(2.5, 2, 0.05)
env.add(area)

robot = UR5()
robot.add_to_env(env)

# 狀態
detect_range = 0.8
patrol = True
pick_and_place = False
target_pos_world = None
target_ball = None
holding = False

mode = "patrol"   # "manual" / "patrol"
e_stop = False    # 🛑 E-STOP


# --------------------------------------------------
# GUI 控制：Joint / Cartesian jog
# --------------------------------------------------
def slider_callback(value_deg, joint_index):
    if mode != "manual" or e_stop:
        return
    q = robot.q.copy()
    joint_index = int(np.clip(joint_index, 0, robot.n-1))
    q[joint_index] = np.deg2rad(float(value_deg))
    robot.q = q
    if holding and target_ball is not None:
        target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)
    gripper_stick_arm()
    robot_stick_base()
    env.step(0.02)

def cartesian_callback(delta, axis):
    if mode != "manual" or e_stop:
        return
    T = robot.fkine(robot.q)
    if axis == "x":
        T_new = T * SE3(delta, 0, 0)
    elif axis == "y":
        T_new = T * SE3(0, delta, 0)
    elif axis == "z":
        T_new = T * SE3(0, 0, delta)
    else:
        return
    q_new = robot.ikine_LM(T_new, q0=robot.q).q
    robot.q = q_new
    if holding and target_ball is not None:
        target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)
    gripper_stick_arm()
    robot_stick_base()
    env.step(0.02)

# Buttons / Sliders
try:
    manual_btn = swift.Button(desc="Manual Mode",
                              cb=lambda _=None: (globals().__setitem__('mode', 'manual'),
                                                 globals().__setitem__('patrol', False),
                                                 globals().__setitem__('pick_and_place', False)))
    patrol_btn = swift.Button(desc="Patrol Mode",
                              cb=lambda _=None: (globals().__setitem__('mode', 'patrol'),
                                                 globals().__setitem__('patrol', True)))
    estop_btn = swift.Button(desc="🛑 E-STOP",
                             cb=lambda _=None: globals().__setitem__('e_stop', True))
    resume_btn = swift.Button(desc="▶ Resume",
                              cb=lambda _=None: globals().__setitem__('e_stop', False))
    env.add(manual_btn); env.add(patrol_btn)
    env.add(estop_btn); env.add(resume_btn)
except Exception:
    pass

# Joint sliders
for i in range(robot.n):
    s = swift.Slider(cb=lambda v, j=i: slider_callback(v, j),
                     min=-180, max=180, step=1,
                     value=np.rad2deg(robot.q[i]),
                     desc=f'Joint {i+1}', unit='°')
    env.add(s)

# Cartesian sliders（單位：cm → 轉成 m）
env.add(swift.Slider(cb=lambda v: cartesian_callback(v*0.01, "x"),
                     min=-10, max=10, step=1, value=0, desc="ΔX", unit="cm"))
env.add(swift.Slider(cb=lambda v: cartesian_callback(v*0.01, "y"),
                     min=-10, max=10, step=1, value=0, desc="ΔY", unit="cm"))
env.add(swift.Slider(cb=lambda v: cartesian_callback(v*0.01, "z"),
                     min=-10, max=10, step=1, value=0, desc="ΔZ", unit="cm"))


# --------------------------------------------------
# 主迴圈
# --------------------------------------------------
while True:
    # E-STOP：維持畫面但不做動作
    if e_stop:
        env.step(0.05)
        time.sleep(0.05)
        continue

    # Manual：只由 GUI 操控
    if mode == "manual":
        gripper_stick_arm()
        robot_stick_base()
        env.step(0.03)
        time.sleep(0.03)
        continue

    # Patrol
    if patrol:
        for segment in range(5):
            if mode != "patrol" or e_stop:
                break
            gripper_stick_arm()
            distance = np.random.uniform(1.0, 2.0)
            step_size = 0.05
            steps = int(distance / step_size)

            for _ in range(steps):
                if mode != "patrol" or e_stop:
                    break
                gripper_stick_arm()
                base_step_with_walls(base_geom, step_size, WALLS)
                robot_stick_base()
                env.step(0.05)
                time.sleep(0.05)

                # 偵測球（只在 Patrol 時進行）
                for ball in list(balls):
                    ball_pos_world = ball.T[:3, 3]
                    base_pos = base_geom.T[:3, 3]
                    dist = np.linalg.norm(ball_pos_world[:2] - base_pos[:2])
                    if dist < detect_range:
                        patrol = False
                        pick_and_place = True
                        target_pos_world = ball_pos_world
                        target_ball = ball
                        print(f"🎯 偵測到球：{target_pos_world}")
                        break
                if not patrol:
                    break
            if not patrol:
                break

            # 隨機轉向
            total_angle = np.random.uniform(-np.pi, np.pi)
            angle_step = total_angle / 20
            for _ in range(20):
                if mode != "patrol" or e_stop:
                    break
                gripper_stick_arm()
                base_geom.T = base_geom.T * SE3.Rz(angle_step)
                robot_stick_base()
                env.step(0.05)
                time.sleep(0.05)

    # Pick & Place：Resume 後若已 holding，就跳過「接近＋關夾」
    elif pick_and_place and target_pos_world is not None and mode == "patrol":
        # 1) 若尚未抓到，才執行 接近 + 關夾（避免 Resume 後重抓）
        if not holding:
            target = SE3(target_pos_world[0], target_pos_world[1], target_pos_world[2] + 0.06) * SE3.Rx(pi)
            q_pick = robot.ikine_LM(target, q0=robot.q).q

            for q in safe_rrt_path(robot.q, q_pick):
                if mode != "patrol" or e_stop:
                    break
                robot.q = q
                # 尚未 holding，不更新球
                gripper_stick_arm()
                env.step(0.02)

            if e_stop:
                continue

            # 關夾（抓球）
            for i in range(50):
                if mode != "patrol" or e_stop:
                    break
                gripper_1.q = traj3[i]
                gripper_2.q = traj4[i]
                gripper_stick_arm()
                env.step(0.02)

            if e_stop:
                continue

            holding = True  # ★ 關夾完成，正式抓到
            RMRC_lift()
            if e_stop:
                continue

        # 2) 無論是剛抓到或 Resume 後 holding=True，往 Home 移動
        go_to_home()
        if e_stop:
            continue

        # 3) 下降到放置點
        q_down = robot.ikine_LM(area.T * SE3.Rx(pi) * SE3(0, 0, -0.1), q0=robot.q).q
        for q in safe_rrt_path(robot.q, q_down):
            if mode != "patrol" or e_stop:
                break
            robot.q = q
            if holding and target_ball is not None:
                target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)  # ★ 路上球跟著走
            gripper_stick_arm()
            env.step(0.02)

        # 4) 放球 + 從列表移除
        if target_ball is not None:
            target_ball.T = area.T * SE3(0, 0, 0.06)
            if target_ball in balls:
                balls.remove(target_ball)

        holding = False
        RMRC_lift()
        patrol = True
        pick_and_place = False

    else:
        env.step(0.03)
        time.sleep(0.03)

env.hold()
