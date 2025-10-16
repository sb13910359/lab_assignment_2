import time
import random
import threading
import numpy as np
from math import pi
import os
import roboticstoolbox as rtb
from roboticstoolbox import DHLink, DHRobot, trapezoidal
from spatialmath import SE3, SO3
from spatialmath.base import rpy2r, tr2rpy
from spatialgeometry import Cuboid, Cylinder, Sphere
import spatialgeometry as geometry
from ir_support import CylindricalDHRobotPlot, line_plane_intersection
import swift
from Gen3Lite_mesh import Gen3Lite
from irb1200_mesh_v2 import IRB1200
from ur3_scaled import UR3_Scaled
from environment_builder import EnvironmentBuilder
from gui_controller import RobotGUI


#robot1 = gen3lite
#robot2 = ur3
#robot3 = irb1200

#####


#連桿碰撞檢測   Connecting rod collision detection . common for all robots
def check_collision(q, robot):
    tr = robot.fkine_all(q).A
    planes = {"floor": {"normal": [0, 0, 1], "point": [0, 0, 0],"location_x": [0, 10], "location_y": [0, 10]},
              "wall1": {"normal": [0, 1, 0], "point": [0, 0,0],"location_x": [0, 10], "location_y": [0, 10]},
              "wall2": {"normal": [0, 1, 0], "point": [9, 9, 0],"location_x": [0, 10], "location_y": [0,10]},        
            }
    for i in range(6):
        p0 = tr[i][:3, 3]
        p1 = tr[i+1][:3, 3]
        for plane in planes.values():
            n, P = plane["normal"], plane["point"]
            intersect, check = line_plane_intersection(n, P, p0, p1)
            if check == 1:
                xmin, xmax = plane["location_x"]
                ymin, ymax = plane["location_y"]
                if xmin <= intersect[0] <= xmax and ymin <= intersect[1] <= ymax:
                    return True
    return False


#---------------------------------------------------------------


#robot1 1 / base_geom
#讓底座嘗試前進一步並檢查有沒有撞牆  Let the base try to move forward and check if it hits a wall
def base_step_with_walls(base_geom, step_size=0.05):
    planes = {
            "wall1": {"normal": [0, 1, 0], "point": [0.1, 0, 0],"location_x": [0, 10], "location_y": [0, 10]},
            "wall2": {"normal": [0, 1, 0], "point": [8.5, 8.5, 0],"location_x": [0, 10], "location_y": [0, 10]},
            "wall3": {"normal": [1, 0, 0], "point": [4, 0, 0],"location_x": [0, 10], "location_y": [0, 10]}
        }

    T_now = base_geom.T
    p0 = T_now[0:3, 3]                        # 當前位置
    p1 = (T_now * SE3(step_size, 0, 0))[0:3, 3]  # 嘗試往前走一步後的位置

    for plane in planes.values():
        n, P = plane["normal"], plane["point"]       # 平面的法向量和通過點
        intersect, check = line_plane_intersection(n, P, p0, p1)

        if check == 1:  # 有交點
            xmin, xmax = plane["location_x"]
            ymin, ymax = plane["location_y"]

            # 檢查交點是否在平面定義的矩形區域內
            if xmin <= intersect[0] <= xmax and ymin <= intersect[1] <= ymax:
                # 如果有撞到牆 → 隨機選轉角避免撞牆
                angle = np.random.choice([np.pi, -np.pi, np.pi/2, -np.pi/2])
                turn = angle / 20  # 每次要轉的小角度
                print("撞到牆 Hitting the wall")
                for _ in range(20):
                   gripper_stick_arm() 
                   base_geom.T = base_geom.T * SE3.Rz(turn)
                   robot1_stick_base() 
                   env.step(0.02)  # 更新環境 (動畫更順)
                   time.sleep(0.02)  # 控制轉動速度
                   
                print("正在轉 turning")
                return False
  

    # 如果所有平面都沒撞到 → 真的走一步
    base_geom.T = T_now * SE3(step_size, 0, 0)
    return True

#robot1 (base_geom)
#往基座走去  Go to the base
def move_base_towards(base_geom, target_xy, step_size=0.05, max_iters=800):
    def _yaw_of(T):
        R = T[:3, :3]
        return np.arctan2(R[1, 0], R[0, 0])
        #機器人當前在 XY 平面的朝向 (yaw)
    it = 0
    while it < max_iters:
        if e_stop or mode == "manual": 
            return
        it += 1
        p = base_geom.T[0:3, 3]
        dx, dy = target_xy[0] - p[0], target_xy[1] - p[1]
        #計算當前位置到目標的距離，如果比一步還短，就當作已經到達，停止迴圈。
        if np.hypot(dx, dy) < step_size:
            break
        #(dx, dy) 計算出「理想的朝向角度」
        desired_yaw = np.arctan2(dy, dx)
        #底座目前的朝向角
        cur_yaw = _yaw_of(base_geom.T)
        #這段就是再算從cur_yaw轉到 desired_yaw最近需要轉的度數
        yaw_err = (desired_yaw - cur_yaw + np.pi) % (2*np.pi) - np.pi
        #要轉的角度差 yaw_err 限制在 ±yaw_step 之內，確保機器人每次只會小幅度轉向
        turn = np.clip(yaw_err, -np.deg2rad(15), np.deg2rad(15))
        base_geom.T = base_geom.T * SE3.Rz(turn)
        #不是只有一次turn 因為是在while loop所以是轉一點走一步轉一點
        moved = base_step_with_walls(base_geom, step_size)
        #嘗試往前走一步，如果成功走了，moved=True；如果被牆擋住，moved=False

        #「如果前面有牆擋住走不動，那就往目標方向的那一邊小轉 15° 再試。
        if not moved:
            base_geom.T = base_geom.T * SE3.Rz(np.sign(yaw_err) * np.deg2rad(15))

        if holding ==True:
            target_ball.T = robot1.fkine(robot1.q) * trash_offset

        gripper_stick_arm()
        robot1_stick_base()
        env.step(0.03)
        time.sleep(0.03)

#robot1
def safe_rrt_path(q1, q2, max_iters=300):
    robot1.q = q1
    env.step()
    time.sleep(0.01)

    q_waypoints = np.array([q1, q2])#目前已知的路徑點，最初只包含 [起點, 終點]
    checked_till_waypoint = 0 #紀錄已經檢查到哪個 waypoint
    q_matrix = [] #完整路徑（會存放所有插值後的關節軌跡）

    iters = 0
    while iters < max_iters:
        if e_stop: 
            return np.array(q_matrix) #回傳機器人已經走過路徑
        iters += 1
        start_waypoint = checked_till_waypoint
        progressed = False

        for i in range(start_waypoint, len(q_waypoints)-1):
            if e_stop:   
                return np.array(q_matrix)

            q_traj = rtb.jtraj(q_waypoints[i], q_waypoints[i+1], 50).q
            is_collision_check = any(check_collision(q, robot1) for q in q_traj)
            #沒碰撞
            if not is_collision_check:
                q_matrix.extend(q_traj.tolist())
                #把這段安全的插值軌跡加到完整路徑 q_matrix 裡。
                checked_till_waypoint = i+1
                # 表示：我已經確認「第 i → 第 i+1」這段路徑是安全的。
                #下一輪從i+1點開始檢查
                progressed = True
                #試看看中繼點到終點 
                q_traj2 = rtb.jtraj(q_matrix[-1], q2, 50).q
                #又沒碰撞
                if not any(check_collision(q, robot1) for q in q_traj2):
                    #把剩下路徑加到qmatrix
                    q_matrix.extend(q_traj2.tolist())
                    return np.array(q_matrix)
            else:
                #有撞到
                #隨機加一組q (-pi到pi)
                q_rand = (2 * np.random.rand(robot1.n) - 1) * pi
                #check會不會撞
                while check_collision(q_rand, robot1):
                    if e_stop:  
                        return np.array(q_matrix)
                    #會撞再重新生成一組新的
                    q_rand = (2 * np.random.rand(robot1.n) - 1) * pi
                #不會撞就把這個安全的隨機點 插入到目前的 waypoint 路徑裡
                q_waypoints = np.concatenate(
                    (q_waypoints[:i+1], [q_rand], q_waypoints[i+1:]),
                   # q_waypoints[:i+1] 起點到第i個
                    axis=0
                    #axis=0 → 沿著「列 (row)」的方向操作
                )
                progressed = True
                break

        if not progressed:
        #避免進入死循環(沒新增路徑、沒新增隨機點)
            print(f"死循環") 
            return rtb.jtraj(q1, q2, 50).q
        

    return rtb.jtraj(q1, q2, 50).q  
    #如果嘗試了 max_iters 次還沒找到路徑 → 直接回傳直線插值（最後手段）

#robot1
def robot1_stick_base():
    robot1.base = base_geom.T * SE3(0, 0, 0.12)

#robot1
def gripper_stick_arm():
    arm_T = robot1.fkine(robot1.q) * SE3(0.03, 0, 0)
    adjust = SE3.Ry(-pi/2) * SE3(0, 0, 0.03) * SE3.Rx(-pi/2)
    gripper_1.base = arm_T * adjust
    gripper_2.base = arm_T * adjust

#robot1
def RMRC_lift():
    steps = 60
    delta_t = 0.02
    lift_h = 0.50#抬升的總高度 = 0.5 公尺

    T0 = robot1.fkine(robot1.q).A
    z0 = T0[2, 3]
    z1 = z0 + lift_h

    #產生 z0-z1的平滑中間點
    s = trapezoidal(0, 1, steps).q
    z = (1 - s) * z0 + s * z1
    #建立一個矩陣來存放 每一步的關節角度
    q_matrix = np.zeros((steps, robot1.n))
    #把目前的機械臂關節角度存到 q_matrix 的第 0 行
    q_matrix[0, :] = robot1.q.copy()

    for i in range(steps - 1):
        if e_stop:  
            return
        #Z速
        zdot = (z[i + 1] - z[i]) / delta_t
        #x速
        xdot = np.array([0.0, 0.0, zdot])
        #當前關節角度下的 Jacobian 矩陣
        J = robot1.jacob0(q_matrix[i, :])
        Jv = J[:3, :]
        #計算所需關節速度
        qdot = np.linalg.pinv(Jv) @ xdot
        #下一個關節= 這個關節加上q速(q變化量)
        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot
    #走過q
    for q in q_matrix:
        if e_stop:   
            return
        robot1.q = q
        if holding == True:
            target_ball.T = robot1.fkine(robot1.q) * trash_offset
        gripper_stick_arm()
        env.step( 0.02)
        time.sleep( 0.02)

#robot1
def go_to_home():
   
        move_base_towards(base_geom, target_xy=(4, 5.7), step_size=0.05)
        target_ball.T = robot1.fkine(robot1.q) * trash_offset
        gripper_stick_arm()
        env.step(0.03)
        time.sleep(0.03)


#robot2
def rmrc_move_ur3(robot, env, T_start, T_goal,
              steps=80, delta_t=0.015, epsilon=0.05, lambda_max=0.1, draw_path=False,
              follow_object=False, obj=None, obj_offset=None, z_arc=False, ee_down=True):

    # Helper: Damped Least Squares inverse
    def damped_ls(J, lam):
        return np.linalg.inv(J.T @ J + lam**2 * np.eye(J.shape[1])) @ J.T

    # Create trajectory in Cartesian space (interpolated positions)
    s_profile = trapezoidal(0, 1, steps)
    s = s_profile.q
    x = np.zeros((3, steps))
    theta = np.zeros((3, steps))

    R_down = SO3.Rx(np.pi)              # -Z (gripper facing down)
    R0     = SO3(T_start.R)             # start orientation as SO3
    R1     = R_down                     # target orientation


    if ee_down:
        R0 = SO3(T_start.R)
        R1 = SO3.Rx(np.pi)
    else:
        R0 = SO3(T_start.R)
        R1  = SO3(T_goal.R)


    for i in range(steps):
        # Linear interpolation between start and goal
        x[0, i] = (1 - s[i]) * T_start.t[0] + s[i] * T_goal.t[0]
        x[1, i] = (1 - s[i]) * T_start.t[1] + s[i] * T_goal.t[1]
        x[2, i] = (1 - s[i]) * T_start.t[2] + s[i] * T_goal.t[2]

        if z_arc == True:
            x[2, i] += 0.15 * np.sin(np.pi * s[i])
    
        # Keep gripper vertical
        #theta[:, i] = [np.pi, 0, 0]
        R_interp   = R0.interp(R1, s[i])         # slerp
        theta[:, i] = tr2rpy(R_interp.R)

    # Initialize storage
    q_matrix = np.zeros((steps, robot.n))
    qdot = np.zeros((steps, robot.n))
    m = np.zeros(steps)
    q_matrix[0, :] = robot.q.copy()
    #qlim = np.array(robot.qlim).T
    ee_path = []

    # RMRC loop
    for i in range(steps - 1):
        # --- Forward kinematics ---
        T = robot.fkine(q_matrix[i, :]).A
        pos, R = T[:3, 3], T[:3, :3]

        # --- Compute desired motion ---
        delta_x = x[:, i + 1] - pos
        Rd = rpy2r(theta[0, i + 1], theta[1, i + 1], theta[2, i + 1])
        Rdot = (Rd - R) / delta_t
        S = Rdot @ R.T

        linear_velocity = delta_x / delta_t
        angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
        xdot = np.hstack((linear_velocity, angular_velocity))
        W = np.diag([1,1,1, 0.5,0.5,0.5])
        xdot = W @ xdot

        # --- Jacobian and manipulability ---
        J = robot.jacob0(q_matrix[i, :])
        m[i] = np.sqrt(np.linalg.det(J @ J.T))
        if m[i] < epsilon:  #Check if we are near a singularity
            ratio = m[i] / epsilon          ## ranges from 0 (at singularity) to 1 (safe)
            lam = (1 - ratio) * lambda_max  # damping value between 0 → lambda_max
        else:
            lam = 0                 # If robot is not near singularity, no damping needed
        invJ = damped_ls(J, lam)

        # --- Solve joint velocities ---
        qdot[i, :] = (invJ @ xdot).T
    

        # --- Joint limit check ---
        #for j in range(6):
        #    if i > 0 and np.sign(qdot[i, j]) != np.sign(qdot[i - 1, j]) and abs(qdot[i, j]) > 0.5:
        #        qdot[i, j] *= 0.5                       #avoid multiple ikine solutions 
        #for j in range(6):                                                # Loop through joints 1 to 6
        #    if q_matrix[i,j] + delta_t * qdot[i,j] < qlim[j,0]:             # If next joint angle is lower than joint limit...
        #        qdot[i,j] = 0 # Stop the motor
        #    elif q_matrix[i,j] + delta_t * qdot[i,j] > qlim[j,1]:           # If next joint angle is greater than joint limit ...
        #        qdot[i,j] = 0 # Stop the motor

        # --- Integrate joint motion ---
        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot[i, :]

        # --- Update robot in Swift ---
        robot.q = q_matrix[i + 1, :]
        gripper_stick_arm2()

        # --- For picking up objects ---
        if follow_object and obj is not None:
            global trash_offset_ur3
            if obj_offset is True:
                obj.T = robot.fkine(robot.q) * trash_offset_ur3
            else:
                obj.T = robot.fkine(robot.q) * SE3(0, 0, 0.06) * SE3.Rx(np.pi)

        
            
            

        ee_path.append(robot.fkine(robot.q).t)
        env.step(delta_t)
        time.sleep(delta_t)

    # Draw trajectory markers
    if draw_path:
        ee_path = np.array(ee_path)
        for p in ee_path[::5]:
            env.add(geometry.Sphere(radius=0.01,
                                    color=[0.8, 0.1, 0.9, 1],
                                    pose=SE3(p[0], p[1], p[2])))

    return q_matrix[-1, :]

#robot2
def ur3_pick_and_place():
    # ---------------------------------------------------------------------
    # 3️⃣ Target poses (using IK)
    # ---------------------------------------------------------------------
    q_rest = np.array([np.pi/4 - 0.15, -np.pi/2 + 0.15, - 0.3, -np.pi/2 - 0.15, np.pi/2, 0])
    q_pick  = robot2.ikine_LM(area.T * SE3.Tz(0.1) * SE3.Rx(np.pi),  q0=np.array([np.pi/4 + -0.15, -3*np.pi/4 + 0.15, -np.pi/2 + -0.15, -np.pi/4, np.pi/2, 0])).q
    q_place = robot2.ikine_LM(area_place.T * SE3.Tz(0.1) * SE3.Rx(np.pi), q0=np.array([-3*np.pi/8, -7*np.pi/8, -np.pi/4, -np.pi/4, np.pi/2, 0])).q
    q_box   = robot2.ikine_LM(area_box.T * SE3.Tz(0.1) * SE3.Rx(np.pi),   q0=np.array([-7*np.pi/8, -2*np.pi + 0.15, -np.pi/4, -np.pi/4, np.pi/2, 0])).q

    T_rest  = robot2.fkine(q_rest)
    T_pick  = robot2.fkine(q_pick) * SE3(0, 0, -0.06)
    T_place = robot2.fkine(q_place)* SE3(0, 0, 0.04)
    T_box   = robot2.fkine(q_box)

    #----------------------------------------------------------------
    global current_trash_index, current_ur3_object, trash_offset_ur3

    while len(area_trash) > 0:

        ur3_ball = area_trash[0]
        current_ur3_object = ur3_ball

        rmrc_move_ur3(robot2, env, T_rest, T_pick)        # traj1

        ee_T = robot2.fkine(robot2.q)
        trash_offset_ur3 = ee_T.inv() * ur3_ball.T

        rmrc_move_ur3(robot2, env, T_pick, T_place, follow_object=True, obj=ur3_ball, obj_offset=True, ee_down=False, z_arc=True)       # traj2

        ur3_ball.T = area_place.T  

        rmrc_move_ur3(robot2, env, T_place, T_rest)       # traj3

        global crusher_trigger
        crusher_trigger = True
        print("🟣 Crusher trigger set! (trash index:", current_trash_index, ")")

        time.sleep(4.0) 

        rmrc_move_ur3(robot2, env, T_rest, T_place, ee_down=False)       # traj4
        rmrc_move_ur3(robot2, env, T_place, T_box, follow_object=True, obj=crushed, ee_down=False, z_arc=True)        # traj5

        crushed.T = area_box.T * SE3.Tz(-0.1)

        rmrc_move_ur3(robot2, env, T_box, T_rest, ee_down=False)         # traj6

        area_trash.remove(ur3_ball)

#robot2
def gripper_stick_arm2():
    arm2 = robot2.fkine(robot2.q) * SE3(0.03, 0, 0)
    adjust = SE3.Ry(-pi/2) * SE3(0, 0, 0.03) * SE3.Rx(-pi/2)
    gripper_3.base = arm2 * adjust
    gripper_4.base = arm2 * adjust


#robot 3
def swap_to_crushed_object():
    global crushed, current_ur3_object, current_trash_index

    if current_ur3_object is None:
        return

    current_ur3_object.T = SE3(current_ur3_object.T) * SE3(0, 0, -2.0)
    crushed = squashed_trash_list[current_trash_index]
    crushed.T = area_place.T * SE3.Tz(0.01)
    env.step()

# robot3
def crusher_rmrc_trajectory():

    steps = 20
    delta_t = 0.02

    R_down = SE3.Rx(np.pi) 
    T_start = SE3(area_place.T) * SE3(0, 0, 0.7) * R_down 
    T_end = SE3(area_place.T) * R_down

    s = trapezoidal(0, 1, steps).q
    pos_traj = np.zeros((3, steps))
    for i in range(steps):
        pos_traj[:, i] = (1 - s[i]) * T_start.t + s[i] * T_end.t

    q_matrix = np.zeros((steps, robot3.n))
    q_matrix[0, :] = robot3.ikine_LM(
        T_start, q0=np.array([0, np.pi / 4, 0, 0, np.pi / 4, 0])
    ).q

    R_des = R_down.R
    lam, k_omega = 1e-3, 2.0

    for i in range(steps - 1):
        xdot = (pos_traj[:, i + 1] - pos_traj[:, i]) / delta_t
        T_now = robot3.fkine(q_matrix[i, :])
        R_now = T_now.R
        R_err = R_des @ R_now.T
        omega_vec = SO3(R_err).log()
        if omega_vec.shape != (3,):
            omega_vec = np.array([omega_vec[2, 1], omega_vec[0, 2], omega_vec[1, 0]])
        omega_vec = k_omega * np.array(omega_vec).reshape(3,)

        v = np.concatenate((xdot, omega_vec), axis=0).reshape(6, 1)
        J = robot3.jacob0(q_matrix[i, :])
        JJt = J @ J.T
        invJJt = np.linalg.inv(JJt + (lam**2) * np.eye(6))
        qdot = (J.T @ invJJt @ v).flatten()

        q_next = q_matrix[i, :] + delta_t * qdot
        q_next = np.clip(q_next, robot3.qlim[0, :], robot3.qlim[1, :])
        q_matrix[i + 1, :] = q_next

    print("🦾 Starting crushing simulation...")


    for q in q_matrix:
        robot3.q = q
        T = robot3.fkine(q)
        robot3_ee.T = T
        if check_collision(q, robot3):
            print("🚨 Zone collision detected")
        env.step()
        time.sleep(delta_t)

    try:
        swap_to_crushed_object()
    except Exception as e:
        print(f"⚠️ Swap error during crushing: {e}")


        # Upward motion
    for q in q_matrix[::-1]:
        robot3.q = q
        T = robot3.fkine(q)
        robot3_ee.T = T
        if check_collision(q, robot3):
            print("🚨 Zone collision detected")
        env.step()
        time.sleep(delta_t)

    print("✅ Crushing sequence complete (swap pending).")


# 初始化環境 Initialize the environment

env_builder = EnvironmentBuilder()
env = env_builder.env

area = env_builder.area
area_place = env_builder.area_place
area_box = env_builder.area_box


current_dir = os.path.dirname(os.path.abspath(__file__))
bottle3_stl_path = os.path.join(current_dir, "bottle3.stl")
bottle2_stl_path = os.path.join(current_dir, "bottle2.stl")
paper2_stl_path = os.path.join(current_dir, "paper2.stl")



# 隨機垃圾 (random trash)
balls = []
area_trash = [] 
trash_amt = 30
squashed_trash_list = []
for _ in range(trash_amt):
    # 隨機挑一種垃圾 STL     Randomly choose a trash STL type
    trash_type = random.choice([
        (bottle3_stl_path, [0.35, 0.35, 0.35], [np.random.uniform(0.0,0.2), np.random.uniform(0.2,0.5), np.random.uniform(0.5,0.9), np.random.uniform(0.4,0.7)]),  # 藍瓶
        (bottle2_stl_path, [0.35, 0.35, 0.25], [np.random.uniform(0.7,1.0), np.random.uniform(0.2,0.5), np.random.uniform(0.0,0.2), np.random.uniform(0.5,1.0)]),  # 橘紅瓶
        (paper2_stl_path,  [0.0018, 0.0018, 0.0018], [0.92, 0.92, 0.92, 1])  # 小紙屑
    ])

    fname, scale, color = trash_type

    # 隨機位置 (放在地板上) Random position (on the floor)
    x = np.random.uniform(4, 9)
    y = np.random.uniform(0, 9)
    z = 0.05

    # 姿態 (倒下去 + 隨機旋轉)  Orientation (lying down + random rotation)
    pose = SE3(x, y, z) * SE3.Rx(pi/2) * SE3.Ry(np.random.uniform(-pi, pi))

    trash = geometry.Mesh(fname, pose=pose, scale=scale, color=color)
    env.add(trash)
    balls.append(trash)

    crushed = geometry.Cylinder(
        radius=0.05 * 1.3,  length=0.05 * 0.25,  # thinner height
        color=[0.5, 0.5, 0.5, 1],   pose=SE3(x, y, -1.0)   # hidden below the visible floor
    )
    env.add(crushed)
    squashed_trash_list.append(crushed)

# 夾爪（可視化用 DH 兩節） Gripper (visualization with DH two sections)

#robot1 gripper
l1_1 = DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi]) 
l1_2 = DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi]) 
gripper1 = DHRobot([l1_1, l1_2 ], name="gripper1") 
r1_1 = DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi]) 
r1_2 = DHLink(d=0, a=0.05, alpha=0, qlim=[-pi, pi]) 
gripper2 = DHRobot([r1_1, r1_2], name="gripper2") 

#robot2 gripper
l2_1 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
l2_2 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
gripper3 = DHRobot([l2_1 , l2_2], name="gripper3") 
r2_1 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
r2_2 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
gripper4 = DHRobot([r2_1, r2_2], name="gripper4") 

g1 = CylindricalDHRobotPlot(gripper1, cylinder_radius=0.01, color="#7D7060") 
gripper_1 = g1.create_cylinders() 
g2 = CylindricalDHRobotPlot(gripper2, cylinder_radius=0.01, color="#5C5247") 
gripper_2 = g2.create_cylinders() 
env.add(gripper_1) ;env.add(gripper_2) 


g3 = CylindricalDHRobotPlot(gripper3, cylinder_radius=0.01, color="#1E324D") 
gripper_3 = g3.create_cylinders() 
g4 = CylindricalDHRobotPlot(gripper4, cylinder_radius=0.01, color="#1E324D") 
gripper_4 = g4.create_cylinders() 
env.add(gripper_3) ;env.add(gripper_4) 

# 夾爪開關      Gripper switch
q1_open = [-pi / 2.5, pi / 3.5] 
q2_open = [ pi / 2.5, -pi / 3.5] 
q1_close = [-pi / 3, pi / 4] 
q2_close = [ pi / 3, -pi / 4] 
gripper_1.q = q1_open 
gripper_2.q = q2_open 
gripper_3.q = q1_open 
gripper_4.q = q2_open 
traj1 = rtb.jtraj(q1_close, q1_open, 50).q 
traj2 = rtb.jtraj( q2_close,q2_open, 50).q 
traj3 = rtb.jtraj(q1_open, q1_close, 50).q 
traj4 = rtb.jtraj(q2_open, q2_close, 50).q 

# 機器人與基座      Robot and base
base_geom=Sphere(radius=0.2, color= (0.45, 0.42, 0.40, 1))
base_geom.T = SE3(6, 5, -0.01) 
env.add(base_geom)

#robot init

robot1 = Gen3Lite()
robot2 = UR3_Scaled()
robot3 = IRB1200()
robot3.base = robot3.base * area_place.T * SE3.Ty(-0.3) * SE3.Rz(pi/2) 
robot3_base = Cylinder(radius=0.25, length=0.6,
                       color=(0.20, 0.12, 0.06),
                       pose=robot3.base * SE3(0, 0, -0.31)) 
robot2_base = Cylinder(radius=0.1, length=0.6,
                       color=[0.3, 0.3, 0.5, 1],
                       pose=SE3(2.9, 5.6, 0.1))
robot2.base = robot2_base.T * SE3(0, 0, 0.05)
robot2.q = np.array([np.pi/4 - 0.15, -np.pi/2 + 0.15, - 0.3, -np.pi/2 - 0.15, np.pi/2, 0])
robot3.q = np.array([0,0,0,0,np.pi/2,0])
robot3_ee = Cuboid(scale=[0.30, 0.30, 0.2], color=[0.96, 0.70, 0.82, 1], pose=robot3.fkine(robot3.q) @ SE3(0,0,0.1))
robot1.add_to_env(env)
robot2.add_to_env(env) 
robot3.add_to_env(env)
env.add(robot2_base)
env.add(robot3_base)
env.add(robot3_ee)
gripper_stick_arm2()

# --------------------------------------------------
# 🌟 GUI：手動模式 + 多機械手臂滑桿控制
# --------------------------------------------------

active_robot = robot1   # 預設控制 robot1

# ---------- 模式控制 ----------
def enter_manual_mode(_=None):
    global mode, patrol, pick_and_place, e_stop
    mode = "manual"
    patrol = False
    pick_and_place = False
    e_stop = False
    print("🟡 Entered MANUAL MODE — all robots stopped. You can now control each robot manually via sliders.")

def resume_auto_mode(_=None):
    global mode, patrol
    mode = "patrol"
    patrol = True
    print("🟢 Resumed AUTONOMOUS MODE — robots continue their tasks.")

manual_btn = swift.Button(desc="🟡 Manual Mode", cb=enter_manual_mode)
resume_btn = swift.Button(desc="🟢 Resume Auto", cb=resume_auto_mode)
env.add(manual_btn)
env.add(resume_btn)

# ---------- 選擇控制哪台機械手臂 ----------
def select_robot1(_=None):
    global active_robot
    active_robot = robot1
    print("🎛 Now controlling: Robot1 (Gen3 Lite)")

def select_robot2(_=None):
    global active_robot
    active_robot = robot2
    print("🎛 Now controlling: Robot2 (UR3)")

def select_robot3(_=None):
    global active_robot
    active_robot = robot3
    print("🎛 Now controlling: Robot3 (IRB1200)")

btn_r1 = swift.Button(desc="Control Robot1", cb=select_robot1)
btn_r2 = swift.Button(desc="Control Robot2", cb=select_robot2)
btn_r3 = swift.Button(desc="Control Robot3", cb=select_robot3)
env.add(btn_r1)
env.add(btn_r2)
env.add(btn_r3)

# ---------- 共用滑桿控制 ----------
def slider_callback(value_deg, joint_index):
    global active_robot, mode, e_stop

    # 只有在 manual 模式下才能動
    if mode != "manual" or e_stop:
        return

    q = active_robot.q.copy()
    q[joint_index] = np.deg2rad(float(value_deg))
    active_robot.q = q

    # 更新畫面 (不同機器用不同的更新函式)
    if active_robot == robot1:
        gripper_stick_arm()
        robot1_stick_base()
    elif active_robot == robot2:
        gripper_stick_arm2()
    elif active_robot == robot3:
        robot3_ee.T = active_robot.fkine(active_robot.q)

    env.step(0.02)

# ---------- 建立 6 條滑桿 ----------
sliders = []
for i in range(6):
    s = swift.Slider(
        cb=lambda v, j=i: slider_callback(v, j),
        min=-180, max=180, step=1,
        value=0,
        desc=f"Joint {i+1} (Active Robot)",
        unit="°"
    )
    env.add(s)
    sliders.append(s)

print("✅ Manual control system ready.")
print("Press 🟡 Manual Mode to pause automation, then use Control Robot1/2/3 + sliders to move each robot.")

# 狀態 state
patrol = True 
pick_and_place = False 
target_pos_world = None 
target_ball = None 
holding = False
mode = "patrol"  
e_stop = False   
crusher_trigger = False
crusher_busy = False
current_trash_index = None
current_ur3_object = None
trash_offset_ur3 = None

def crusher_watcher():
    global crusher_trigger, crusher_busy, current_trash_index
    while True:
        if crusher_trigger and not crusher_busy:
            crusher_trigger = False
            crusher_busy = True
            print("🟢 Crusher thread starting...")

            def run_crusher():
                try:
                    # Pass both the object and environment safely
                    crusher_rmrc_trajectory()
                except Exception as e:
                    print(f"⚠️ Crusher error: {e}")
                finally:
                    global crusher_busy
                    crusher_busy = False
                    print("✅ Crusher finished, ready for next trigger")

            threading.Thread(target=run_crusher, daemon=True).start()
        time.sleep(0.1)

# Start watcher thread
threading.Thread(target=crusher_watcher, daemon=True).start()

#gui = RobotGUI(env, robot1, gripper_stick_arm, robot1_stick_base) 

# -------------------------------------------------- 
# 主迴圈 main loop
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
        robot1_stick_base() 
        env.step(0.03) 
        time.sleep(0.03) 
        continue 

    # Patrol 
    if patrol: 
        if 'ur3_thread' not in globals() or not ur3_thread.is_alive():
         ur3_thread = threading.Thread(target=lambda: ur3_pick_and_place())
         ur3_thread.daemon = True
         ur3_thread.start()
            #先轉頭 Turn head first
        total_angle =pi
        angle_step = total_angle / 20 
        for _ in range(20): 
                if mode != "patrol" or e_stop: 
                    break 
                gripper_stick_arm() 
                base_geom.T = base_geom.T * SE3.Rz(angle_step) 
                robot1_stick_base() 
                env.step(0.05) 
                time.sleep(0.05) 

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
                base_step_with_walls(base_geom, step_size) 
                robot1_stick_base() 
                env.step(0.05) 
                time.sleep(0.05) 

                # 偵測球（只在 Patrol 時進行） 
                for ball in list(balls): 
                    ball_pos_world = ball.T[:3, 3] 
                    base_pos = base_geom.T[:3, 3] 
                    dist = np.linalg.norm(ball_pos_world[:2] - base_pos[:2]) 
                    if dist < 0.5: 
                        patrol = False 
                        pick_and_place = True 
                        target_pos_world = ball_pos_world 
                        target_ball = ball 
                        print(f"偵測到球 Ball detected：{target_pos_world}") 
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
                robot1_stick_base() 
                env.step(0.05) 
                time.sleep(0.05) 

    # Pick & Place：Resume 後若已 holding，就跳過「接近＋關夾」 
    elif pick_and_place and target_pos_world is not None and mode == "patrol": 
        # 1) 若尚未抓到，才執行 接近 + 關夾（避免 Resume 後重抓） 
        if not holding: 
            target = SE3(target_pos_world[0], target_pos_world[1], target_pos_world[2] + 0.08) * SE3.Rx(pi) 
            q_pick = robot1.ikine_LM(target, q0=robot1.q).q 
            for q in safe_rrt_path(robot1.q, q_pick): 
                if mode != "patrol" or e_stop: 
                    break 
                robot1.q = q  # 尚未 holding，不更新球 
                gripper_stick_arm() 
                env.step(0.02) 
                
            ee_T = robot1.fkine(robot1.q)                # 末端位姿
            trash_offset = ee_T.inv() * target_ball.T  
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
        q_down = robot1.ikine_LM(area.T * SE3.Rx(pi) * SE3(0, 0, -0.14), q0=robot1.q).q 
        for q in safe_rrt_path(robot1.q, q_down): 
            if mode != "patrol" or e_stop: 
                break 
            robot1.q = q 
            if holding and target_ball is not None: 
                target_ball.T = robot1.fkine(robot1.q) * trash_offset 
            gripper_stick_arm() 
            env.step(0.02) 
        # 打開
        for i in range(50):
            gripper_1.q = traj1[i]
            gripper_2.q = traj2[i]
            gripper_stick_arm() 
            env.step(0.02)

        holding = False 
        R_old = target_ball.T[:3, :3]   # 抓到時的旋轉
        target_ball.T = SE3.Rt(R_old, (area.T * SE3(0, 0, 0.06))[:3, 3])
        ur3_ball=target_ball

        try:
            current_trash_index = balls.index(target_ball)
        except ValueError:
            current_trash_index = None

        balls.remove(target_ball) 
        area_trash.append(ur3_ball)

        RMRC_lift() 
        patrol = True 
        pick_and_place = False 


    else: 
        env.step(0.03) 
        time.sleep(0.03)

