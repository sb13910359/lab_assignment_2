'''
------------------------- IMPORT -------------------------
'''
import os
import time
import random
import threading
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import trapezoidal
from spatialmath import SE3, SO3
from spatialmath.base import rpy2r, tr2rpy
from spatialgeometry import Cylinder, Sphere, Mesh
from ir_support import line_plane_intersection

#import robot classes
from Gen3Lite_mesh import Gen3Lite          #robot1 = gen3lite
from ur3_scaled import UR3_Scaled           #robot2 = ur3        
from irb1200_mesh_v2 import IRB1200         #robot3 = irb1200

#import other custom classes
from environment_builder import EnvironmentBuilder
from robot_gui import RobotGUI
from human import Human

#import serial reader (for hardware e-stop)
import serial


'''
------------------------- COMMON FUNCTIONS -------------------------
'''

def moving_wall_collision(human, baserobot, robot_id):
  
    safe_radius = 0.6

    while True:
        hx, hy, hz = human.T[0, 3], human.T[1, 3], human.T[2, 3]
        T = baserobot.T
        # 六條邊線
        p0_x  = (T * SE3(-0.25, 0, 0))[0:3, 3]
        p1_x  = (T * SE3( 0.25, 0, 0))[0:3, 3]
        p0_y  = (T * SE3(0, -0.25, 0))[0:3, 3]
        p1_y  = (T * SE3(0,  0.25, 0))[0:3, 3]
        p0_d1 = (T * SE3(-0.25, -0.25, 0))[0:3, 3]
        p1_d1 = (T * SE3( 0.25,  0.25, 0))[0:3, 3]
        p0_d2 = (T * SE3(-0.25,  0.25, 0))[0:3, 3]
        p1_d2 = (T * SE3( 0.25, -0.25, 0))[0:3, 3]

        # 動態牆
        moving_planes = {
            "front": {"normal": [0, -1, 0], "point": [hx, hy + safe_radius, hz],
                      "location_x": [hx - safe_radius, hx + safe_radius],
                      "location_y": [hy + safe_radius, hy + safe_radius]},
            "back":  {"normal": [0, 1, 0], "point": [hx, hy - safe_radius, hz],
                      "location_x": [hx - safe_radius, hx + safe_radius],
                      "location_y": [hy - safe_radius, hy - safe_radius]},
            "right": {"normal": [-1, 0, 0], "point": [hx + safe_radius, hy, hz],
                      "location_x": [hx + safe_radius, hx + safe_radius],
                      "location_y": [hy - safe_radius, hy + safe_radius]},
            "left":  {"normal": [1, 0, 0], "point": [hx - safe_radius, hy, hz],
                      "location_x": [hx - safe_radius, hx - safe_radius],
                      "location_y": [hy - safe_radius, hy + safe_radius]},
        }

        hit = False
        for (p0, p1) in [(p0_x, p1_x), (p0_y, p1_y), (p0_d1, p1_d1), (p0_d2, p1_d2)]:
            for plane_name, plane in moving_planes.items():
                n, P = plane["normal"], plane["point"]
                intersect, check = line_plane_intersection(n, P, p0, p1)
                if check == 1:
                    xmin, xmax = plane["location_x"]
                    ymin, ymax = plane["location_y"]
                    if xmin <= intersect[0] <= xmax and ymin <= intersect[1] <= ymax:
                        print(f"E-STOP triggered")
                        set_estop(robot_id)
                        hit = True
                        break
            if hit:
                break
            
            time.sleep(0.05)

def check_collision(q, robot):
    """
    連桿碰撞檢測
    """
    
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


'''
------------------------- ROBOT 1 FUNCTIONS -------------------------
'''

#robot 1
def robot1_main_cycle():
    """
    """
    #宣告會被此函式讀寫的全域變數 
    global target_pos_world, target_ball, holding, trash_offset_gen3, current_trash_index
  

    #看現在是auto 還是manual
    mode = get_mode()
 
    #巡邏+自動狀態
    if state["r1_patrol"] and mode == "auto":

        # 先轉180度
        total_angle = np.pi
        angle_step = total_angle / 20
        for _ in range(20):
            #E-STOP 或 mode不再是 auto，立刻中止
            if is_estop(1) or get_mode() != "auto":
                return
            robot1.gripper.attach_to_robot(robot1)
            base_geom.T = base_geom.T * SE3.Rz(angle_step)
            robot1_stick_base()
            env.step(0.05)
            time.sleep(0.05)

        #巡邏5次
        for _ in range(5):
            if is_estop(1) or get_mode() != "auto":
                return
            
            #隨機走一小段距離
            distance = np.random.uniform(1.0, 2.0)
            step_size = 0.05
            steps = int(distance / step_size)
            for _ in range(steps):
                if is_estop(1) or get_mode() != "auto":
                    return
                robot1.gripper.attach_to_robot(robot1)
                base_step_with_walls(base_geom, step_size) #如果沒撞牆就走一步撞牆就轉彎
                robot1_stick_base()
                env.step(0.05)
                time.sleep(0.05)

                #偵測球
                for ball in list(balls):
                    ball_pos = ball.T[:3, 3]
                    base_pos = base_geom.T[:3, 3]
                    #球跟base距離
                    dist = np.linalg.norm(ball_pos[:2] - base_pos[:2])
                    if dist < 0.5:
                        #找到球後換state
                        #目標球位置
                        target_pos_world = ball_pos
                        target_ball = ball       
                        #找到球後換state                
                        state["r1_patrol"] = False
                        state["pick_and_place"] = True

                        print(f"Ball detected")
                        return

            #隨機轉彎
            total_angle = np.random.uniform(-np.pi, np.pi)
            angle_step = total_angle / 20
            for _ in range(20):
                if is_estop(1) or get_mode() != "auto":
                    return
                robot1.gripper.attach_to_robot(robot1)
                base_geom.T = base_geom.T * SE3.Rz(angle_step)
                robot1_stick_base()
                env.step(0.05)
                time.sleep(0.05)

    #抓球
    elif state["pick_and_place"] :

        #如果沒抓球就執行抓球
        if not holding:
            target = SE3(target_pos_world[0], target_pos_world[1], target_pos_world[2] + 0.08) * SE3.Rx(np.pi)
            q_pick = robot1.ikine_LM(target, q0=first_q).q

            #加這兩行感覺比較不會跳
            robot1_stick_base()
            robot1.gripper.attach_to_robot(robot1)

            #向下要抓球           
            for q in safe_rrt_path(robot1.q, q_pick):
                if is_estop(1) or get_mode() != "auto":
                    return
                robot1.q = q
                robot1.gripper.attach_to_robot(robot1)
                env.step(0.02)

            ee_T = robot1.fkine(robot1.q)
            #記錄垃圾在robot end effector的相對位置
            trash_offset_gen3 = ee_T.inv() * target_ball.T

            #關夾
            for i in range(50):
                if is_estop(1) or get_mode() != "auto":
                    return
                robot1.gripper.close(i=i)
                env.step(0.01)

            holding = True

            #抬起垃圾
            RMRC_lift()

        #回家
        move_base_towards(base_geom, target_xy=(4, 5.7), step_size=0.05)
        
        #加這兩行感覺比較不會跳
        robot1_stick_base()
        robot1.gripper.attach_to_robot(robot1)
        
        #放置
        q_down = robot1.ikine_LM(area.T * SE3.Rx(np.pi) * SE3(0, 0, -0.14), q0=first_q).q
        for q in safe_rrt_path(robot1.q, q_down):
            if is_estop(1) or get_mode() != "auto":
                return
            robot1.q = q
            target_ball.T = robot1.fkine(robot1.q) * trash_offset_gen3
            robot1.gripper.attach_to_robot(robot1)
            env.step(0.02)

        #開夾
        for i in range(50):
            robot1.gripper.open(i=i)
            env.step(0.01)

        holding = False
        #放球       
        target_ball.T = target_ball.T.copy()

        #給UR3球
        ur3_ball = target_ball

        current_trash_index = balls.index(target_ball)#紀錄trash 是哪一個 index 方便之後IRB swap trash


        balls.remove(target_ball)
        area_trash.append(ur3_ball)
        RMRC_lift()
        state["pick_and_place"] = False
        state["r1_patrol"] = True

    # estop 或maunal mode 進入的地方
    else:
        env.step(0.03)
        time.sleep(0.03)

#走一步看有沒有撞牆
def base_step_with_walls(base_geom, step_size=0.05):

    planes = {
            "wall1": {"normal": [0, 1, 0], "point": [0.1, 0, 0],"location_x": [0, 10], "location_y": [0, 10]},
            "wall2": {"normal": [0, 1, 0], "point": [8.5, 8.5, 0],"location_x": [0, 10], "location_y": [0, 10]},
            "wall3": {"normal": [1, 0, 0], "point": [4, 0, 0],"location_x": [0, 10], "location_y": [0, 10]}
        }

    T_now = base_geom.T
    #線條
    p0 = T_now[0:3, 3]                       
    p1 = (T_now * SE3(step_size, 0, 0))[0:3, 3] 
    for plane in planes.values():
        n, P = plane["normal"], plane["point"]      
        intersect, check = line_plane_intersection(n, P, p0, p1)

        if check == 1: 
            xmin, xmax = plane["location_x"]
            ymin, ymax = plane["location_y"]

            # 檢查交點是否在平面定義的矩形區域內
            if xmin <= intersect[0] <= xmax and ymin <= intersect[1] <= ymax:

                # 如果有撞到牆 → 隨機選轉
                angle = np.random.choice([np.pi, -np.pi, np.pi/2, -np.pi/2])
                turn = angle / 20  # 每次要轉的小角度
                print("撞到牆 Hitting the wall")
                for _ in range(20):
                   robot1.gripper.attach_to_robot(robot1) 
                   base_geom.T = base_geom.T * SE3.Rz(turn)
                   if holding == True:
                        target_ball.T = robot1.fkine(robot1.q) * trash_offset_gen3
                   robot1_stick_base() 
                   env.step(0.02)
                   time.sleep(0.02)                  
                print("剛轉彎")              
                return False  #回報撞牆給move_base_towards()知道
  

    # 如果所有平面都沒撞到 → 真的走一步
    base_geom.T = T_now * SE3(step_size, 0, 0)
    return True #回報沒撞給move_base_towards()知道


#往基座走去
def move_base_towards(base_geom, target_xy, step_size=0.05, max_iters=800):
    def _yaw_of(T):
        R = T[:3, :3]
        return np.arctan2(R[1, 0], R[0, 0])  # 算出目前robot在 XY 平面的朝向角度

    it = 0
    while it < max_iters:
        if is_estop(1) or get_mode() == "manual":
            return
        it += 1
        
        #目前base位置
        p = base_geom.T[0:3, 3]
        dx, dy = target_xy[0] - p[0], target_xy[1] - p[1]
        #計算current T到目標T的距離，如果比一步還短，就當作已經到達，停止迴圈。
        if np.hypot(dx, dy) < step_size: # np.hypot(dx, dy)=距離很像 lingnorm
            break

        desired_yaw = np.arctan2(dy, dx)#也就是機器人如果要直直走向目標，頭應該朝的方向
        cur_yaw = _yaw_of(base_geom.T) #底座目前的朝向角
 
        yaw_err = (desired_yaw - cur_yaw + np.pi) % (2*np.pi) - np.pi #把誤差轉成 -pi 到 pi 之間 #就像轉350ep 轉10 度最後朝向一樣
        turn = np.clip(yaw_err, -np.deg2rad(15), np.deg2rad(15)) #限制 value 的數值必須介於 min 和 max 之間
        base_geom.T = base_geom.T * SE3.Rz(turn)

        moved = base_step_with_walls(base_geom, step_size) #嘗試往前走一步，如果成功走了，moved=True；如果被牆擋住，moved=False

        #「如果前面有牆擋住走不動，那就往目標方向的那一邊小轉 15° 再試。
        #if not moved:
        #    base_geom.T = base_geom.T * SE3.Rz(np.sign(yaw_err) * np.deg2rad(15))

        target_ball.T = robot1.fkine(robot1.q) * trash_offset_gen3
        robot1.gripper.attach_to_robot(robot1)
        robot1_stick_base()
        env.step(0.03)
        time.sleep(0.03)


def safe_rrt_path(q1, q2, max_iters=300):
    robot1.q = q1
    env.step()
    time.sleep(0.01)

    q_waypoints = np.array([q1, q2])#目前已知的路徑點，最初只包含 [起點, 終點]
    checked_till_waypoint = 0 #紀錄已經檢查到哪個 waypoint
    q_matrix = [] #完整路徑（會存放所有插值後的關節軌跡）

    iters = 0
    while iters < max_iters:
        if is_estop(1): 
            return np.array(q_matrix) #回傳機器人已經走過路徑
        iters += 1
        start_waypoint = checked_till_waypoint
        progressed = False

        for i in range(start_waypoint, len(q_waypoints)-1):
            if is_estop(1):   
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
                q_rand = (2 * np.random.rand(robot1.n) - 1) * np.pi
                #check會不會撞
                while check_collision(q_rand, robot1):
                    if is_estop(1):  
                        return np.array(q_matrix)
                    #會撞再重新生成一組新的
                    q_rand = (2 * np.random.rand(robot1.n) - 1) * np.pi
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
        
    print(f"找不到")
    return rtb.jtraj(q1, q2, 50).q  
    #如果嘗試了 max_iters 次還沒找到路徑 → 直接回傳直線插值（最後手段）


def robot1_stick_base():
    robot1.base = base_geom.T * SE3(0, 0, 0.12)


def RMRC_lift():
    steps = 60
    delta_t = 0.02
    lift_h = 0.50 #抬升0.5m

    T0 = robot1.fkine(robot1.q).A
    #現在Z
    z0 = T0[2, 3]
    #目標Z
    z1 = z0 + lift_h

    #產生 z0-z1的平滑中間點
    s = trapezoidal(0, 1, steps).q
    z = (1 - s) * z0 + s * z1
    
    #建立一個矩陣來存放 每一步的關節角度
    q_matrix = np.zeros((steps, robot1.n))
    #把目前的機械臂關節角度存到 q_matrix 的第 0 行
    q_matrix[0, :] = robot1.q.copy()
    
    #計算q矩陣
    for i in range(steps - 1):
        zdot = (z[i + 1] - z[i]) / delta_t#Z速

        xdot = np.array([0.0, 0.0, zdot])#x速 (只往上動 所以x,y速都是0)

        J = robot1.jacob0(q_matrix[i, :])#當前關節角度下的 Jacobian 矩陣
        Jv = J[:3, :]

        qdot = np.linalg.pinv(Jv) @ xdot#計算所需關節速度

        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot#下一個關節= 這個關節加上q速(q變化量)

    #走過q
    for q in q_matrix:
        if is_estop(1) or get_mode() != "auto":
            return
        robot1.q = q
        if holding == True:
            target_ball.T = robot1.fkine(robot1.q) * trash_offset_gen3
        robot1.gripper.attach_to_robot(robot1)
        env.step( 0.02)
        time.sleep( 0.02)

'''
------------------------- ROBOT 2 FUNCTIONS -------------------------
'''

#robot2
def rmrc_move_ur3(robot, env, T_start, T_goal,
              steps=80, delta_t=0.015, epsilon=0.05, lambda_max=0.1, 
              follow_object=False, obj=None, obj_offset=None, z_arc=False):
                  
    # Create trajectory in Cartesian space 
    s_profile = trapezoidal(0, 1, steps)        # smooth velocity profile / trapezoidal
    s = s_profile.q
    x = np.zeros((3, steps))                # position trajectory
    theta = np.zeros((3, steps))            # orientation trajectory

    R0     = SO3(T_start.R)             # start orientation as SO3
    R1     = SO3(T_goal.R)               # target orientation  

    for i in range(steps):
        # Linear interpolation between start and goal
        x[0, i] = (1 - s[i]) * T_start.t[0] + s[i] * T_goal.t[0]
        x[1, i] = (1 - s[i]) * T_start.t[1] + s[i] * T_goal.t[1]
        x[2, i] = (1 - s[i]) * T_start.t[2] + s[i] * T_goal.t[2]

        if z_arc == True:
            x[2, i] += 0.18 * np.sin(np.pi * s[i])        #option for upward arc 
    
        # Keep gripper vertical
        #theta[:, i] = [np.pi, 0, 0]
        R_interp   = R0.interp(R1, s[i])         # slerp
        theta[:, i] = tr2rpy(R_interp.R)

    # Initialize storage
    q_matrix = np.zeros((steps, robot.n))
    qdot = np.zeros((steps, robot.n))
    m = np.zeros(steps)
    q_matrix[0, :] = robot.q.copy()

    # RMRC loop
    for i in range(steps - 1):

        if get_mode() != "auto" or not is_robot_active(2):  
            print("🔴 UR3 paused (manual mode or inactive).")
            return

        if is_estop(2):
            print("🚨 E-STOP: UR3")
            return q_matrix[i, :]

        # Forward kinematics
        T = robot.fkine(q_matrix[i, :]).A
        pos, R = T[:3, 3], T[:3, :3]

        # Compute desired motion
        delta_x = x[:, i + 1] - pos
        Rd = rpy2r(theta[0, i + 1], theta[1, i + 1], theta[2, i + 1])
        Rdot = (Rd - R) / delta_t
        S = Rdot @ R.T

        linear_velocity = delta_x / delta_t
        angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
        xdot = np.hstack((linear_velocity, angular_velocity))
        W = np.diag([1,1,1, 0.5,0.5,0.5])
        xdot = W @ xdot

        # Jacobian and manipulability
        J = robot.jacob0(q_matrix[i, :])
        m[i] = np.sqrt(np.linalg.det(J @ J.T))
        if m[i] < epsilon:      #Check if we are near a singularity
            ratio = m[i] / epsilon          ## ranges from 0 (at singularity) to 1 (safe)
            lam = (1 - ratio) * lambda_max  # damping value between 0 and lambda_max
        else:
            lam = 0                 # If robot is not near singularity, no damping needed
        invJ = np.linalg.inv(J.T @ J + lam**2 * np.eye(J.shape[1])) @ J.T       # damped least squares inverse

        # Solve joint velocities
        qdot[i, :] = (invJ @ xdot).T
    

        # Integrate joint motion 
        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot[i, :]

        # Update robot 
        robot.q = q_matrix[i + 1, :]
        robot2.gripper.attach_to_robot(robot2)

        if is_estop(2):
            while is_estop(2):
                time.sleep(0.05)

        # option for picking up objects
        if follow_object and obj is not None:
            global trash_offset_ur3
            if obj_offset is True:
                obj.T = robot.fkine(robot.q) * trash_offset_ur3
            else:
                obj.T = robot.fkine(robot.q) * SE3(0, 0, 0.06) * SE3.Rx(np.pi)
            
        if is_estop(2):
            while is_estop(2):
                time.sleep(0.05)

        env.step(delta_t)

    return q_matrix[-1, :]

#robot2
def ur3_pick_and_place():
    global current_trash_index, current_ur3_object, trash_offset_ur3, crusher_trigger

    q_rest = np.array([np.pi/4 - 0.15, -np.pi/2 + 0.15, - 0.3, -np.pi/2 - 0.15, np.pi/2, 0])
    q_pick  = robot2.ikine_LM(area.T * SE3.Tz(0.1) * SE3.Rx(np.pi),  q0=np.array([np.pi/4 + -0.15, -3*np.pi/4 + 0.15, -np.pi/2 + -0.15, -np.pi/4, np.pi/2, 0])).q
    q_place = robot2.ikine_LM(area_place.T * SE3.Tz(0.1) * SE3.Rx(np.pi), q0=np.array([-3*np.pi/8, -7*np.pi/8, -np.pi/4, -np.pi/4, np.pi/2, 0])).q
    q_box   = robot2.ikine_LM(area_box.T * SE3.Tz(0.1) * SE3.Rx(np.pi),   q0=np.array([-7*np.pi/8, -2*np.pi + 0.15, -np.pi/4, -np.pi/4, np.pi/2, 0])).q

    T_rest  = robot2.fkine(q_rest)
    T_pick  = robot2.fkine(q_pick) * SE3(0, 0, -0.06)
    T_place = robot2.fkine(q_place) * SE3(0, 0, -0.1)
    T_box   = robot2.fkine(q_box)

    while len(area_trash) > 0:

        if is_estop(2):
            while is_estop(2):
                time.sleep(0.3)

        if get_mode() != "auto" or not is_robot_active(2):   
            return
        
        q_current = robot2.q.copy()

        rest_traj = rtb.jtraj(q_current, q_rest, 10).q      #reset to q_rest
        for q in rest_traj:
            robot2.q = q
            robot2.gripper.attach_to_robot(robot2)  
            env.step(0.02)
            time.sleep(0.02)
            

        ur3_ball = area_trash[0]
        current_ur3_object = ur3_ball

        rmrc_move_ur3(robot2, env, T_rest, T_pick)        # traj1

        ee_T = robot2.fkine(robot2.q)
        trash_offset_ur3 = ee_T.inv() * ur3_ball.T

        for i in range(50):
            robot2.gripper.close(i=i)       #close gripper
            env.step(0.01)

        if is_estop(2):
            while is_estop(2):
                time.sleep(0.05)

        while crusher_busy or is_estop(3):
            time.sleep(0.2)

        rmrc_move_ur3(robot2, env, T_pick, T_place, follow_object=True, obj=ur3_ball, obj_offset=True, z_arc=True)       # traj2

        for i in range(50):
            robot2.gripper.open(i=i)        #open gripper
            env.step(0.01)

        ur3_ball.T = ur3_ball.T.copy()

        rmrc_move_ur3(robot2, env, T_place, T_rest)       # traj3

        crusher_trigger = True
        print("🟣 Crusher trigger set! (trash index:", current_trash_index, ")")

        time.sleep(4.0)

        while crusher_busy or is_estop(3):         # UR3 waiting for robot3/IRB1200 to finish before enter its space
            time.sleep(0.2)

        rmrc_move_ur3(robot2, env, T_rest, (T_place * SE3(0,0,0.1)) )       # traj4

        for i in range(50):
            robot2.gripper.close(i=i)       #close gripper
            env.step(0.01)

        while crusher_busy or is_estop(3):
            time.sleep(0.2)

        rmrc_move_ur3(robot2, env, (T_place * SE3(0,0,0.1)), T_box, follow_object=True, obj=crushed, z_arc=True)        # traj5

        for i in range(50):
            robot2.gripper.open(i=i)        #open gripper
            env.step(0.01)

        crushed.T = area_box.T * SE3.Tz(-0.1)

        rmrc_move_ur3(robot2, env, T_box, T_rest)         # traj6

        area_trash.remove(ur3_ball)        # finish dealing with trash from ur3




'''
------------------------- ROBOT 3 FUNCTIONS -------------------------
'''

#robot 3
def swap_to_crushed_object():
    """
    Swaps the current UR3 object with its corresponding crushed version.
    """
    global crushed, current_ur3_object, current_trash_index, squashed_trash_list

    # Safety check: nothing to crush yet
    if current_ur3_object is None or current_trash_index is None:
        print("No current UR3 object to crush.")
        return None

    # Move the original object below the floor
    current_ur3_object.T = SE3(current_ur3_object.T) * SE3(0, -2.0, 0)

    # Bring up its corresponding squashed object
    crushed = squashed_trash_list[current_trash_index]
    crushed.T = area_place.T * SE3.Tz(0.01)
    env.step()
    print(f"🟣 Crushed object #{current_trash_index} swapped in.")
    return crushed   # return the reference

# robot3
def crusher_rmrc_trajectory():

    if get_mode() != "auto" or not is_robot_active(3):   # (for robot3)
        print("🔴 IRB1200 paused (manual mode or inactive).")
        return
    
    q_current = robot3.q.copy()

    q_rest = np.array([-2.729201677813205e-11, 0.07552781760311615, -0.19380546091465126,
                       8.621482220609894e-11, 1.6890746391256057, -1.5707963266042881])
                     
    rest_traj = rtb.jtraj(q_current, q_rest, 10).q      #reset to q_rest if moved around
    for q in rest_traj:
        robot3.q = q
        robot3.ee.attach_to_robot(robot3)
        env.step(0.02)
        time.sleep(0.02)

    steps = 35
    delta_t = 0.02
    epsilon = 0.05          # Manipulability threshold
    lambda_max = 0.1         # Max damping
    W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])  # Linear vs angular weighting

    # Define start & end poses (vertical crush motion)
    R_down_start = SO3.Rx(np.pi)                 # Facing -Z
    R_down_end   = SO3.Rx(np.pi)                 # Same orientation
    T_start = SE3(area_place.T) * SE3(0, 0, 0.77) * SE3.Rx(np.pi)
    T_end   = SE3(area_place.T) * SE3(0, 0, 0.21) * SE3.Rx(np.pi)


    # Generate smooth trapezoidal position trajectory
    s = trapezoidal(0, 1, steps).q
    x = np.zeros((3, steps))
    for i in range(steps):
        x[:, i] = (1 - s[i]) * T_start.t + s[i] * T_end.t

    # Orientation interpolation (SO3 slerp)
    R_traj = [R_down_start.interp(R_down_end, si) for si in s]

    # Initialise storage
    q_matrix = np.zeros((steps, robot3.n))
    qdot = np.zeros((steps, robot3.n))
    m = np.zeros(steps)

    # Initial joint pose
    q_matrix[0, :] = robot3.ikine_LM(
        T_start, q0=np.array([0, np.pi/4, 0, 0, np.pi/4, 0])
    ).q

    print("Starting crusher RMRC sequence...")

    # RMRC Loop (downward motion)
    for i in range(steps - 1):
        if is_estop(3):
            print("🚨 E-STOP active for IRB1200")
            while is_estop(3):
                time.sleep(0.1)  # wait until cleared
            print("🟢 E-STOP cleared for IRB1200")

        # Forward kinematics
        T_now = robot3.fkine(q_matrix[i, :]).A
        pos, R_now = T_now[:3, 3], SO3(T_now[:3, :3])

        # Desired next pose
        delta_x = x[:, i + 1] - pos
        R_next = R_traj[i + 1]
        Rdot = (R_next.R - R_now.R) / delta_t
        S = Rdot @ R_now.R.T

        # Linear + angular velocity
        linear_velocity = delta_x / delta_t
        angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
        xdot = np.hstack((linear_velocity, angular_velocity))
        xdot = W @ xdot

        # Jacobian & manipulability
        J = robot3.jacob0(q_matrix[i, :])
        m[i] = np.sqrt(np.linalg.det(J @ J.T))

        # Adaptive damping
        if m[i] < epsilon:
            ratio = m[i] / epsilon
            lam = (1 - ratio) * lambda_max
        else:
            lam = 0

        # Damped Least Squares inverse
        invJ = np.linalg.inv(J.T @ J + lam**2 * np.eye(J.shape[1])) @ J.T

        # Joint velocities and integration
        qdot[i, :] = (invJ @ xdot).T
        q_next = q_matrix[i, :] + delta_t * qdot[i, :]
        q_next = np.clip(q_next, robot3.qlim[0, :], robot3.qlim[1, :])
        q_matrix[i + 1, :] = q_next

        # Update robot + crusher model
        robot3.q = q_next
        robot3.ee.attach_to_robot(robot3)
        env.step(delta_t)
        time.sleep(delta_t)

    # Try to swap object after crush
    try:
        if is_estop(3):
            print("🚨 E-STOP active for IRB1200")
            while is_estop(3):
                time.sleep(0.1)  # wait until cleared
            print("🟢 E-STOP cleared for IRB1200")
        
        global crushed
        crushed = swap_to_crushed_object()

    except Exception as e:
        print(f"Error during crushing: {e}")


    # Upward release motion
    for q in q_matrix[::-1]:
        if is_estop(3):
            print("🚨 E-STOP active for IRB1200")
            while is_estop(3):
                time.sleep(0.1)  # wait until cleared
            print("🟢 E-STOP cleared for IRB1200")
        robot3.q = q
        robot3.ee.attach_to_robot(robot3)
        env.step(delta_t)
        time.sleep(delta_t)

    print("✅ Crusher RMRC sequence complete!")

'''
------------------------- INITIALISE 始化環境 -------------------------
'''

env_builder = EnvironmentBuilder()
env = env_builder.env

area = env_builder.area
area_place = env_builder.area_place
area_box = env_builder.area_box

#robot init

robot1 = Gen3Lite()
robot2 = UR3_Scaled()
robot3 = IRB1200()

#robot1 base
base_geom=Sphere(radius=0.2, color= (0.45, 0.42, 0.40, 1))
base_geom.T = SE3(6, 5, -0.01) 

#robot2 base
robot2_base = Cylinder(radius=0.1, length=0.6,
                       color=[0.3, 0.3, 0.5, 1],
                       pose=SE3(2.9, 5.6, 0.1))
robot2.base = robot2_base.T * SE3(0, 0, 0.05)

#robot3 base
robot3.base = robot3.base * area_place.T * SE3.Ty(-0.37) * SE3.Rz(np.pi/2) 
robot3_base = Cylinder(radius=0.25, length=0.6,
                       color=(0.20, 0.12, 0.06),
                       pose=robot3.base * SE3(0, 0, -0.31)) 

env.add(base_geom)
env.add(robot2_base)
env.add(robot3_base)

first_q = robot1.q.copy()
robot2.q = np.array([np.pi/4 - 0.15, -np.pi/2 + 0.15, - 0.3, -np.pi/2 - 0.15, np.pi/2, 0])
robot3.q = np.array([0, 0, - 0.15, 0, np.pi/2, -np.pi/2])

robot1.add_to_env(env)
robot2.add_to_env(env) 
robot3.add_to_env(env)

robot1.gripper.add_to_env(env)
robot2.gripper.add_to_env(env)
robot3.ee.add_to_env(env)

robot1.gripper.attach_to_robot(robot1)
robot2.gripper.attach_to_robot(robot2)
robot3.ee.attach_to_robot(robot3)


#object init 

current_dir = os.path.dirname(os.path.abspath(__file__))
obj_dir = os.path.join(current_dir, "meshes", "obj")

#spawn trash
bottle3_stl_path = os.path.join(obj_dir, "bottle3.stl")
bottle2_stl_path = os.path.join(obj_dir, "bottle2.stl")
paper2_stl_path = os.path.join(obj_dir, "paper2.stl")

def spawn_random_trash(
    env,
    n_items: int = 30,
    x_range=(4, 9),
    y_range=(0, 9),
    floor_z: float = 0.05,
):
    """
    Spawn random trash meshes into the environment and a matching list of
    'crushed' cylinders kept index-aligned with the original items.
    """
    balls = []
    area_trash = []               
    squashed_trash_list = []

    for _ in range(n_items):
        # Randomly choose a trash STL type
        trash_type = random.choice([
            (bottle3_stl_path, [0.35, 0.35, 0.35], [np.random.uniform(0.0, 0.2), np.random.uniform(0.2, 0.5), np.random.uniform(0.5, 0.9), np.random.uniform(0.4, 0.7)]),  # 藍瓶
            (bottle2_stl_path, [0.35, 0.35, 0.25], [np.random.uniform(0.7, 1.0), np.random.uniform(0.2, 0.5), np.random.uniform(0.0, 0.2), np.random.uniform(0.5, 1.0)]),  # 橘紅瓶
            (paper2_stl_path,  [0.0018, 0.0018, 0.0018], [0.92, 0.92, 0.92, 1])  # 小紙屑
        ])
        fname, scale, color = trash_type

        # Random floor position + orientation
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        pose = SE3(x, y, floor_z) * SE3.Rx(np.pi/2) * SE3.Ry(np.random.uniform(-np.pi, np.pi))

        # Visible trash
        trash = Mesh(fname, pose=pose, scale=scale, color=color)
        env.add(trash)
        balls.append(trash)

        # Hidden crushed trash (kept below floor; same index as trash)
        crushed = Cylinder(
            radius=0.05 * 1.3,
            length=0.05 * 0.25,
            color=[0.5, 0.5, 0.5, 1],
            pose=SE3(x, y, -1.0)
        )
        env.add(crushed)
        squashed_trash_list.append(crushed)

    return balls, area_trash, squashed_trash_list

balls, area_trash, squashed_trash_list = spawn_random_trash(env)

#add human / mobile collision obj
human_obj = Human(env, obj_dir)


'''
------------------------- STATES, E-STOP AND GUI -------------------------
'''


state = {
    "mode": "auto",
    "auto": True,
    "r1_patrol": True,        # robot1 patrol flag
    "r2_active": True,
    "r3_active": True,

    "pick_and_place": False,
    "target_pos_world": None,
    "target_ball": None,
    "e_stop": False
}

# 狀態 state
target_pos_world = None 
target_ball = None 
holding = False

crusher_trigger = False
crusher_busy = False

current_trash_index = None
current_ur3_object = None
trash_offset_ur3 = None
trash_offset_gen3 = None

r2_thread = None


# E-STOP SYSTEM 
estop_r1 = threading.Event()       # Robot 1 (Gen3Lite)
estop_r2 = threading.Event()       # Robot 2 (UR3)
estop_r3 = threading.Event()       # Robot 3 (IRB1200)

gui = RobotGUI(
    env=env,
    robot1=robot1,
    robot2=robot2,
    robot3=robot3,
    gripper_stick_arm=lambda: robot1.gripper.attach_to_robot(robot1),
    robot1_stick_base=robot1_stick_base,
    gripper_stick_arm2=lambda: robot2.gripper.attach_to_robot(robot2),
    update_robot3_ee=lambda: robot3.ee.attach_to_robot(robot3),
    state_dict=state,
    set_estop_func=lambda rid=None, val=True: set_estop(rid, val),
    clear_estop_func=lambda rid=None: set_estop(rid, False)
)

def is_estop(robot_id=None):
    """Check E-STOP for a given robot or global."""
    if robot_id == 1:
        return estop_r1.is_set() or state.get("r1_estop", False)
    elif robot_id == 2:
        return estop_r2.is_set() or state.get("r2_estop", False)
    elif robot_id == 3:
        return estop_r3.is_set() or state.get("r3_estop", False)
    return False

def set_estop(robot_id=None, value=True):
    """Set or clear E-STOP for a specific robot"""
    if robot_id is None:  
        print("Please select which robot to target.")
        return

    event = {1: estop_r1, 2: estop_r2, 3: estop_r3}.get(robot_id)
    if event is None:
        return
    
    if value:
        event.set()
        state[f"r{robot_id}_estop"] = True     # sync GUI state
        state["e_stop"] = True
        sync_estop_label(robot_id, "active")
        print(f"🚨 E-STOP ON — Robot {robot_id} halted.")
    else:
        event.clear()
        state[f"r{robot_id}_estop"] = False    # sync GUI state
        state["e_stop"] = False
        sync_estop_label(robot_id, "clear")
        set_mode("manual")
        print(f"✅ E-STOP CLEARED — Robot {robot_id} ready.")
 
def get_mode():
    """Return current robot mode (auto/manual/etc.)."""
    return state["mode"]

def set_mode(value: str):
    """Change robot mode and keep GUI in sync."""
    state["mode"] = value

def activate_robot(robot_id, active=True):
    """Activate or deactivate robot based on current mode."""
    if robot_id == 2:
        state["r2_active"] = active
        print(f"{'🟢' if active else '🔴'} Robot 2 {'activated' if active else 'paused'}")
    elif robot_id == 3:
        state["r3_active"] = active
        print(f"{'🟢' if active else '🔴'} Robot 3 {'activated' if active else 'paused'}")

def is_robot_active(robot_id):
    """Return whether a robot is currently active."""
    if robot_id == 2:
        return state.get("r2_active", False)
    elif robot_id == 3:
        return state.get("r3_active", False)
    return False

def keep_swift_alive(env):
    """Ensures Swift keeps rendering even during E-STOPs."""
    while True:
        env.step(0.05)
        time.sleep(0.05)

def sync_estop_label(robot_id, state_label="active"):
    """Update GUI E-STOP button label & colour when triggered externally."""
    btn_map = {1: gui.estop_btn_r1, 2: gui.estop_btn_r2, 3: gui.estop_btn_r3}
    if robot_id not in btn_map:
        return
    btn = btn_map[robot_id]

    if state_label == "active":
        btn.desc = f"🚨 E-STOP ACTIVE (Robot {robot_id})"

    elif state_label == "confirm":
        btn.desc = f"⚪ RELEASE E-STOP? (Robot {robot_id})"

    elif state_label == "clear":
        btn.desc = f"E-STOP (Robot {robot_id})"


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

        if is_estop(3) and crusher_busy:
            print("IRB1200 E-STOP while busy")
        time.sleep(0.1)

# Arduino E-STOP
def hardware_estop_listener():
    """Listens to Arduino serial for 'ESTOP' or 'CLEAR' commands."""
    while True:
        if arduino:
            try:
                line = arduino.readline().decode('utf-8').strip()

                if line.startswith("ESTOP_R"):
                    rid = int(line[-1])             #check which button is pressed
                    print(f"🚨 Hardware E-STOP pressed: Robot {rid}")
                    set_estop(rid, True)
                    sync_estop_label(rid, "active")

                elif line.startswith("CONFIRM_R"):  #confirm?
                    rid = int(line[-1])             
                    print(f"⚪ Confirm release requested: Robot {rid}")
                    sync_estop_label(rid, "confirm")

                elif line.startswith("CLEAR_R"):       #clear estop
                    rid = int(line[-1])
                    print(f"✅ Hardware E-STOP cleared: Robot {rid}")
                    set_estop(rid, False)
                    sync_estop_label(rid, "clear")
                

            except Exception as e:
                print("⚠️ Arduino read error:", e)
        time.sleep(0.05)


try:
    arduino = serial.Serial('COM4', 9600, timeout=0.1)   
    print("Arduino connected for hardware E-STOP.")
except Exception as e:
    print(f"Arduino not found: {e}")
    arduino = None

# Start watcher threads
threading.Thread(target=crusher_watcher, daemon=True).start()
threading.Thread(target=moving_wall_collision, args=(human_obj.human,base_geom, 1), daemon=True).start()
threading.Thread(target=moving_wall_collision, args=(human_obj.human,robot3_base, 3), daemon=True).start()
threading.Thread(target=moving_wall_collision, args=(human_obj.human,robot2_base, 2), daemon=True).start()
threading.Thread(target=keep_swift_alive, args=(env,), daemon=True).start()
threading.Thread(target=hardware_estop_listener, daemon=True).start()

'''
------------------------- MAIN LOOP 主迴圈 -------------------------
'''

while True:
    mode = get_mode()

    # --- AUTO MODE ---
    if mode == "auto":
        # Start or resume Robot 2 (UR3)
        if not r2_thread or not r2_thread.is_alive():
            if not is_robot_active(2):
                activate_robot(2, True)
            r2_thread = threading.Thread(target=ur3_pick_and_place, daemon=True)
            r2_thread.start()

        # Start or resume Robot 3 (Crusher)
        if not is_robot_active(3):
            activate_robot(3, True)


    # --- MANUAL MODE ---
    elif mode == "manual":
        if is_robot_active(2):
            activate_robot(2, False)
        if is_robot_active(3):
            activate_robot(3, False)

        robot1.gripper.attach_to_robot(robot1)
        robot1_stick_base()
        env.step(0.03)
        time.sleep(0.03)
        continue

    # --- ROBOT 1 MAIN LOOP ---
    robot1_main_cycle()












