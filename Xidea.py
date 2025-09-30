from itertools import combinations 
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
import swift
import roboticstoolbox as rtb
from spatialmath import SE3
from ir_support.robots.DHRobot3D import DHRobot3D
import os, time
from math import pi
import spatialmath.base as spb
# 初始化環境 
env = swift.Swift() 
env.launch(realtime=True) 
env.set_camera_pose([3, 3, 2], [0, 0, 0]) 
#####

class Gen3Lite(DHRobot3D):
    def __init__(self):
        """
        Kinova Gen3 Lite Robot using DH parameters and STL visualization
        """
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
        d = [0.128, 0, 0, 0.2428, -0.055, 0.15]   # ✅ 最後一節沿 z 偏移 0.0285
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


#讓底座嘗試前進一步並檢查有沒有撞牆
def base_step_with_walls(base_geom, step_size=0.05):
    planes = {
            "wall1": {"normal": [0, 1, 0], "point": [0, 0, 0],"location_x": [0, 10], "location_y": [0, 10]},
            "wall2": {"normal": [0, 1, 0], "point": [9, 9, 0],"location_x": [0, 10], "location_y": [0, 10]}
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
                angle = np.random.choice([np.pi/4, -np.pi/4, np.pi/2, -np.pi/2])
                turn = angle / 10  # 每次要轉的小角度
                print("撞到牆")
                for _ in range(10):
                   base_geom.T = base_geom.T * SE3.Rz(turn)
                   env.step(0.02)  # 更新環境 (動畫更順)
                   time.sleep(0.02)  # 控制轉動速度
                return False

    # 如果所有平面都沒撞到 → 真的走一步
    base_geom.T = T_now * SE3(step_size, 0, 0)
    return True


#往基座走去
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
            target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)

        gripper_stick_arm()
        robot_stick_base()
        env.step(0.03)
        time.sleep(0.03)

# --------------------------------------------------
#連桿碰撞檢測
# --------------------------------------------------
def check_collision(q):
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

# --------------------------------------------------
# RRT（安全路徑）
# --------------------------------------------------
def safe_rrt_path(q1, q2, max_iters=300):
    robot.q = q1
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
            is_collision_check = any(check_collision(q) for q in q_traj)
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
                if not any(check_collision( q) for q in q_traj2):
                    #把剩下路徑加到qmatrix
                    q_matrix.extend(q_traj2.tolist())
                    return np.array(q_matrix)
            else:
                #有撞到
                #隨機加一組q (-pi到pi)
                q_rand = (2 * np.random.rand(robot.n) - 1) * pi
                #check會不會撞
                while check_collision(q_rand):
                    if e_stop:  
                        return np.array(q_matrix)
                    #會撞再重新生成一組新的
                    q_rand = (2 * np.random.rand(robot.n) - 1) * pi
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

# --------------------------------------------------
# 機器人 / 夾爪
# --------------------------------------------------
def robot_stick_base():
    robot.base = base_geom.T * SE3(0, 0, 0.12)

def gripper_stick_arm():
    arm_T = robot.fkine(robot.q) * SE3(0.03, 0, 0)
    adjust = SE3.Ry(-pi/2) * SE3(0, 0, 0.03) * SE3.Rx(-pi/2)
    gripper_1.base = arm_T * adjust
    gripper_2.base = arm_T * adjust

def RMRC_lift():
    steps = 60
    delta_t = 0.02
    lift_h = 0.50#抬升的總高度 = 0.5 公尺

    T0 = robot.fkine(robot.q).A
    z0 = T0[2, 3]
    z1 = z0 + lift_h

    #產生 z0-z1的平滑中間點
    s = trapezoidal(0, 1, steps).q
    z = (1 - s) * z0 + s * z1
    #建立一個矩陣來存放 每一步的關節角度
    q_matrix = np.zeros((steps, robot.n))
    #把目前的機械臂關節角度存到 q_matrix 的第 0 行
    q_matrix[0, :] = robot.q.copy()

    for i in range(steps - 1):
        if e_stop:  
            return
        #Z速
        zdot = (z[i + 1] - z[i]) / delta_t
        #x速
        xdot = np.array([0.0, 0.0, zdot])
        #當前關節角度下的 Jacobian 矩陣
        J = robot.jacob0(q_matrix[i, :])
        Jv = J[:3, :]
        #計算所需關節速度
        qdot = np.linalg.pinv(Jv) @ xdot
        #下一個關節= 這個關節加上q速(q變化量)
        q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot
    #走過q
    for q in q_matrix:
        if e_stop:   
            return
        robot.q = q
        if holding == True:
            target_ball.T = robot.fkine(robot.q)*SE3(0,0,0.06)
        gripper_stick_arm()
        env.step( 0.02)
        time.sleep( 0.02)


def go_to_home():
   
        move_base_towards(base_geom, target_xy=(3.2, 2), step_size=0.05)
        target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06)
        gripper_stick_arm()
        env.step(0.03)
        time.sleep(0.03)
      
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
env.add(floor) ;env.add(Workingzone) ;env.add(table) ;env.add(table2) 
env.add(wall1) ;env.add(wall2) ;env.add(wall3) 

# 隨機小球 
balls = [] 
for _ in range(30): 
    s = Sphere(radius=0.05, color=np.random.rand(3).tolist()) 
    s.T = SE3(np.random.uniform(4, 9), np.random.uniform(0, 9),  0.05 ) 
    env.add(s) 
    balls.append(s) 

# 夾爪（可視化用 DH 兩節） 
l1 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
l2 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
gripper1 = DHRobot([l1, l2], name="gripper1") 
l11 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
l22 = DHLink(d=0, a=0.045, alpha=0, qlim=[-pi, pi]) 
gripper2 = DHRobot([l11, l22], name="gripper2") 

g1 = CylindricalDHRobotPlot(gripper1, cylinder_radius=0.01, color="#7D7060") 
gripper_1 = g1.create_cylinders() 
g2 = CylindricalDHRobotPlot(gripper2, cylinder_radius=0.01, color="#5C5247") 
gripper_2 = g2.create_cylinders() 
env.add(gripper_1) ;env.add(gripper_2) 

# 夾爪開關
q1_open = [-pi / 2.5, pi / 3.5] 
q2_open = [ pi / 2.5, -pi / 3.5] 
q1_close = [-pi / 4, pi / 5] 
q2_close = [ pi / 4, -pi / 5] 
gripper_1.q = q1_open 
gripper_2.q = q2_open 
traj3 = rtb.jtraj(q1_open, q1_close, 50).q 
traj4 = rtb.jtraj(q2_open, q2_close, 50).q 

# 機器人與基座 
#base_geom = Cylinder(radius=0.2, length=0.25, color=[0.3, 0.3, 0.3, 1]) 
base_geom=Sphere(radius=0.2, color= (0.45, 0.42, 0.40, 1))
base_geom.T = SE3(5, 5, -0.01) 
env.add(base_geom) 

area = Cuboid(scale=[0.6, 0.6, 0.01], color=[1, 0.6, 0, 1]) 
area.T = SE3(2.5, 2, 0.05) 
env.add(area) 

robot = Gen3Lite() 
robot.add_to_env(env) 

# 狀態 
patrol = True 
pick_and_place = False 
target_pos_world = None 
target_ball = None 
holding = False 
mode = "patrol"  
e_stop = False   

# -------------------------------------------------- 
# GUI 
# -------------------------------------------------- 
#(關節控制)用silder 傳進來的值去跟新關節角
def slider_callback(value_deg, joint_index): 
    if mode =="patrol" or e_stop: 
        return 
    q = robot.q.copy() 
    q[joint_index] = np.deg2rad(float(value_deg)) 
    robot.q = q 
    if holding ==True: 
        target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06) 
    gripper_stick_arm() 
    robot_stick_base() 
    env.step(0.02) 
#(末端控制)傳detla 進去 改末端位置IK算所需關節角度更新關節 (這段很不順)
def cartesian_callback(delta, axis): 
    if mode =="patrol"  or e_stop: 
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
    q=rtb.jtraj(robot.q,q_new,50).q
    for q in q:
      robot.q = q
      gripper_stick_arm()
      robot_stick_base()
      env.step(0.02)
      time.sleep(0.02)
      if holding ==True: 
          target_ball.T = robot.fkine(robot.q) * SE3(0, 0, 0.06) 
    gripper_stick_arm() 
    robot_stick_base() 
    env.step(0.02) 

# Buttons / Sliders 
#建立一個 按鈕，文字是 "Manual Mode"
# cb= 是 callback（點擊按鈕會執行的動作）。
# 把 mode 設成 "manual"
# 把 patrol 設成 False
# 把 pick_and_place 設成 False
manual_btn = swift.Button(desc="Manual Mode", cb=lambda _=None: (globals().__setitem__('mode', 'manual'), globals().__setitem__('patrol', False), globals().__setitem__('pick_and_place', False))) 
patrol_btn = swift.Button(desc="Patrol Mode", cb=lambda _=None: (globals().__setitem__('mode', 'patrol'), globals().__setitem__('patrol', True))) 
#"Patrol Mode" 按鈕。按下後會：把 mode 設成 "patrol"把 patrol 設成 True
estop_btn = swift.Button(desc="--E-STOP--", cb=lambda _=None: globals().__setitem__('e_stop', True)) 
resume_btn = swift.Button(desc="-- Resume--", cb=lambda _=None: globals().__setitem__('e_stop', False)) 
env.add(manual_btn); env.add(patrol_btn) 
env.add(estop_btn); env.add(resume_btn) 
# Joint sliders 
for i in range(robot.n): 
    s = swift.Slider(cb=lambda v, j=i: slider_callback(v, j), min=-180, max=180, step=1, value=np.rad2deg(robot.q[i]), desc=f'Joint {i+1}', unit='°') 
    env.add(s) 
# Cartesian sliders
env.add(swift.Slider(cb=lambda v: cartesian_callback(v*0.01, "x"), min=-10, max=10, step=1, value=0, desc="ΔX", unit="cm")) 
#v → 來自滑桿的值, 然後傳進去callback 裡面
env.add(swift.Slider(cb=lambda v: cartesian_callback(v*0.01, "y"), min=-10, max=10, step=1, value=0, desc="ΔY", unit="cm")) 
env.add(swift.Slider(cb=lambda v: cartesian_callback(v*0.01, "z"), min=-10, max=10, step=1, value=0, desc="ΔZ", unit="cm")) 

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
                base_step_with_walls(base_geom, step_size) 
                robot_stick_base() 
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
                        print(f"偵測到球：{target_pos_world}") 
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
                robot.q = q  # 尚未 holding，不更新球 
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
        holding = False 
        target_ball.T = area.T * SE3(0, 0, 0.06) 
        balls.remove(target_ball) 

        RMRC_lift() 
        patrol = True 
        pick_and_place = False 

    else: 
        env.step(0.03) 
        time.sleep(0.03)
