import numpy as np
import time
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Sphere
from roboticstoolbox import models
from ir_support import line_plane_intersection   # using your existing function

# -----------------------------
# 1) Load robot + Swift environment
robot = models.UR5()
env = swift.Swift()
env.launch(realtime=True)
env.add(robot)

print("✅ Loaded UR5 robot and launched Swift environment.")

# -----------------------------
# 2) Define "safety zone" walls (planes)
plane1_normal = np.array([1, 0, 0])      # facing -x
plane1_point = np.array([0.6, 0, 0])     # passes through (0.6, 0, 0)
wall1 = Cuboid(scale=[0.01, 1.2, 1.2], color=[1, 0, 0, 0.3])
wall1.T = SE3(0.6, 0, 0.6)
env.add(wall1)

plane2_normal = np.array([0, 1, 0])      # facing -y
plane2_point = np.array([0, 0.6, 0])     # passes through (0, 0.6, 0)
wall2 = Cuboid(scale=[1.2, 0.01, 1.2], color=[0, 0, 1, 0.3])
wall2.T = SE3(0, 0.6, 0.6)
env.add(wall2)

print("✅ Defined safety zone walls at x=0.6 and y=0.6.")

# -----------------------------
# 3) Define a simple trajectory
q_start = [0, 0, 0, 0, 0, 0]
q_goal = [0, -1.2, 1.2, 0, 0, 0]
steps = 50
q_traj = np.linspace(q_start, q_goal, steps)

print(f"✅ Trajectory generated: {steps} steps")

# -----------------------------
# 4) Animate with collision detection
collisions = []

for step_idx, q in enumerate(q_traj):
    print(f"\n🔄 Step {step_idx+1}/{steps}: Moving robot...")
    robot.q = q
    trs = robot.fkine_all(q)   # list of SE3 transforms

    # Check each link segment
    for i in range(len(trs) - 1):
        p1 = trs[i].t
        p2 = trs[i+1].t

        # Check plane 1
        intersect, check = line_plane_intersection(plane1_normal, plane1_point, p1, p2)
        if check == 1:
            print(f"⚠️ Collision with Wall 1 at {intersect}")
            ball = Sphere(radius=0.03, color=[1, 0, 0, 1])
            ball.T = SE3(intersect[0], intersect[1], intersect[2])
            env.add(ball)
            collisions.append(ball)

        # Check plane 2
        intersect, check = line_plane_intersection(plane2_normal, plane2_point, p1, p2)
        if check == 1:
            print(f"⚠️ Collision with Wall 2 at {intersect}")
            ball = Sphere(radius=0.03, color=[0, 0, 1, 1])
            ball.T = SE3(intersect[0], intersect[1], intersect[2])
            env.add(ball)
            collisions.append(ball)

    env.step()
    time.sleep(0.05)

input("\n✅ Simulation finished. Press Enter to exit...")
env.close()
