import numpy as np
import time
import swift
from spatialmath import SE3, SO3
from roboticstoolbox import models, trapezoidal
from spatialgeometry import Cuboid

# -----------------------------
# 1) Load a 6DOF robot (UR5 example)
ur5 = models.UR5()
steps = 50
delta_t = 0.05

# -----------------------------
# 2) Define start and end poses
T_start = SE3(0.5, 0, 0.70)
T_end   = SE3(0.5, 0, 0.0)

# -----------------------------
# 3) Interpolate Cartesian trajectory (positions only)
s = trapezoidal(0, 1, steps).q
pos_traj = np.zeros((3, steps))
for i in range(steps):
    pos_traj[:, i] = (1 - s[i]) * T_start.t + s[i] * T_end.t

# -----------------------------
# 4) Initialize joint trajectory
q_matrix = np.zeros((steps, ur5.n))
q_matrix[0, :] = ur5.ikine_LM(T_start, q0=np.array([0, -np.pi/2, 0, 0, np.pi/2, np.pi/2])).q

# -----------------------------
# 5) RMRC loop (position + fixed orientation)
for i in range(steps - 1):
    xdot = (pos_traj[:, i+1] - pos_traj[:, i]) / delta_t
    T = SE3(ur5.fkine(q_matrix[i, :]))

    # Orientation error
    R_err = T.R.T
    rotvec_err = SO3(R_err).rpy()
    omega = rotvec_err / delta_t

    v = np.hstack((xdot, omega))
    J = ur5.jacob0(q_matrix[i, :])
    q_dot = np.linalg.pinv(J) @ v
    q_matrix[i+1, :] = q_matrix[i, :] + delta_t * q_dot

# -----------------------------
# 6) Launch Swift visualization
env = swift.Swift()
env.launch(realtime=True)
env.add(ur5)

# Add an arrow to represent EE direction (red cylinder)
ee = Cuboid(scale=[0.05, 0.05, 0.1], color=[1.0, 0.0, 0.0, 1.0])
env.add(ee)

# Add object(s) to compress
bottle = Cuboid(scale=[0.05, 0.05, 0.15], color=[0.0, 1.0, 0.0, 0.5])
bottle.T = SE3(0.5, 0, 0)
env.add(bottle)

# -----------------------------
# 7) Animate robot motion in Swift
for q in q_matrix:
    ur5.q = q
    T = ur5.fkine(q)
    ee.T = T
    env.step()
    time.sleep(delta_t)

for q in q_matrix[::-1]:  # iterate backwards
    ur5.q = q
    T = ur5.fkine(q)
    ee.T = T
    env.step()
    time.sleep(delta_t)

input("Press Enter to exit...")
env.close()
