# Require libraries
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import swift
from spatialmath.base import *
from spatialmath import SE3
from spatialgeometry import Sphere, Arrow, Mesh
from roboticstoolbox import DHLink, DHRobot, models
from ir_support import CylindricalDHRobotPlot
import os

# Useful variables
from math import pi



point_freq = 1    # Frequency of points to plot end-effector path (1 = every point, 2 = every two points, etc) -
                  # increase if CPU lagging     

# 1.1) Make a 3DOF planar arm model


# Define the robot using DH parameters
l1 = DHLink(d=0, a=1, alpha=0, qlim=[-pi, pi])
l2 = DHLink(d=0, a=1, alpha=0, qlim=[-pi, pi])
l3 = DHLink(d=0, a=1, alpha=0, qlim=[-pi, pi])
robot = DHRobot([l1, l2, l3], name='my_robot')

# Give the robot a cylinder mesh (links) to display in Swift environment
cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="#3478f6")
robot = cyl_viz.create_cylinders()


# 1.2) Rotate the base around the X axis so the Z axis faces down ways (and the Y-axis is flipped)

robot.base = trotx(pi)

# 1.3, 1.4) Set workspace, scale and initial joint state, then plot and teach

workspace = [-3, 3, -3, 3, -0.05, 2]

q = np.zeros([1,3])

fig = robot.teach(q, limits = workspace, block = False)


# 1.5, 1.6) Get a solution for the end effector at [-0.75,-0.5,0]. 

# 1.7) Check how close it got to the goal transform of transl(-0.75,-0.5,0)

# 1.8, 1.9) Go through a loop using the previous joint as the guess to draw a line from [-0.75,-0.5,0] to [-0.75,0.5,0]

# 1.10) Using ikine to get the newQ and fkine to determine the actual point, 
# and animate to move the robot to “draw” a circle around the robot with a radius of 0.5m

# 1.11 + 1.12) Add a pen (this part will only work if the .dae file is in the same folder as this script!)

# 1.13) Set pen transform to be at robot end-effector and translate it 0.1m along Z

# 1.14) Move the pen as if it were on the end-effector through a naively-created arm trajectory

