import numpy as np
import threading, time
import swift
from math import pi
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from spatialmath.base import tr2rpy
from spatialgeometry import Cuboid, Cylinder, Mesh
import os

# -------------------------------------------------------------------------------------- #
# 1) Define IRB1200 robot
def create_irb1200():
    # DH parameters from table:
    # d1=399 mm, a2=350 mm, a3=42 mm, d4=351 mm, d6=82 mm


    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_stl_path = os.path.join(current_dir, "base.stl")
    link1_stl_path = os.path.join(current_dir, "link1.stl")
    link2_stl_path = os.path.join(current_dir, "link2.stl")
    link3_stl_path = os.path.join(current_dir, "link3.stl")
    link4_stl_path = os.path.join(current_dir, "link4.stl")
    link5_stl_path = os.path.join(current_dir, "link5.stl")
    link6_stl_path = os.path.join(current_dir, "link6.stl")
    
    links = [
        # Joint 1
        RevoluteDH(
            d=0.0, a=0.0, alpha=-np.pi/2, offset=0.0,
            geometry=[Mesh(filename=link1_stl_path, color=[0.7, 0.7, 0.7, 1], scale=[0.001]*3, pose=SE3.Rx(np.pi/2))]
        ),
        
        # Joint 2 
        RevoluteDH(
            d=0.0, a=350e-3, alpha=0.0, offset=-np.pi/2,
            geometry=[Cuboid(scale=[0.35, 0.08, 0.08], color=[0.8, 0.2, 0.2, 1], pose=SE3.Tx(0.35/2))]
        ),

        # Joint 3
        RevoluteDH(
            d=0.0, a=42e-3, alpha=-np.pi/2, offset=0.0,
            geometry=[Cuboid(scale=[0.25, 0.08, 0.08],color=[0.2, 0.8, 0.2, 1],pose=SE3.Tx(0.25/2) @ SE3.Ry(np.pi/2) @ SE3.Tz(0.1))]
        ),

        # Joint 4
        RevoluteDH(
            d=351e-3, a=0.0, alpha=np.pi/2, offset=0.0,
            geometry=[Cuboid(scale=[0.08, 0.08, 0.30], color=[0.2, 0.2, 0.8, 1], pose=SE3.Tz(0.351/2))]
        ),

        # Joint 5
        RevoluteDH(
            d=0.0, a=0.0, alpha=-np.pi/2, offset=np.pi/2,
            geometry=[Cylinder(radius=0.04, length=0.1, color=[0.9, 0.6, 0.2, 1],) ]
        ),

        # Joint 6 
        RevoluteDH(
            d=82e-3, a=0.0, alpha=0.0, offset=-np.pi,
            geometry=[Cuboid(scale=[0.03, 0.03, 0.12], color=[0.3, 0.3, 0.3, 1])]
        ),
    ]

    robot = DHRobot(links, name="IRB1200_from_table")

    # Attach base mesh (static, follows robot.base)
    base_mesh = Mesh(filename=base_stl_path, color=[0.5, 0.5, 0.5, 1], scale=[0.001]*3)

    # Attach it as an attribute so we can add to Swift later
    robot.base_mesh = base_mesh

    # Set joint limits (in radians, converted from degrees)
    robot.qlim = np.deg2rad([
        [ -170,  170 ],   # Joint 1
        [ -100,  135 ],   # Joint 2
        [ -200,   70 ],   # Joint 3
        [ -270,  270 ],   # Joint 4
        [ -130,  130 ],   # Joint 5
        [ -360,  360 ]    # Joint 6
    ]).T  # .T so each column = [min, max] for a joint

    return robot


# -------------------------------------------------------------------------------------- #
# 2) Teach Pendant GUI
class TeachPendant:
    def __init__(self, robot):
        self.robot = robot
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.env.add(robot)
        self.env.add(robot.base_mesh)
        self.labels = []
        self.sliders = []

        self.stop_event = threading.Event()
        threading.Thread(target=self.wait_for_enter).start()

    def wait_for_enter(self):
        input("Press Enter to quit teach pendant...\n")
        self.stop_event.set()

    def run(self):
        # Add XYZ + RPY labels
        self.labels = [
            swift.Label("X: 0"),
            swift.Label("Y: 0"),
            swift.Label("Z: 0"),
            swift.Label("Roll: 0"),
            swift.Label("Pitch: 0"),
            swift.Label("Yaw: 0"),
        ]
        for lbl in self.labels:
            self.env.add(lbl)

        # Add 6 sliders
        for i in range(6):
            slider = swift.Slider(
                cb=lambda val, idx=i: self.slider_callback(val, idx),
                min=-180, max=180, step=1, value=0,
                desc=f"Joint {i+1}", unit="°"
            )
            self.env.add(slider)
            self.sliders.append(slider)

        # Simple test object in workspace
        cube = Cuboid(scale=[0.2,0.2,0.2], color=[0,1,0,0.5])
        cube.T = np.eye(4)
        self.env.add(cube)

        # Live update loop
        while not self.stop_event.is_set():
            self.env.step()
            time.sleep(0.05)
        self.env.close()

    def slider_callback(self, value, index):
        radians = np.deg2rad(value)
        self.robot.q[index] = radians
        ee = self.robot.fkine(self.robot.q).A
        xyz = np.round(ee[0:3,3], 3)
        rpy = np.round(tr2rpy(ee, unit="deg"), 2)

        # Update labels
        self.labels[0].desc = f"X: {xyz[0]}"
        self.labels[1].desc = f"Y: {xyz[1]}"
        self.labels[2].desc = f"Z: {xyz[2]}"
        self.labels[3].desc = f"Roll: {rpy[0]}"
        self.labels[4].desc = f"Pitch: {rpy[1]}"
        self.labels[5].desc = f"Yaw: {rpy[2]}"

# -------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    robot = create_irb1200()
    tp = TeachPendant(robot)
    tp.run()