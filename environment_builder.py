import numpy as np
from math import pi
import swift
import spatialgeometry as geometry
from spatialgeometry import Cuboid
from spatialmath import SE3
import os

class EnvironmentBuilder:
    def __init__(self):
        """Initialise Swift environment and load all objects."""
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.env.set_camera_pose([6, 6, 3], [6, 6, 0])

        # Containers for later reference
        self.objects = []

        # Build static environment
        self._add_static_objects()

        # Add pick-and-place areas
        self._add_task_areas()

    # --------------------------------------------------------
    # Static objects (walls, tables, etc.)
    # --------------------------------------------------------
    def _add_static_objects(self):
        """Add all static geometry to the environment."""
        env = self.env

        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_dir = os.path.join(current_dir, "meshes", "env")


        wall1 = geometry.Mesh(os.path.join(env_dir, '3d-model.stl'), pose=SE3(5.7,-0.1, 0)* SE3.Rz(-pi),color = [0.80, 0.78, 0.70, 1],scale=[0.255,0.05,0.052]) 
        wall2 = geometry.Mesh(os.path.join(env_dir, '3d-model.stl'), pose=SE3(0.8, 4.5, 0)* SE3.Rz(pi/2),color = [0.80, 0.78, 0.70, 1],scale=[0.24,0.05,0.052]) 
        wall3 = geometry.Mesh(os.path.join(env_dir, '3d-model.stl'), pose=SE3(5.8, 9.3, 0),color = [0.80, 0.78, 0.70, 1],scale=[0.255,0.05,0.052])     

        table = geometry.Mesh(os.path.join(env_dir,'table.stl'), pose=SE3(2.5, 4.6, 0),color=(0.25, 0.15, 0.08), scale=[2, 1.7, 1.5]) 
        table_3 = geometry.Mesh(os.path.join(env_dir,'table1.stl'), pose=SE3(2.5, 1.3, 0),color=(0.30, 0.18, 0.10), scale=[0.0022, 0.002 ,0.0002]) 
        table2 = geometry.Mesh(os.path.join(env_dir,'neutra_table.stl'), pose=SE3(5.5, -8, 0),color=[0.25, 0.25, 0.25, 1], scale=[0.007, 0.015, 0.0078]) 
        table_area = geometry.Mesh(os.path.join(env_dir,'table.stl'), pose=SE3(3.6, 5.85, 0),color=[0.25, 0.15, 0.08, 1], scale=[0.38, 0.5, 0.6])

        fence1 = geometry.Mesh(os.path.join(env_dir,'double_fence.stl'), pose=SE3(3.8, 0.15, 4.85)*SE3.Rx(pi/2)*SE3.Ry(pi/2), color=[1.0, 0.55, 0.0, 1], scale=[0.0105, 0.01, 0.01])
        fence2 = geometry.Mesh(os.path.join(env_dir,'double_fence.stl'), pose=SE3(3.8, 3.35, 4.85)*SE3.Rx(pi/2)*SE3.Ry(pi/2), color=[1.0, 0.55, 0.0, 1], scale=[0.009, 0.01, 0.01])
        fence3 = geometry.Mesh(os.path.join(env_dir,'double_fence.stl'), pose=SE3(3.8, 6, 4.85)*SE3.Rx(pi/2)*SE3.Ry(pi/2), color=[1.0, 0.55, 0.0, 1], scale=[0.013, 0.01, 0.01])

        wall4 = geometry.Cuboid(scale=[20.3, 9.65, 0.01], color=[0.98, 0.96, 0.88, 1], pose=SE3(11, 10, 4.85)*SE3.Rx(pi/2)*SE3.Ry(pi/2))

        saftey_button1 = geometry.Mesh(os.path.join(env_dir,'stop_button.stl'), pose=SE3(6.75, 0.2, 1.5)* SE3.Rz(-pi)* SE3.Rx(pi/2),color =[0.55, 0.0, 0.0, 1],scale=[0.004,0.004,0.002])
        saftey_button2 = geometry.Mesh(os.path.join(env_dir,'stop_button_base.stl'), pose=SE3(6.75, 0.2, 1.5)* SE3.Rz(-pi)* SE3.Rx(pi/2),color = [0.7, 0.55, 0.0, 1],scale=[0.004,0.004,0.002])

        floor = Cuboid(scale=[10.3, 9.65, 0.01], color=[0.78, 0.86, 0.73, 1])  # 只用 scale 定義大小 
        floor.T = SE3(5.8, 4.6, 0)  # 位置 
        Workingzone = Cuboid(scale=[3.2, 9.5, 0.02], color=[0.78, 0.086, 0.073, 0.5])  # 只用 scale 定義大小 
        Workingzone.T = SE3(2.2, 4.5, 0) 

        plate = geometry.Mesh(os.path.join(env_dir,'plate.stl'),pose=SE3(2, 7.2, 0),color =(0.76, 0.60, 0.42),scale=[2, 1, 1])
        plate2= geometry.Mesh(os.path.join(env_dir,'plate.stl'),pose=SE3(2, 8.5, 0),color =(0.76, 0.60, 0.42),scale=[2, 1, 1])   
        fan = geometry.Mesh(os.path.join(env_dir,'fan.stl'),pose=SE3(1, 2, 4.5)*SE3.Rx(pi/2)*SE3.Ry(pi/2)*SE3.Rz(pi),color =  (0.50, 0.42, 0.35), scale=[0.05, 0.05, 0.01])
        window = geometry.Mesh(os.path.join(env_dir,'window1.stl'),pose=SE3(0, 0, 0)*SE3.Rz(pi/2),color =[np.random.uniform(0.0,0.2), np.random.uniform(0.2,0.5), np.random.uniform(0.5,0.9), np.random.uniform(0.4,0.7)],scale=[0.001, 0.001, 0.001])   
        window_wall = geometry.Mesh(os.path.join(env_dir,'window2.stl'),pose=SE3(5.8, 0.1, 0)*SE3.Rz(pi/2),color = (0.75, 0.75, 0.8), scale=[0.01, 0.15,0.015]) 
        
        control1 = geometry.Mesh(os.path.join(env_dir,'window2.stl'),pose=SE3(5, 0.15, 1.1)*SE3.Rz(pi/2),color = (0.5, 0.8, 1.0, 0.4), scale=[0.02, 0.021,0.01])    
        control2 = geometry.Mesh(os.path.join(env_dir,'window2.stl'),pose=SE3(5, 0.1, 1.1)*SE3.Rz(pi/2),color =  (0.3, 0.3, 0.5,0.8), scale=[0.03, 0.03,0.01]) 
        
        window1 = geometry.Mesh(os.path.join(env_dir,'window2.stl'),pose=SE3(5, 0.15, 2)*SE3.Rz(pi/2),color = (0.75, 0.76, 0.78), scale=[0.01, 0.03,0.03])    
        window2 = geometry.Mesh(os.path.join(env_dir,'window2.stl'),pose=SE3(8, 0.15, 2)*SE3.Rz(pi/2),color = (0.75, 0.76, 0.78), scale=[0.01, 0.03,0.03])    
        small_can = geometry.Mesh(os.path.join(env_dir,'small_can.stl'),pose=SE3(1.8, 1, 0.1)*SE3.Rz(pi),color =(0.13, 0.45, 0.25), scale=[2.3,2, 1])    
        small_can1 = geometry.Mesh(os.path.join(env_dir,'small_can.stl'),pose=SE3(3.2, 1, 0.1)*SE3.Rz(pi),color =(0.9, 0.7, 0.1), scale=[2.3,2, 1])    
        light1= geometry.Mesh(os.path.join(env_dir,'light1.stl'),pose=SE3(3, 4, 4.3),color =(0.65, 0.67, 0.7),scale=[0.009, 0.015, 0.009]) 
        
        light2= geometry.Mesh(os.path.join(env_dir,'light1.stl'),pose=SE3(6, 4, 4.3),color =(0.65, 0.67, 0.7),scale=[0.009, 0.015, 0.009]) 
        box1= geometry.Mesh(os.path.join(env_dir,'box.stl'),pose=SE3(2.2, 5.9, 0),color = (0.45, 0.32, 0.25,0.8),scale=[0.008, 0.008, 0.008])   
        fire1 = geometry.Mesh(os.path.join(env_dir,'firetop.stl'),pose=SE3(7.5, 0.5, 0.7),color = (0.15, 0.15, 0.15), scale=[0.01, 0.01, 0.01])
        fire2 = geometry.Mesh(os.path.join(env_dir,'firebottom.stl'),pose=SE3(7.5, 0.5, 0.7),color = (0.6, 0.05, 0.05), scale=[0.01, 0.01, 0.01])     
        
        trashbag= geometry.Mesh(os.path.join(env_dir,'trashbag.stl'),pose=SE3(1, 9, 0.2),color =(0.35, 0.27, 0.27),scale=[0.6, 0.6, 0.6])   
        boxes = geometry.Mesh(os.path.join(env_dir,'Boxes.stl'),pose=SE3(2, 8.5, 0.1),color = (0.74, 0.56, 0.34), scale=[1,1, 1])   
        button1= geometry.Cylinder(radius=0.05, length=0.07, color=[1.0, 0.55, 0.0, 1]) 
        button2= geometry.Cylinder(radius=0.05, length=0.07, color=(0.6, 0.05, 0.05))  
        button1.T=SE3(5.8, 0.3, 1.55)*SE3.Rx(pi/2)
        button2.T=SE3(5.8, 0.3, 1.35)*SE3.Rx(pi/2)
        
        shutter= geometry.Mesh(os.path.join(env_dir, 'shutter.stl'),pose=SE3(1, 8, 0)*SE3.Rx(pi/2)*SE3.Ry(pi/2),color =(0.55, 0.47, 0.40, 0.9),scale=[0.001, 0.0015, 0.003])   
        shelf = geometry.Mesh(os.path.join(env_dir, 'shelf.stl'),pose=SE3(9, 0.5, 0.5)*SE3.Rz(pi/2),color =(0.38, 0.26, 0.18, 0.9), scale=[0.6,0.6, 0.8])    

        for env_obj in [wall1, wall2, wall3, wall4, table, table_3, table2, table_area,
                    fence1, fence2, fence3, saftey_button1, saftey_button2,
                    floor, Workingzone, plate, plate2, fan,
                    window_wall, control1, control2, window, window1, window2,
                    shutter, small_can, small_can1, light1, light2,shelf,
                    box1, fire1, fire2, trashbag, boxes]:
            env.add(env_obj)
            self.objects.append(env_obj)


    # --------------------------------------------------------
    # Task areas (pick zone, bin zone)
    # --------------------------------------------------------
    def _add_task_areas(self):
        """Add work areas for pick and place."""

        #self.robot2_base = SE3(2.7, 5.6, 0.25)

        self.area_pose = SE3(3.6, 5.85, 0.2)      
        self.area_place_pose = SE3(2.7, 4.9, 0.48)     
        self.area_box_pose = SE3(2.2, 5.9, 0.3)

        self.area = Cuboid(scale=[0.5, 0.5, 0.01], color=(0.45, 0.30, 0.20, 0.0000000000001), pose=self.area_pose)
        self.area_place = Cuboid(scale=[0.8, 0.5, 0.01], color=[0.6, 0.6, 0.6, 0.08], pose=self.area_place_pose)
        self.area_box = Cuboid(scale=[0.3, 0.3, 0.01], color=[0.5, 0.6, 0, 0.0000000000000000001], pose=self.area_box_pose)

        self.env.add(self.area)
        self.env.add(self.area_place)
        self.env.add(self.area_box)
