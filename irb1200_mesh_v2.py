#irb1200
import numpy as np
import swift
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
from spatialmath import SE3
import spatialmath.base as spb
from spatialgeometry import Cuboid, Cylinder
import time
import os


class IRB1200(DHRobot3D):
    def __init__(self):

        self.ee = IRB1200EE()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        mesh_dir = os.path.join(current_dir, "meshes", "robot3")


        # DH links
        links = self._create_DH()

        link3D_names = dict(link0 = 'base_scaled',      # color option only takes effect for stl file
                            link1 = 'link1_scaled',
                            link2 = 'link2_scaled',
                            link3 = 'link3_scaled',
                            link4 = 'link4_scaled',
                            link5 = 'link5_scaled',
                            link6 = 'link6_scaled')
        
        # A joint config and the 3D object transforms to match that config
        qtest = [0,0,0,0,0,0]
        qtest_transforms = [spb.transl(0.00 , 0 , 0.00 ) ,                                               #base
                            spb.transl(0.00 , 0 , 0.20 ) ,                                               #link1
                            spb.transl(0.00 , 0 , 0.40 ) @ spb.troty(-np.pi/2) @ spb.trotx(np.pi/2) ,    #link2
                            spb.transl(0.00 , 0 , 0.75 ) @ spb.trotx(np.pi/2) ,                          #link3
                            spb.transl(0.17 , 0 , 0.79 ) @ spb.troty(np.pi/2) ,                          #link4
                            spb.transl(0.35 , 0 , 0.79 ) @ spb.trotx(np.pi/2) ,                          #link5
                            spb.transl(0.44, 0 , 0.792 ) @ spb.troty(np.pi/2)                            #link6
                            ]


        super().__init__(links, link3D_names, name = 'IRB1200', link3d_dir = mesh_dir, qtest = qtest, qtest_transforms = qtest_transforms)
        
        colors = [
            (0.98, 0.82, 0.88, 1),  
            (0.99, 0.85, 0.90, 1),   
            (0.97, 0.75, 0.85, 1),  
            (0.96, 0.70, 0.82, 1),   
            (0.99, 0.83, 0.88, 0.9), 
            (0.98, 0.78, 0.87, 1),  
            (0.99, 0.86, 0.92, 1)    
]

        for i, mesh in enumerate(self.links_3d):
            mesh.color = tuple(float(c) for c in colors[i])

        self.q = qtest

    def _create_DH(self):
        """
        Create robot's standard DH model
        """
        links = []
        a = [0.0, 350e-3, 42e-3, 0.0, 0.0, 0.0]
        d = [399e-3, 0.0, 0.0, 351e-3, 0.0, 82e-3]     
        alpha = [-np.pi/2, 0.0, -np.pi/2, np.pi/2, -np.pi/2, 0]
        offset = [ 0.0, -np.pi/2, 0.0, 0.0, 0.0, -np.pi ]
        d2r = np.pi / 180       # deg 2 rad helper
        qlim = [
            [-170 * d2r, 170 * d2r],
            [-100 * d2r, 80 * d2r],
            [-150 * d2r, 70  * d2r],
            [-270 * d2r, 270 * d2r],
            [-130 * d2r, 130 * d2r],
            [-360 * d2r, 360 * d2r]
        ]
        
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], offset=offset[i], qlim=qlim[i])
            links.append(link)
        return links
    

    # -----------------------------------------------------------------------------------#
    def test(self):
        """
        Test the class by adding 3d objects into a new Swift window and do a simple movement
        """
        env = swift.Swift()
        env.launch(realtime= True)
        self.q = self._qtest
        self.add_to_env(env)

        q_goal = [self.q[i]-np.pi/3 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q

        for q in qtraj:
            self.q = q
            env.step(0.02)

        env.hold()
        time.sleep(3)

#IRB1200 End effector
class IRB1200EE:            
    def __init__(self, cyl_radius=0.04, cyl_length=0.10,
                 plate_size=0.30, plate_thickness=0.11,
                 color_cyl=[0.96, 0.70, 0.82, 1],
                 color_plate=[0.3, 0.3, 0.3, 1]):
        # --- geometry parts ---
        self.cyl = Cylinder(radius=cyl_radius, length=(cyl_length), color=color_cyl)
        self.plate = Cuboid(scale=(plate_size, (plate_size + 0.08), plate_thickness), color=color_plate)

        # --- stored params ---
        self.cyl_radius = cyl_radius
        self.cyl_length = cyl_length
        self.plate_size = plate_size
        self.plate_thickness = plate_thickness

    # --------------------------------------------------------
    def add_to_env(self, env):
        """Add both parts to Swift environment."""
        env.add(self.cyl)
        env.add(self.plate)

    # --------------------------------------------------------
    def attach_to_robot(self, robot):
        """
        Update transforms of crusher parts based on robot pose.
        Faces downward (-Z) with slight rotation offset to align visually.
        """
        T = robot.fkine(robot.q)
        R_down = SE3.Rz(np.pi/2) * SE3.Rx(np.pi)

        # Center the cylinder under the flange
        self.cyl.T = T * R_down * SE3(0, 0, -self.cyl_length / 2)

        # Attach the plate directly under the cylinder
        plate_offset_z = -(self.cyl_length + self.plate_thickness / 2)
        self.plate.T = T * R_down * SE3(0, 0, plate_offset_z)


    



