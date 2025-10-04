#irb1200
import numpy as np
import swift
import roboticstoolbox as rtb
from ir_support.robots.DHRobot3D import DHRobot3D
import spatialmath.base as spb
from spatialmath import SE3
from spatialgeometry import Mesh
import time
import os


class IRB1200(DHRobot3D):
    def __init__(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_stl_path = os.path.join(current_dir, "base.stl")
        self.link1_stl_path = os.path.join(current_dir, "link1.stl")
        self.link2_stl_path = os.path.join(current_dir, "link2.stl")
        self.link3_stl_path = os.path.join(current_dir, "link3.stl")
        self.link4_stl_path = os.path.join(current_dir, "link4.stl")
        self.link5_stl_path = os.path.join(current_dir, "link5.stl")
        self.link6_stl_path = os.path.join(current_dir, "link6.stl")

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
        qtest_transforms = [spb.transl(0,0,0),
                            spb.transl(0,0.146,0) ,
                            spb.transl(0,0.146,-0.13),
                            spb.transl(0,0.39,-0.0378) ,
                            spb.transl(0,0.603,-0.0378) ,
                            spb.transl(0,0.603,-0.1225),
                            spb.transl(0.08535,0.603,-0.1225) ]


        super().__init__(links, link3D_names, name = 'IRB1200', link3d_dir = current_dir, qtest = qtest, qtest_transforms = qtest_transforms)
        self.base = self.base * SE3.Rx(np.pi/2) * SE3.Ry(np.pi/2)

        self.base_mesh = Mesh(filename=self.base_stl_path,color=[0.5, 0.5, 0.5, 1.0],scale=[0.001]*3)

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
        qlim = [
            [-2.7, 2.7],
            [-2.7, 2.7],
            [-2.7, 2.7],
            [-2.6, 2.6],
            [-2.6, 2.6],
            [-2.6, 2.6]
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
        # fig = self.plot(self.q, limits= [-1,1,-1,1,-1,1])
        # fig._add_teach_panel(self, self.q)
        for q in qtraj:
            self.q = q
            env.step(0.02)
            # fig.step(0.01)
        # fig.hold()
        env.hold()
        time.sleep(3)



#------------------------------ GUI ------------------------------
env = swift.Swift()
env.launch(realtime=True)
robot = IRB1200()
robot.add_to_env(env)

robot.base = SE3(2, 2, 0.05)

def slider_callback(value_deg, joint_index):
    q = robot.q.copy()
    q[joint_index] = float(np.deg2rad(float(value_deg)))
    robot.q = q
    env.step(0.02)

for i in range(robot.n):
    env.add(
        swift.Slider(
            cb=lambda v, j=i: slider_callback(v, j),
            min=-180, max=180, step=1,
            value=float(np.rad2deg(float(robot.q[i]))),
            desc=f'Joint {i+1}', unit='°'
        )
    )   


# ---------------------------------------------------------------------------------------#
#if __name__ == "__main__":
#    robot = IRB1200()
#    robot.test()

    
    
    

    