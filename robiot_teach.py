import numpy as np
from math import pi
import roboticstoolbox as rtb
from roboticstoolbox import DHLink, DHRobot

# Classical DH (Craig/標準DH) —— Gen3 Lite（m）
a     = [0, 0.28, 0.28, 0, 0.0285, -0.105]
d     = [0.12825, 0.115, 0, 0.02, 0.105, 0.0285]
alpha = [pi/2, pi, pi/2, pi/2, -pi/2, -pi/2]
# 每個關節的角度偏移（θi = qi + offset[i]）
offset= [0.0,  0,0,0,0,0]
qlim = [
    [-2.7, 2.7],
    [-2.7, 2.7],
    [-2.7, 2.7],
    [-2.6, 2.6],
    [-2.6, 2.6],
    [-2.6, 2.6]
]

# 建立 DH links
links = []
for i in range(6):
    links.append(
        DHLink(
            a=a[i],
            d=d[i],
            alpha=alpha[i],
            offset=offset[i],
            qlim=qlim[i]
        )
    )

# 建立機器人模型
gen3lite = DHRobot(links, name="Gen3Lite")
# 初始關節角度 (全部 0)
q0 = [0, 0, 0, 0, 0, 0]



# 設定 workspace 大一點，例如 [-2, 2, -2, 2, -0.5, 2]
gen3lite.teach(
    q0,
    block=True,
    limits=[-0.3, 0.3, -0.3, 0.3, -0.25, 0.3]   # [xmin, xmax, ymin, ymax, zmin, zmax]
)