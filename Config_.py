# -*- coding: utf-8 -*-

import numpy as np

IF_GUI = True
GUI_Width_Debug = 1280
GUI_Height_Debug = 720

# Map
Map_DimRange = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-np.pi/2, np.pi/2], [-np.pi, np.pi], [-np.pi, np.pi]]  # XYZ Euler
Map_ObstacleCount = 100
Map_ObstacleMinSize = 0.1
Map_ObstacleMaxSize = 1.0
Map_Start = (0.0, 0.0, 0.0, np.pi*0.1, np.pi*0.2, np.pi*0.3)  # 起点
Map_StartBlockZone = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-np.pi*1.0, np.pi*1.0], [-np.pi*1.0, np.pi*1.0], [-np.pi*1.0, np.pi*1.0]]
Map_Goal = (4.0, 4.0, 4.0, np.pi*0.4, np.pi*0.5, np.pi*0.6)  # 终点
Map_GoalBlockZone = [[3.0, 5.0], [3.0, 5.0], [3.0, 5.0], [-np.pi*1.0, np.pi*1.0], [-np.pi*1.0, np.pi*1.0], [-np.pi*1.0, np.pi*1.0]]
# Body
Body_Size = [0.5, 0.5, 0.5]

# Check
Check_MinSize = 0.01

#
UpdateFS = 240.0

#
Name_Map = "Random"
Name_Moving = "Cube"
# Name_Moving = r"E:\WorkSpace\XBB\Program\PyBullet_CollisionDetection_4.0.00\Model\PUMP.obj"

