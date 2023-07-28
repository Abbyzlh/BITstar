from BIT_Star import bitstar
import Config_
import pybullet as p
from Bullet.Module_PyBullet_CollisionOnly import PyBullet_CollisionOnly
from Node import Node

simulator = PyBullet_CollisionOnly()
simulator.InitBeforeLoad(client_num=1, client_bias=[[(0.0, 0.0, 0.0), p.getQuaternionFromEuler((0.0, 0.0, 0.0))]])
simulator.load_Body(Config_.Name_Moving)
simulator.load_Map(Config_.Name_Map, Config_.Map_Start, Config_.Map_Goal)
simulator.InitAfterLoad()
planner=bitstar(Node(Config_.Map_Start, gt=0), Node(Config_.Map_Goal), simulator, dim=6, stop_time=20, rbit=5.0)
planner.make_plan()
simulator.Plot_FinalMove("Client_0", Config_.Name_Moving, planner.solution)
