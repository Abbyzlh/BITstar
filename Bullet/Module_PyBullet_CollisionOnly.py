import copy
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BITstar6D.Node import Node
import os

import Config_

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
plt.ion()


class PyBullet_CollisionOnly:
    """
    打包一个PyBullet的仿真环境，Id_.*代表对应的索引号，Info_.*代表整个系统的描绘信息，State_.*代表整个系统的状态信息，Control_.*代表外界输入的控制量
    """
    # Client p_Root.func==p.func(physicsClientId=Id_Root)
    p_Root: bc.BulletClient  # 默认的基础Client,不指名id_client的都会给到p_Root，
    p_Global: {str: bc.BulletClient}  # 通过设置多个Client来对于系统中的东西分组管理，例如机械臂系统ABCD，但是目前不提供这个，统一管理在一个p_Physics下。目前只实现GUI和动力学的分离
    # Id
    Id_Client_Root: int  # p_GUI的physicsClientId,一般来说，不显式指定的physicsClientId，采用0，也就是p_GUI
    Id_Client_Global: {str: int}  # 通过Client名称索引Client的Id。除了Root以外的所有physicsClientId
    Id_Body_Global: {str: {str: int}}  # 通过Client名称与Body名称索引Body的Id
    Id_Joint_Global: {str: {str: {str: int}}}  # 通过Client名称与Body名称与Joint名称索引Joint的Id
    # State
    State_Collision_Global: {str: {str: {str: pd.DataFrame}}}  # 通过Client名称与BodyA与BodyB的名称对索引Body，然后给出DataFrame格式的ContactPoints
    State_Solver_Global: {}  # 由stepSimulation返回的求解状态
    # Control
    Control_RealTimeBase: {str: {str: (tuple, tuple)}}  # 实时的控制Body的Base的控制值。通过Client名称与Body名称索引Base的target值。包括位置和位姿
    # Plot
    Id_Vertex: {str: {Node: int}}  # 通过Client名称和Node来索引的Vertex的Id
    Id_Edge: {str: {(Node, Node): int}}  # 通过Client名称和Node对（从Parent到Child）来索引的Edge的Id
    Id_Finish_Line: {str: {(Node, Node): int}}  # 通过Client名称和Node对（从Parent到Chile）来索引的DebugLine的Id
    Id_Finish_Coordinate: {str: {Node: int}}  # 通过Client名称和Node来索引的DebugLine的Id

    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def InitBeforeLoad(self, client_num, client_bias):
        """
        在加载模型前初始化，初始化多个Client，以及他们的错位bias

        :param client_num:
        :param client_bias:
        :return:
        """
        client_list = ['Client_{}'.format(ii) for ii in range(client_num)]
        self.client_bias = client_bias
        self.Init_World(client_list)
        self.Id_Body_Global = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}

    def Init_World(self, client_list: [str], *args, **kwargs):
        """
        初始化物理引擎的参数，并且分割出多个Client

        :param client_list: 有哪些client
        :param args:
        :param kwargs: 直接作用于 p.setPhysicsEngineParameter
        :return:
        """
        # 默认值
        if len(kwargs) == 0:
            kwargs = dict(
                numSolverIterations=100,  # 默认50
                reportSolverAnalytics=True,
                solverResidualThreshold=0.001,  # 默认1e-7
            )

        # # 创建物理引擎实例并连接到仿真
        # Root
        if Config_.IF_GUI:
            self.p_Root = bc.BulletClient(p.GUI_SERVER, options='--width={} --height={}'.format(Config_.GUI_Width_Debug, Config_.GUI_Height_Debug))
        else:
            self.p_Root = bc.BulletClient(p.SHARED_MEMORY_SERVER)
        self.Id_Client_Root = self.p_Root._client
        # Path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加PyBullet自带的系统文件夹，包括一些常用的形状
        # Client
        self.Id_Client_Global = {}
        self.p_Global = {}
        for ii, Name_Client in enumerate(client_list):
            p_ = bc.BulletClient(p.SHARED_MEMORY)
            self.p_Global[Name_Client] = p_
            self.Id_Client_Global[Name_Client] = p_._client
        # GUI
        self.p_Root.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        self.p_Root.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        self.p_Root.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.p_Root.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.p_Root.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.p_Root.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        # self.p_Root.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # Physics
        # # 设置仿真环境并运行仿真
        self.p_Root.setGravity(0, 0, -9.8,)
        self.p_Root.setPhysicsEngineParameter(**kwargs)

    def load_Map(self, map_name, start, goal):
        """
        Load地图

        :param map_name: Random代表随机生成的地图，
        :param start: 起始位置，通过Map_StartBlockZone避免起始位置碰撞
        :param goal: 目标起始位置，通过Map_GoalBlockZone避免目标位置碰撞
        :return:
        """
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            basePosition, baseOrientation = self.client_bias[ii]
            if map_name == 'Random':
                baseMass = 0.0
                baseCollisionShapeIndex = -1
                baseVisualShapeIndex = -1
                linkMasses = [0.0 for ii in range(Config_.Map_ObstacleCount)]
                linkCollisionShapeIndices = [-1 for ii in range(Config_.Map_ObstacleCount)]
                linkVisualShapeIndices = [-1 for ii in range(Config_.Map_ObstacleCount)]
                linkPositions = [[0.0, 0.0, 0.0] for ii in range(Config_.Map_ObstacleCount)]
                linkOrientations = [p.getQuaternionFromEuler([0.0, 0.0, 0.0]) for ii in range(Config_.Map_ObstacleCount)]
                linkInertialFramePositions = [[0.0, 0.0, 0.0] for ii in range(Config_.Map_ObstacleCount)]
                linkInertialFrameOrientations = [p.getQuaternionFromEuler([0.0, 0.0, 0.0]) for ii in range(Config_.Map_ObstacleCount)]
                linkParentIndices = [0 for ii in range(Config_.Map_ObstacleCount)]
                linkJointTypes = [p.JOINT_FIXED for ii in range(Config_.Map_ObstacleCount)]
                linkJointAxis = [[0.0, 0.0, 0.0] for ii in range(Config_.Map_ObstacleCount)]
                for ii in range(Config_.Map_ObstacleCount):
                    linkMasses[ii] = 0.0
                    halfExtents = [np.random.uniform(Config_.Map_ObstacleMinSize, Config_.Map_ObstacleMaxSize) for ii in range(3)]
                    linkCollisionShapeIndices[ii] = p.createCollisionShape(
                        p.GEOM_BOX,
                        halfExtents=halfExtents,
                        physicsClientId=Id_Client,
                    )
                    linkVisualShapeIndices[ii] = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=halfExtents,
                        rgbaColor=[1.0, 1.0, 1.0, 0.5],
                        physicsClientId=Id_Client,
                    )
                    # linkPositions[ii] = [np.random.uniform(dim[0], dim[1]) for dim in Config_.Map_DimRange[0:3]]
                    # linkOrientations[ii] = p.getQuaternionFromEuler([np.random.uniform(dim[0], dim[1]) for dim in Config_.Map_DimRange[3:6]])
                    flag_accessible = True
                    while flag_accessible:
                        pos = [np.random.uniform(dim[0], dim[1]) for dim in Config_.Map_DimRange[0:3]]
                        orn = [np.random.uniform(dim[0], dim[1]) for dim in Config_.Map_DimRange[3:6]]
                        flag_pos = all([(dim[0] < pos[ii]) and (pos[ii] < dim[1]) for ii, dim in enumerate(Config_.Map_StartBlockZone[0:3])]) or all([(dim[0] < pos[ii]) and (pos[ii] < dim[1]) for ii, dim in enumerate(Config_.Map_GoalBlockZone[0:3])])
                        flag_orn = all([(dim[0] < orn[ii]) and (orn[ii] < dim[1]) for ii, dim in enumerate(Config_.Map_StartBlockZone[3:6])]) or all([(dim[0] < orn[ii]) and (orn[ii] < dim[1]) for ii, dim in enumerate(Config_.Map_GoalBlockZone[3:6])])
                        flag_accessible = flag_pos and flag_orn
                    linkPositions[ii] = pos
                    linkOrientations[ii] = p.getQuaternionFromEuler(orn)
                    linkInertialFramePositions[ii] = [0.0, 0.0, 0.0]
                    linkInertialFrameOrientations[ii] = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
                    linkParentIndices[ii] = 0
                    linkJointTypes[ii] = p.JOINT_FIXED
                    linkJointAxis[ii] = [0.0, 0.0, 0.0]
                Id_Map = p.createMultiBody(
                    physicsClientId=Id_Client,
                    baseMass=baseMass,
                    baseCollisionShapeIndex=baseCollisionShapeIndex,
                    baseVisualShapeIndex=baseVisualShapeIndex,
                    basePosition=basePosition,
                    baseOrientation=baseOrientation,
                    linkMasses=linkMasses,
                    linkCollisionShapeIndices=linkCollisionShapeIndices,
                    linkVisualShapeIndices=linkCollisionShapeIndices,
                    linkPositions=linkPositions,
                    linkOrientations=linkOrientations,
                    linkInertialFramePositions=linkInertialFramePositions,
                    linkInertialFrameOrientations=linkInertialFrameOrientations,
                    linkParentIndices=linkParentIndices,
                    linkJointTypes=linkJointTypes,
                    linkJointAxis=linkJointAxis,
                )
                gemo_Collision_Start = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=Config_.Body_Size,
                    physicsClientId=Id_Client
                )
                gemo_Visual_Start = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=Config_.Body_Size,
                    rgbaColor=[0.0, 1.0, 1.0, 1.0],
                    physicsClientId=Id_Client,
                )
                Id_Start = p.createMultiBody(
                    physicsClientId=Id_Client,
                    baseMass=0.0,
                    baseCollisionShapeIndex=gemo_Collision_Start,
                    baseVisualShapeIndex=gemo_Visual_Start,
                    basePosition=start[0:3],
                    baseOrientation=p.getQuaternionFromEuler(start[3:6]),
                )
                gemo_Collision_Goal = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=Config_.Body_Size,
                    physicsClientId=Id_Client
                )
                gemo_Visual_Goal = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=Config_.Body_Size,
                    rgbaColor=[1.0, 0.0, 1.0, 1.0],
                    physicsClientId=Id_Client
                )
                Id_Goal = p.createMultiBody(
                    physicsClientId=Id_Client,
                    baseMass=0.0,
                    baseCollisionShapeIndex=gemo_Collision_Goal,
                    baseVisualShapeIndex=gemo_Visual_Goal,
                    basePosition=goal[0:3],
                    baseOrientation=p.getQuaternionFromEuler(goal[3:6])
                )
            self.Id_Body_Global[Name_Client][map_name] = Id_Map
            self.Id_Body_Global[Name_Client]["Start"] = Id_Start
            self.Id_Body_Global[Name_Client]["Goal"] = Id_Goal

    def load_Body(self, Name_Body):
        """
        导入移动的物体，可以是Cube，也可以是obj的路径，会自动把obj进行vhacd

        :param Name_Body:
        :return:
        """
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            basePosition, baseOrientation = self.client_bias[ii]
            if Name_Body == 'Cube':
                geom_Collision = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=Config_.Body_Size,
                    physicsClientId=Id_Client,
                )
                gemo_Visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=Config_.Body_Size,
                    rgbaColor=[0.0, 0.0, 1.0, 1.0],
                    physicsClientId=Id_Client,
                )
                Id_Body = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=geom_Collision,
                    baseVisualShapeIndex=gemo_Visual,
                    basePosition=basePosition,
                    baseOrientation=baseOrientation,
                    physicsClientId=Id_Client,
                )
            else:
                obj_path = Name_Body
                obj_vhacd_path = obj_path.replace('.obj', '_vhacd.obj')
                obj_log_path = obj_path.replace('.obj', '.txt')
                if not os.path.exists(obj_vhacd_path):
                    p.vhacd(
                        fileNameIn=obj_path, fileNameOut=obj_vhacd_path, fileNameLogging=obj_log_path,
                        # concavity=0.00001, alpha=0.001, beta=0.001, gamma=0.00001,
                        # resolution=16000000, minVolumePerCH=0.0000001, maxNumVerticesPerCH=256,
                    )
                geom_Collision = p.createCollisionShape(
                    p.GEOM_MESH,
                    fileName=obj_vhacd_path,
                    physicsClientId=Id_Client,
                    meshScale=[0.001, 0.001, 0.001],
                )
                gemo_Visual = p.createVisualShape(
                    p.GEOM_MESH,
                    fileName=obj_vhacd_path,
                    rgbaColor=[0.0, 0.0, 1.0, 1.0],
                    physicsClientId=Id_Client,
                    meshScale=[0.001, 0.001, 0.001],
                )
                Id_Body = p.createMultiBody(
                    baseMass=1.0,
                    baseCollisionShapeIndex=geom_Collision,
                    baseVisualShapeIndex=gemo_Visual,
                    basePosition=basePosition,
                    baseOrientation=baseOrientation,
                    physicsClientId=Id_Client,
                )
            self.Id_Body_Global[Name_Client][Name_Body] = Id_Body

    def InitAfterLoad(self):
        self.Init_Model()
        self.Init_Control()
        self.Init_State()
        self.Init_Plot()

    def Init_Model(self, *args, **kwargs):
        """
        ！可以不看！初始化模型信息

        :param args:
        :param kwargs:
        :return:
        """
        # 模型关节
        self.Id_Joint_Global = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            self.Id_Joint_Global[Name_Client] = {}
            for jj, (Name_Body, Id_Body) in enumerate(self.Id_Body_Global[Name_Client].items()):
                self.Id_Joint_Global[Name_Client][Name_Body] = {}
                for kk in range(p.getNumJoints(Id_Body, self.Id_Client_Global[Name_Client])):
                    JointInfo = p.getJointInfo(
                        physicsClientId=self.Id_Client_Global[Name_Client],
                        bodyUniqueId=Id_Body,
                        jointIndex=kk,
                    )
                    self.Id_Joint_Global[Name_Client][Name_Body][JointInfo[1].decode('UTF-8')] = JointInfo[0]

    def Init_Control(self, *args, **kwargs):
        """
        ！可以不看！初始化控制Base的值

        :param args:
        :param kwargs:
        :return:
        """
        # Base驱动
        self.Control_RealTimeBase = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            for jj, (Name_Body, Id_Body) in enumerate(self.Id_Body_Global[Name_Client].items()):
                self.Control_RealTimeBase[Name_Client][Name_Body] = p.getBasePositionAndOrientation(
                    physicsClientId=Id_Client,
                    bodyUniqueId=Id_Body,
                )

    def Init_State(self, *args, **kwargs):
        """
        初始化一系列对于整个系统的状态信息，包括

        :param args:
        :param kwargs:
        :return:
        """
        # CollisionState
        # 初始化
        self.State_Collision_Global = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            self.State_Collision_Global[Name_Client] = {}
            for jj, (Name_BodyA, Id_BodyA) in enumerate(self.Id_Body_Global[Name_Client].items()):
                self.State_Collision_Global[Name_Client][Name_BodyA] = {}
                for kk, (Name_BodyB, Id_BodyB) in enumerate(self.Id_Body_Global[Name_Client].items()):
                    if Name_BodyA != Name_BodyB:
                        self.State_Collision_Global[Name_Client][Name_BodyA][Name_BodyB] = pd.DataFrame(
                            columns=[
                                'contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB', 'linkIndexA', 'linkIndexB', 'positionOnA', 'positionOnB',
                                'contactNormalOnB', 'contactDistance', 'normalForce',
                                'lateralFriction1', 'lateralFrictionDir1', 'lateralFriction2', 'lateralFrictionDir2',
                            ],
                        )
        # 更新
        self.Update_CollisionState()  # 更新

        # Solver
        self.State_Solver_Global = ()

    # @Decorator_Timer(logger=logger, level='DEBUG')
    def Update_State(self):
        """
        ！可以不看！

        :return:
        """
        self.Update_CollisionState()

    def Update_CollisionState(self):
        """
        ！可以不看！更新所有的State_Collision_Global以保持最新。

        :return:
        """
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            for jj, (Name_BodyA, Id_BodyA) in enumerate(self.Id_Body_Global[Name_Client].items()):
                for kk, (Name_BodyB, Id_BodyB) in enumerate(self.Id_Body_Global[Name_Client].items()):
                    if Name_BodyA != Name_BodyB:
                        self.State_Collision_Global[Name_Client][Name_BodyA][Name_BodyB] = pd.DataFrame(
                            p.getContactPoints(
                                physicsClientId=Id_Client,
                                bodyA=Id_BodyA,
                                bodyB=Id_BodyB,
                            ),
                            columns=self.State_Collision_Global[Name_Client][Name_BodyA][Name_BodyB].columns
                        )
                        if self.State_Collision_Global[Name_Client][Name_BodyA][Name_BodyB].shape[0] > 0:
                            print(Name_Client, Name_BodyA, Name_BodyB)

    # @Decorator_Timer(logger=logger, level='DEBUG')
    def simulate(self):
        """
        ！可以不看！

        :return:
        """
        self.State_Solver_Global = self.p_Root.performCollisionDetection()
        # Control
        self.simulate_Control()

    def simulate_Control(self):
        """
        ！可以不看！

        :return:
        """
        # Base驱动
        for ii, (Name_Client, Id_Client) in enumerate(self.Id_Client_Global.items()):
            for jj, (Name_Body, Id_Body) in enumerate(self.Id_Body_Global[Name_Client].items()):
                pos = self.Control_RealTimeBase[Name_Client][Name_Body][0]
                orn = self.Control_RealTimeBase[Name_Client][Name_Body][1]
                # 控制
                p.resetBasePositionAndOrientation(
                    physicsClientId=Id_Client,
                    bodyUniqueId=Id_Body,
                    posObj=pos,
                    ornObj=orn,
                )
                # 绘制坐标系
                if Name_Body not in self.Id_Vertex[Name_Client].keys():
                    self.Id_Vertex[Name_Client][Name_Body] = self.Plot_AddCoordinate(
                        Name_Client=Name_Client,
                        pos=pos,
                        orn=orn,
                    )
                else:
                    self.Id_Vertex[Name_Client][Name_Body] = self.Plot_ReplaceCoordinate(
                        Name_Client=Name_Client,
                        pos=pos,
                        orn=orn,
                        replaceItemUniqueId=self.Id_Vertex[Name_Client][Name_Body],
                    )
                # 绘制连线
                if Name_Body not in self.Id_Edge[Name_Client].keys():
                    self.Id_Edge[Name_Client][Name_Body] = self.Plot_AddLine(
                        Name_Client=Name_Client,
                        lineFromXYZ=[0.0, 0.0, 0.0],
                        lineToXYZ=pos,
                    )
                else:
                    self.Id_Edge[Name_Client][Name_Body] = self.Plot_ReplaceLine(
                        Name_Client=Name_Client,
                        lineFromXYZ=[0.0, 0.0, 0.0],
                        lineToXYZ=pos,
                        replaceItemUniqueId=self.Id_Edge[Name_Client][Name_Body],
                    )

    def Check_Collision_Point(self, Name_Client, Name_BodyA, pos, orn):
        # 修改位置
        Id_Client = self.Id_Client_Global[Name_Client]
        Id_BodyA = self.Id_Body_Global[Name_Client][Name_BodyA]
        p.resetBasePositionAndOrientation(
            physicsClientId=Id_Client,
            bodyUniqueId=Id_BodyA,
            posObj=pos,
            ornObj=orn,
        )
        # 计算碰撞
        self.p_Root.performCollisionDetection()
        State_Collision_Check = {}
        for ii, (Name_BodyB, Id_BodyB) in enumerate(self.Id_Body_Global[Name_Client].items()):
            if (Name_BodyB != Name_BodyA) and (Name_BodyB != "Start") and (Name_BodyB != "Goal"):
                State_Collision_Check[Name_BodyB] = pd.DataFrame(
                    p.getContactPoints(
                        physicsClientId=Id_Client,
                        bodyA=Id_BodyA,
                        bodyB=Id_BodyB,
                    ),
                    columns=self.State_Collision_Global[Name_Client][Name_BodyA][Name_BodyB].columns
                )
                if State_Collision_Check[Name_BodyB].shape[0] > 0:
                    return True
        return False

    def Check_Collision_Line(self, Name_Client, Name_BodyA, pos_start, orn_start, pos_end, orn_end):
        # 修改位置
        NN = 10
        for ii in range(NN):
            # TODO 这里有点问题，orn不能这么弄，但是先这样吧
            pos = (np.array(pos_end) - np.array(pos_start)) / NN * ii + np.array(pos_start)
            orn = (np.array(orn_end) - np.array(orn_start)) / NN * ii + np.array(orn_start)
            if_collision = self.Check_Collision_Point(
                Name_Client=Name_Client,
                Name_BodyA=Name_BodyA,
                pos=pos,
                orn=orn,
            )
            if if_collision:
                return True
        return False

    def Init_Plot(self, *args, **kwargs):
        """
        一些绘制的可视化用的东西

        :param args:
        :param kwargs:
        :return:
        """
        self.Id_Vertex = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}
        self.Id_Edge = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}
        self.Id_Finish_Coordinate = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}
        self.Id_Finish_Line = {Name_Client: {} for Name_Client in self.Id_Client_Global.keys()}

    def Plot_ReplaceCoordinate(self, Name_Client, pos, orn, replaceItemUniqueId, lineWidth=2.0, lineLength=1.0):
        Id_dict = {'X': None, 'Y': None, 'Z': None}
        # RGB XYZ
        Id_dict['X'] = self.p_Root.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=pos,
            lineToXYZ=np.array(pos) + np.matmul(np.array(p.getMatrixFromQuaternion(orn)).reshape([3,3]), np.array([lineLength, 0.0, 0.0])),
            lineColorRGB=[1.0, 0.0, 0.0],
            lineWidth=lineWidth,
            lifeTime=0,
            replaceItemUniqueId=replaceItemUniqueId['X'],
        )
        Id_dict['Y'] = self.p_Root.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=pos,
            lineToXYZ=np.array(pos) + np.matmul(np.array(p.getMatrixFromQuaternion(orn)).reshape([3,3]), np.array([0.0, lineLength, 0.0])),
            lineColorRGB=[0.0, 1.0, 0.0],
            lineWidth=lineWidth,
            lifeTime=0,
            replaceItemUniqueId=replaceItemUniqueId['Y'],
        )
        Id_dict['Z'] = self.p_Root.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=pos,
            lineToXYZ=np.array(pos) + np.matmul(np.array(p.getMatrixFromQuaternion(orn)).reshape([3,3]), np.array([0.0, 0.0, lineLength])),
            lineColorRGB=[0.0, 0.0, 1.0],
            lineWidth=lineWidth,
            lifeTime=0,
            replaceItemUniqueId=replaceItemUniqueId['Z'],
        )
        return Id_dict

    def Plot_AddCoordinate(self, Name_Client, pos, orn, lineWidth=2.0, lineLength=1.0):
        Id_dict = {'X': None, 'Y': None, 'Z': None}
        # RGB XYZ
        Id_dict['X'] = p.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=pos,
            lineToXYZ=np.array(pos) + np.matmul(np.array(p.getMatrixFromQuaternion(orn)).reshape([3,3]), np.array([lineLength, 0.0, 0.0])),
            lineColorRGB=[1.0, 0.0, 0.0],
            lineWidth=lineWidth,
            lifeTime=0,
        )
        Id_dict['Y'] = p.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=pos,
            lineToXYZ=np.array(pos) + np.matmul(np.array(p.getMatrixFromQuaternion(orn)).reshape([3,3]), np.array([0.0, lineLength, 0.0])),
            lineColorRGB=[0.0, 1.0, 0.0],
            lineWidth=lineWidth,
            lifeTime=0,
        )
        Id_dict['Z'] = p.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=pos,
            lineToXYZ=np.array(pos) + np.matmul(np.array(p.getMatrixFromQuaternion(orn)).reshape([3,3]), np.array([0.0, 0.0, lineLength])),
            lineColorRGB=[0.0, 0.0, 1.0],
            lineWidth=lineWidth,
            lifeTime=0,
        )
        return Id_dict

    def Plot_DeleteCoordinate(self, Name_Client, Id_dict):
        p.removeUserDebugItem(
            physicsClientId=self.Id_Client_Global[Name_Client],
            itemUniqueId=Id_dict['X'],
        )
        p.removeUserDebugItem(
            physicsClientId=self.Id_Client_Global[Name_Client],
            itemUniqueId=Id_dict['Y'],
        )
        p.removeUserDebugItem(
            physicsClientId=self.Id_Client_Global[Name_Client],
            itemUniqueId=Id_dict['Z'],
        )

    def Plot_ReplaceLine(self, Name_Client, lineFromXYZ, lineToXYZ, replaceItemUniqueId, lineColorRGB=None, lineWidth=0.1):
        lineColorRGB = [0.0, 0.0, 0.0] if lineColorRGB is None else lineColorRGB
        Id = p.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=lineFromXYZ,
            lineToXYZ=lineToXYZ,
            lineColorRGB=lineColorRGB,
            lineWidth=lineWidth,
            lifeTime=0,
            replaceItemUniqueId=replaceItemUniqueId,
        )
        return Id

    def Plot_AddLine(self, Name_Client, lineFromXYZ, lineToXYZ, lineColorRGB=None, lineWidth=0.1):
        lineColorRGB = [0.0, 0.0, 0.0] if lineColorRGB is None else lineColorRGB
        Id = p.addUserDebugLine(
            physicsClientId=self.Id_Client_Global[Name_Client],
            lineFromXYZ=lineFromXYZ,
            lineToXYZ=lineToXYZ,
            lineColorRGB=lineColorRGB,
            lineWidth=lineWidth,
            lifeTime=0,
        )
        return Id

    def Plot_DeleteLine(self, Name_Client, Id):
        p.removeUserDebugItem(
            physicsClientId=self.Id_Client_Global[Name_Client],
            itemUniqueId=Id,
        )

    def Plot_AddVertex(self, Name_Client, node: Node):
        Id_dict = self.Plot_AddCoordinate(
            Name_Client=Name_Client,
            pos=node.np_arr[0:3],
            orn=p.getQuaternionFromEuler(node.np_arr[3:6]),
            lineWidth=0.5,
        )
        self.Id_Vertex[Name_Client][node] = Id_dict

    def Plot_DeleteVertex(self, Name_Client, node: Node):
        if node in self.Id_Vertex[Name_Client].keys():
            self.Plot_DeleteCoordinate(
                Name_Client=Name_Client,
                Id_dict=self.Id_Vertex[Name_Client][node],
            )
            del self.Id_Vertex[Name_Client][node]

    def Plot_AddEdge(self, Name_Client, node_parent: Node, node_child: Node):
        Id = self.Plot_AddLine(Name_Client, node_parent.np_arr[0:3], node_child.np_arr[0:3])
        self.Id_Edge[Name_Client][(node_parent, node_child)] = Id

    def Plot_DeleteEdge(self, Name_Client, node_parent: Node, node_child: Node):
        if (node_parent, node_child) in self.Id_Edge[Name_Client].keys():
            self.Plot_DeleteLine(Name_Client, self.Id_Edge[Name_Client][(node_parent, node_child)])
            del self.Id_Edge[Name_Client][(node_parent, node_child)]

    def Plot_FinishLine(self, Name_Client, planner):
        # 删除旧的
        # 连线
        for ii, ((node_parent, node_child), Id) in enumerate(self.Id_Finish_Line[Name_Client].items()):
            self.Plot_DeleteLine(
                Name_Client=Name_Client,
                Id=Id,
            )
        # 坐标系
        for ii, (node_parent, Id_dict) in enumerate(self.Id_Finish_Coordinate[Name_Client].items()):
            self.Plot_DeleteCoordinate(
                Name_Client=Name_Client,
                Id_dict=Id_dict,
            )
        # 绘制新的
        for ii, node_parent in enumerate(planner.path[:-1]):
            node_child = planner.path[ii+1]
            # 连线
            # 删除Edge的可视化
            self.Plot_DeleteEdge(
                Name_Client=Name_Client,
                node_parent=node_parent,
                node_child=node_child,
            )
            # 绘制加粗的线
            self.Id_Finish_Line[Name_Client][(node_parent, node_child)] = self.Plot_AddLine(
                Name_Client=Name_Client,
                lineFromXYZ=node_parent.np_arr[0:3],
                lineToXYZ=node_child.np_arr[0:3],
                lineColorRGB=[1.0, 1.0, 0.0],
                lineWidth=10.0,
            )
            # 坐标系
            if ii == 0:  # 起点
                self.Id_Finish_Coordinate[Name_Client][node_parent] = self.Plot_AddCoordinate(
                    Name_Client=Name_Client,
                    pos=node_parent.np_arr[0:3],
                    orn=p.getQuaternionFromEuler(node_parent.np_arr[3:6]),
                )
            self.Id_Finish_Coordinate[Name_Client][node_child] = self.Plot_AddCoordinate(
                Name_Client=Name_Client,
                pos=node_child.np_arr[0:3],
                orn=p.getQuaternionFromEuler(node_child.np_arr[3:6]),
            )

    def Plot_FinishLine(self, Name_Client, path):
        # 删除旧的
        # 连线
        for ii, ((node_parent, node_child), Id) in enumerate(self.Id_Finish_Line[Name_Client].items()):
            self.Plot_DeleteLine(
                Name_Client=Name_Client,
                Id=Id,
            )
        # 坐标系
        for ii, (node_parent, Id_dict) in enumerate(self.Id_Finish_Coordinate[Name_Client].items()):
            self.Plot_DeleteCoordinate(
                Name_Client=Name_Client,
                Id_dict=Id_dict,
            )
        # 绘制新的
        for ii, node_parent in enumerate(path[:-1]):
            node_child = path[ii + 1]
            # 连线
            # 删除Edge的可视化
            self.Plot_DeleteEdge(
                Name_Client=Name_Client,
                node_parent=node_parent,
                node_child=node_child,
            )
            # 绘制加粗的线
            self.Id_Finish_Line[Name_Client][(node_parent, node_child)] = self.Plot_AddLine(
                Name_Client=Name_Client,
                lineFromXYZ=node_parent.np_arr[0:3],
                lineToXYZ=node_child.np_arr[0:3],
                lineColorRGB=[1.0, 1.0, 0.0],
                lineWidth=10.0,
            )
            # 坐标系
            if ii == 0:  # 起点
                self.Id_Finish_Coordinate[Name_Client][node_parent] = self.Plot_AddCoordinate(
                    Name_Client=Name_Client,
                    pos=node_parent.np_arr[0:3],
                    orn=p.getQuaternionFromEuler(node_parent.np_arr[3:6]),
                )
            self.Id_Finish_Coordinate[Name_Client][node_child] = self.Plot_AddCoordinate(
                Name_Client=Name_Client,
                pos=node_child.np_arr[0:3],
                orn=p.getQuaternionFromEuler(node_child.np_arr[3:6]),
            )

    def Plot_FinalMove(self, Name_Client, Name_Body, path):
        # 删除旧的
        p.removeAllUserDebugItems()
        # 绘制新的
        for ii, node_parent in enumerate(path[:-1]):
            node_child = path[ii + 1]
            # 连线
            # 绘制加粗的线
            self.Id_Finish_Line[Name_Client][(node_parent, node_child)] = self.Plot_AddLine(
                Name_Client=Name_Client,
                lineFromXYZ=node_parent.np_arr[0:3],
                lineToXYZ=node_child.np_arr[0:3],
                lineColorRGB=[1.0, 1.0, 0.0],
                lineWidth=10.0,
            )
            # 坐标系
            if ii == 0:  # 起点
                self.Id_Finish_Coordinate[Name_Client][node_parent] = self.Plot_AddCoordinate(
                    Name_Client=Name_Client,
                    pos=node_parent.np_arr[0:3],
                    orn=p.getQuaternionFromEuler(node_parent.np_arr[3:6]),
                )
            self.Id_Finish_Coordinate[Name_Client][node_child] = self.Plot_AddCoordinate(
                Name_Client=Name_Client,
                pos=node_child.np_arr[0:3],
                orn=p.getQuaternionFromEuler(node_child.np_arr[3:6]),
            )
        # 一步一步走
        NN = 10
        for ii, node_parent in enumerate(path[:-1]):
            node_child = path[ii + 1]
            for jj in range(NN+1):
                pos = (node_child.np_arr[0:3] - node_parent.np_arr[0:3]) / NN * jj + node_parent.np_arr[0:3]
                orn = (node_child.np_arr[3:6] - node_parent.np_arr[3:6]) / NN * jj + node_parent.np_arr[3:6]
                Id_Client = self.Id_Client_Global[Name_Client]
                Id_Body = self.Id_Body_Global[Name_Client][Name_Body]
                p.resetBasePositionAndOrientation(
                    physicsClientId=Id_Client,
                    bodyUniqueId=Id_Body,
                    posObj=pos,
                    ornObj=p.getQuaternionFromEuler(orn),
                )
                time.sleep(0.1)
    def close(self):
        p.disconnect(
            # self.Id_Client
        )


if __name__ == "__main__":
    simulator = PyBullet_CollisionOnly()
    # 开启仿真环境
    # simulator.InitBeforeLoad(client_num=2, client_bias=[[(0.0, 0.0, 0.0), p.getQuaternionFromEuler((0.0, 0.0, 0.0))], [(10.0, 10.0, 10.0), p.getQuaternionFromEuler((0.0, 0.0, 0.0))]])
    simulator.InitBeforeLoad(client_num=1, client_bias=[[(0.0, 0.0, 0.0), p.getQuaternionFromEuler((0.0, 0.0, 0.0))]])
    start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # 起点
    goal = (4.0, 4.0, 4.0, 0.0, 0.0, 0.0)  # 终点
    simulator.load_Map(Config_.Name_Map, start, goal)
    simulator.load_Body(Config_.Name_Moving)
    simulator.InitAfterLoad()

    # 仿真
    UpdateFS = Config_.UpdateFS
    theta_arr = np.arange(0, 2 * np.pi, 0.01)
    r_arr = np.sin(theta_arr) + 1.0
    x_arr = np.sin(theta_arr) * r_arr
    y_arr = np.cos(theta_arr) * r_arr
    pitch_arr = np.arange(0, 2 * np.pi, 0.01)
    yaw_arr = np.arange(0, 2 * np.pi, 0.01)

    ii = 0
    while 1:
        ii += 1

        x = x_arr[ii % len(x_arr)]
        y = y_arr[ii % len(y_arr)]
        pitch = pitch_arr[ii % len(pitch_arr)]
        yaw = yaw_arr[ii % len(yaw_arr)]

        basePosition_1 = [x, y, 0.0]
        baseOrientation_1 = p.getQuaternionFromEuler([yaw, pitch, 0])
        simulator.Control_RealTimeBase['Client_0']['Cube'] = (basePosition_1, baseOrientation_1)

        # basePosition_2 = [x+2.0, y+2.0, 2.0]
        # baseOrientation_2 = p.getQuaternionFromEuler([yaw, pitch, 0])
        # simulator.Control_RealTimeBase['Client_1']['Cube'] = (basePosition_2, baseOrientation_2)

        time.sleep(1.0 / UpdateFS)
        simulator.simulate()
        # simulator.Update_State()
