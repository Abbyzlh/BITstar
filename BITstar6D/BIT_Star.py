#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

from queue import PriorityQueue
from typing import List, Tuple
import numpy as np
import time
import json
from Node import Node
from PrintColours import *
import Config_
from Bullet.Module_PyBullet_CollisionOnly import PyBullet_CollisionOnly
import pybullet as p
import math


class bitstar:
    """BIT* algorithm class. This class implements the BIT* algorithm for path planning."""
    solution: List[Tuple[float, float, float, float, float, float]]

    def __init__(
        self,
        start: Node,
        goal: Node,
        simulator: PyBullet_CollisionOnly,
        # occ_map: Map,
        no_samples: int = 20,
        rbit: float = 10,
        dim: int = 2,
        log_dir: str = None,
        stop_time: int = 60,
    ) -> None:
        """Initialize BIT* algorithm.

        Args:
            start (Node): Node object representing the start position.
            goal (Node): Node object representing the goal position.
            occ_map (Map): Map object representing the occupancy grid.
            no_samples (int, optional): Number of samples to be generated in each iteration. Defaults to 20.
            rbit (float, optional): Radius of the ball to be considered for rewire. Defaults to 100.
            dim (int, optional): Dimension of the search space. Defaults to 2.
            log_dir (str, optional): Directory to save the log files. Defaults to None (no logging).
            stop_time (int, optional): Time limit for the algorithm to run in seconds. Defaults to 60s.
        """
        # Set the start node.
        self.start = start
        # Set the goal node.
        self.goal = goal
        # instantiate a simulator
        self.simulator=simulator
        # # Set the occupancy grid.
        # self.map = occ_map
        # Set the dimension of the search space.
        self.dim = dim
        # Set the radius of the ball within which the nodes are considered to be near each other for making connections.
        self.rbit = rbit
        # Number of samples to be generated in each step in the algorithm.
        self.m = no_samples
        # The current cost-to-come of the goal node.
        self.ci = np.inf
        # The old cost-to-come of the goal node.
        self.old_ci = np.inf
        # The minimum cost-to-come of the goal node.
        self.cmin = np.linalg.norm(self.goal.np_arr - self.start.np_arr)
        # # Used to get the length of the map.
        # self.flat_map = self.map.map.flatten()
        # Set the time limit for the algorithm to run. Default is 60s. This is used to stop the algorithm as it can only be asymptotically optimal.
        self.stop_time = stop_time

        # matrix of hyper ellipse
        self.xcenter = (self.goal.np_arr + self.start.np_arr) / 2.0
        self.C = self.rotation_to_world_frame()
        self.L = np.zeros([self.dim, self.dim])
        self.use_PHS=False

        # Set of all the vertices in the tree.
        self.V = set()
        # Set of all the edges in the tree.
        self.E = set()
        # Set of all the vertices used for visualization.
        self.E_vis = set()
        # Set of all new vertices.
        self.x_new = set()
        # Set of all vertices that are to be reused.
        self.x_reuse = set()
        # Set of all vertices that are not expanded.
        self.unexpanded = set()

        # Set of all vertices that are not connected to the tree.
        self.unconnected = set()
        # Set of all vertices that are in the goal set.
        self.vsol = set()

        # Priority queue for the vertices.
        self.qv = PriorityQueue()
        # Priority queue for the edges.
        self.qe = PriorityQueue()
        # This is a workaround when the gt + c + h_hat values for two edges and the order of the edges in the queue is used to break the tie.
        self.qe_order = 0
        # This is a workaround when the gt + h_hat values are the same for two nodes and the order of the nodes in the queue is used to break the tie.
        self.qv_order = 0

        # Add the start node to the tree.
        self.V.add(start)
        # Add the goal node to the unconnected set.
        self.unconnected.add(goal)
        # Add the start node to the unexpanded set.
        self.unexpanded = self.V.copy()
        # Add the start node to the x_new set.
        self.x_new = self.unconnected.copy()

        # Add the start node to the priority queue.
        self.qv.put((start.gt + self.h_hat(start), self.qv_order, start))
        # Increment the order of the priority queue.
        self.qv_order += 1
        # # Get the current Prolate HyperSpheroid (PHS) for the current cost.
        # self.get_PHS()
        # Set the flag to save the log files.
        self.save = False
        # If the log directory is not None, then save the log files.
        if log_dir is not None:
            # Reset the save flag.
            self.save = True
            # Get the path to the log directory.
            self.log_dir = log_dir
            # Template Dictionary to store the contents as a JSON in the log file.
            self.json_contents = {
                "new_edges": [],
                "rem_edges": [],
                "final_path": [],
                "ci": [],
                "final_edge_list": [],
            }
        self.solution = []

    def g_hat(self, node: Node):
        """
        Get the estimated cost from the start to the node
        :param node:
        :return:
        """
        return np.linalg.norm(node.np_arr-self.start.np_arr)

    def h_hat(self,node:Node):
        """
        Get the estimated cost from the node to the goal
        :param node:
        :return:
        """
        return np.linalg.norm(self.goal.np_arr-node.np_arr)

    def f_hat(self,node:Node):
        return self.g_hat(node)+self.h_hat(node)

    def gt(self, node: Node) -> float:
        """Get the cost of the path from the start to the node by traversing through the Tree.

        Args:
            node (Node): The node for which the cost is to be calculated.

        Returns:
            g_t (float): The cost of the path from the start to the node by traversing through the Tree.
        """
        # If the node is the start node, then the cost is 0.
        if node == self.start:
            return 0
        # If the node is not in the tree, then the cost is infinity.
        elif node not in self.V:
            return np.inf
        # Return the cost of the path from the start to the node by traversing through the Tree. This is the sum of the cost of the edge from the parent to the node and the cost of the path from the start to the parent.
        return node.par_cost + node.parent.gt

    def c_hat(self, node1: Node, node2: Node) -> float:
        """Estimated cost of the edge from node1 to node2 using L2 norm.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            c_hat (float): The estimated L2 norm cost of the straight line path from node1 to node2.
        """
        # Return the L2 norm of the difference between the two nodes.
        return np.linalg.norm(node1.np_arr - node2.np_arr)

    def a_hat(self, node1: Node, node2: Node) -> float:
        """This is the sum of the estimated cost of the path from start to node1 (L2 Norm), the estimated cost of the path from node1 to node2 (L2 norm), and the heuristic cost (L2 Norm) of node2 from goal.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            g_hat(node1) + c_hat(node1, node2) + h_hat(node2) (float): The total estimated cost.
        """
        # Return the sum of the estimated cost of the path from start to node1 (L2 Norm), the estimated cost of the path from node1 to node2 (L2 norm), and the heuristic cost (L2 Norm) of node2 from goal.
        return self.g_hat(node1) + self.c_hat(node1, node2) + self.h_hat(node2)

    def c(self, node1: Node, node2: Node) -> float:  #Todo: Collision Check
        """True cost of the edge between node1 and node2. This is the L2 norm cost of the straight line path from node1 to node2. If the path is obstructed, the cost is set to infinity.

        Args:
            node1 (Node): The first node.
            node2 (Node): The second node.

        Returns:
            c(float): The true cost of the edge between node1 and node2.
        """
        if self.collision_check_line(node1,node2):
            return np.inf
        # Return the L2 norm of the difference between the two nodes if the edge is not obstructed.
        return self.c_hat(node1, node2)

    def near(self, search_set: set, node: Node) -> set:
        """Returns the set of nodes in the search_set which are within the radius of the ball centered at node (rbit). The node itself is not included in the returned set.

        Args:
            search_set (set): The set of nodes to be searched for near nodes.
            node (Node): The node about which the ball is centered.

        Returns:
            Near Nodes (set): The set of nodes in the search_set which are within the radius of the ball centered at node (rbit).
        """
        # Set to store the near nodes.
        near = set()
        # For each node in the search_set, check if it is within the radius of the ball centered at node (rbit). If it is, then add it to the set of near nodes.
        for n in search_set:
            if (self.c_hat(n, node) <= self.rbit) and (n != node):
                near.add(n)
        # Return the set of near nodes.
        return near

    def expand_next_vertex(self) -> None:
        """Expands the next vertex in the queue of vertices to be expanded (qv). This function is called by the main loop of the algorithm."""
        # Get the next vertex to be expanded from the queue of vertices to be expanded (qv).
        vmin = self.qv.get(False)[2]
        # Set of nodes in the Tree which are within the radius of the ball centered at vmin (rbit).
        x_near = None
        # If vmin is in the set of unexpanded nodes, then the set of near nodes is the set of unconnected nodes which are within the radius of the ball centered at vmin (rbit).
        if vmin in self.unexpanded:
            x_near = self.near(self.unconnected, vmin)
        # Else, the set of near nodes is the intersection of the set of unconnected nodes and the set of new nodes which are within the radius of the ball centered at vmin (rbit).
        else:
            intersect = self.unconnected & self.x_new
            x_near = self.near(intersect, vmin)

        for x in x_near:
            # Edge is added to the queue of edges if the edge is estimated cost less than the current cost (ci).
            if self.a_hat(vmin, x) < self.ci:
                # Actual cost of the edge is calculated.
                cost = vmin.gt + self.c(vmin, x) + self.h_hat(x)
                # Edge is added to the queue of edges.
                self.qe.put((cost, self.qe_order, (vmin, x)))
                self.qe_order += 1

        if vmin in self.unexpanded:
            # Gets the set of nodes near vmin that is already in the Tree.
            v_near = self.near(self.V, vmin)
            for v in v_near:
                # For all nodes in the near list. If the edge is not in the all edges set, and the estimated cost of the edge is less than the current cost (ci), and the estimated cost of the path from start to v and
                if (
                    (not (vmin, v) in self.E)
                    and (self.a_hat(vmin, v) < self.ci)
                    and (self.g_hat(vmin) + self.c_hat(vmin, v) < v.gt)
                ):
                    # Cost of the edge is calculated.
                    cost = vmin.gt + self.c(vmin, v) + self.h_hat(v)
                    # Edge is added to the queue of edges.
                    self.qe.put((cost, self.qe_order, (vmin, v)))
                    self.qe_order += 1
            # Vertex is removed from the set of unexpanded nodes.
            self.unexpanded.remove(vmin)

    def sample_unit_ball(self) -> np.array:
        """Samples a point uniformly from the unit ball. This is used to sample points from the Prolate HyperSpheroid (PHS).

        Returns:
            Sampled Point (np.array): The sampled point from the unit ball.
        """
        u = np.random.normal(0, 1, self.dim)
        norm = np.linalg.norm(u)
        r = np.random.random() ** (1.0 / self.dim)
        x = r * u / norm
        return x

    def rotation_to_world_frame(self):
        a1 = (self.goal.np_arr - self.start.np_arr) / self.cmin
        one1 = np.eye(1, self.dim)
        M = np.outer(a1, one1)
        U, Sigma, Vt = np.linalg.svd(M)
        diag = np.eye(self.dim)
        diag[-1, -1] = np.linalg.det(U) * np.linalg.det(Vt.T)
        C = U @ diag @ Vt
        return C

    def samplePHS(self) -> np.array:
        """Samples a point from the Prolate HyperSpheroid (PHS) defined by the start and goal nodes.

        Returns:
            Node: The sampled node from the PHS.
        """
        # Calculate the center of the PHS.
        center = self.xcenter
        # Get the radius of the first axis of the PHS.
        r1 = self.ci / 2
        # Get the radius of the other axes of the PHS.
        rn = np.sqrt(self.ci ** 2 - self.cmin ** 2) / 2

        diag_val = np.zeros(self.dim)
        diag_val[0] = self.ci / 2.0
        for i in range(1, self.dim):
            diag_val[i] = rn
        L = np.diag(diag_val)
        self.L = L

        # Sample a point from the PHS.
        while True:
            # Sample a point from the unit ball.
            x_ball = self.sample_unit_ball()
            # Transform the point from the unit ball to the PHS.
            op = self.C @ L @ x_ball + self.xcenter
            # Round the point to 7 decimal places.
            op = np.around(op, 7)
            # Check if the point is in the PHS.
            # if (int(op[0]), int(op[1])) in self.intersection:  #Todo
            #     break
            if self.in_range(Node(op)) and (not self.collision_check_point(Node(op))) :
                break
        return tuple(op)

    def in_range(self,node:Node):
        if_in_range = all(Config_.Map_DimRange[i][0] < node.coords[i] < Config_.Map_DimRange[i][1] for i in range(6))
        return if_in_range

    def if_PHS(self):
        r1 = self.ci / 2
        rn = np.sqrt(self.ci ** 2 - self.cmin ** 2) / 2
        V_PHS = np.pi ** (self.dim / 2) / math.gamma(self.dim / 2) * r1 * (rn**(self.dim-1)) / self.dim * 2
        V_cfg = np.prod(np.array(Config_.Map_DimRange)[:, 1] - np.array(Config_.Map_DimRange)[:, 0])
        return V_PHS<V_cfg

    def sample_free(self)-> np.array:
        """
        Sample a node from free space, not in PHS
        :return:
        """
        while True:
            point = [np.random.uniform(*Config_.Map_DimRange[ii]) for ii in range(len(Config_.Map_DimRange))]
            node = Node(point)
            if_collision = self.collision_check_point(node)
            if not if_collision:
                return tuple(point)

    def sample(self) -> Node:
        """Samples a node from the Prolate HyperSpheroid (PHS) or the free space of the map depending on the current cost (ci).

        Returns:
            Node: The sampled node from the PHS or the free space of the map.
        """
        # A random point is sampled from the PHS.
        xrand = None
        # # Do not generate a new PHS if the cost threshold (ci) has not changed.
        # if self.old_ci != self.ci:
        # # if self.ci<np.inf:
        #     self.get_PHS()

        # If the cardinality of the PHS is less than the cardinality of the free space, sample from the PHS.
        if not self.use_PHS:   # 超椭球的体积小于配置空间体积时,椭球采样
            self.use_PHS=self.if_PHS()
        if self.use_PHS:
            xrand = self.samplePHS()
        # Else sample from the free space.
        else:
            xrand = self.sample_free()
        # Return the sampled node as a Node object.
        return Node(xrand)

    def collision_check_point(self,node:Node):
        if_collision = self.simulator.Check_Collision_Point(
            Name_Client='Client_0',
            Name_BodyA=Config_.Name_Moving,
            pos=node.np_arr[0:3],
            orn=p.getQuaternionFromEuler(node.np_arr[3:6]),
        )
        return if_collision

    def collision_check_line(self,node1:Node,node2:Node):
        if_collision = self.simulator.Check_Collision_Line(
            Name_Client="Client_0",
            Name_BodyA=Config_.Name_Moving,
            pos_start=node1.np_arr[0:3],
            orn_start=p.getQuaternionFromEuler(node1.np_arr[3:6]),
            pos_end=node2.np_arr[0:3],
            orn_end=p.getQuaternionFromEuler(node2.np_arr[3:6]),
        )
        return if_collision

    def prune(self) -> None:
        """Prunes the search tree based on the current cost threshold (ci). It removes all nodes
        from the search tree which have a f_hat value greater than the current cost threshold (ci).
        It also removes all edges which connect to a node which has been removed from the search
        tree."""
        # Set of removed nodes from the search tree but can be reused.
        self.x_reuse = set()
        # Remove all nodes from the search tree which have a f_hat value greater than the current cost threshold (ci).
        new_unconnected = set()
        for n in self.unconnected:
            if self.f_hat(n) < self.ci:
                new_unconnected.add(n)
        self.unconnected = new_unconnected

        # A list of removed edges from the search tree. This is used to update the visualization.
        rem_edge = []
        # Sort the nodes in the search tree by their g_t value.
        sorted_nodes = sorted(self.V, key=lambda x: x.gt, reverse=True)
        # Remove all nodes from the search tree which have a f_hat value greater than the current cost threshold (ci). Also remove all nodes which have a g_t + h_hat value greater than the current cost threshold (ci).
        for v in sorted_nodes:
            # Do not remove the start or goal nodes.
            if v != self.start and v != self.goal:
                if (self.f_hat(v) > self.ci) or (v.gt + self.h_hat(v) > self.ci):
                    self.V.discard(v)
                    self.vsol.discard(v)
                    self.unexpanded.discard(v)
                    self.E.discard((v.parent, v))
                    # self.E_vis.discard((v.parent.coords, v.coords))
                    # If the save flag is set to True, add the removed edge to the list of removed edges.
                    if self.save:
                        rem_edge.append((v.parent.coords, v.coords))
                    # delete edge
                    if Config_.IF_GUI:
                        self.simulator.Plot_DeleteEdge(
                            Name_Client='Client_0',
                            node_parent=v.parent,
                            node_child=v,
                        )
                    v.parent.children.remove(v)
                    # Add the removed node to the set of nodes which can be reused if the node's f_hat < ci.
                    if self.f_hat(v) < self.ci:
                        self.x_reuse.add(v)
                    else:
                        # If the node's f_hat > ci we delete the node.
                        del v
        # If the save flag is set to True, save the removed edges.
        if self.save:
            self.save_data(None, rem_edge)
        # Add the  goal node back to the set of unexpanded nodes.
        self.unconnected.add(self.goal)

    def final_solution(self):
        """Returns the final solution path and the path length.

        Returns:
            Tuple[List[Tuple[float, float]], float]: The final solution path and the path length.
        """
        # If the goal node has an infinite g_t value, then there is no solution.
        if self.goal.gt == np.inf:
            return None, None,None
        # Empty list to store the solution path.
        path = []
        path_node =[]
        # Path length is initialized to 0.
        path_length = 0
        # Start from the goal node and traverse the parent nodes until the start node is reached.
        node = self.goal
        while node != self.start:
            path.append(node.coords)
            path_node.append(node)
            path_length += node.par_cost
            node = node.parent
        # Add the start node to the path.
        path.append(self.start.coords)
        path_node.append(self.start)
        # Reverse the path and return the path and the path length.
        self.solution = path_node[::-1]
        return path[::-1],path_node[::-1], path_length

    def update_children_gt(self, node: Node) -> None:
        """Updates the true cost of a node from start (gt) of all the children of a node in the search tree. This is used when an edge is added/removed from the search tree.

        Args:
            node (Node): The node whose children's true cost needs to be updated.
        """
        # Update the true cost of the children of the node.
        for c in node.children:
            # The true cost of the child is the true cost of the parent + the cost of the edge connecting the parent and the child.
            c.gt = c.par_cost + node.gt
            # Recursively update the true cost of the children of the child.
            self.update_children_gt(c)

    def save_data(
        self, new_edge: tuple, rem_edge: list, new_final: bool = False
    ) -> None:
        """Saves the data as a JSON file for the current iteration of the algorithm. It is used to generate the plots and animations.

        Args:
            new_edge (tuple): The new edge added to the search tree.
            rem_edge (list): The list of edges removed from the search tree.
            new_final (bool, optional): Whether the final solution path has changed. Defaults to False.
        """
        # Update the current cost (ci).
        self.json_contents["ci"].append(self.ci)
        # New edges added to the search tree.
        self.json_contents["new_edges"].append(new_edge)
        # Removed edges from the search tree.
        self.json_contents["rem_edges"].append(rem_edge)

        # If the final solution path has changed, update the final solution path.
        if new_final:
            # Get the final solution path and the path length.
            current_solution, _ = self.final_solution()
            # Add the final solution path to the JSON file.
            self.json_contents["final_path"].append(current_solution)
        else:
            # If the final solution path has not changed, add None to the JSON file.
            self.json_contents["final_path"].append(None)

    def dump_data(self, goal_num: int) -> None:
        """Dumps the data as a JSON file for the current simulation run.

        Args:
            goal_num (int): The Simulation run number. This is used to name the JSON file. The JSON file is saved in the log directory.
        """
        print(f"{CGREENBG}Data saved.{CEND}")
        # Add the final edge list to the JSON file.
        self.json_contents["final_edge_list"] = list(self.E_vis)

        # Converting json_contents to json object.
        json_object = json.dumps(self.json_contents, indent=4,default=float)

        # Open a file and dump the JSON object.
        with open(
            f"{self.log_dir}/path{goal_num:02d}.json",
            "w",
        ) as outfile:
            # Write the JSON object to the file.
            outfile.write(json_object)

        # Reset the JSON contents.
        self.json_contents = {
            "new_edges": [],
            "rem_edges": [],
            "final_path": [],
            "ci": [],
            "final_edge_list": [],
        }

    def make_plan(self):
        """The main BIT* algorithm. It runs the algorithm until the time limit is reached. It also saves the data for the current simulation run if the save flag is set to True.

        Returns:
            Tuple[List[Tuple[int, int]], float, List[float]]: The final solution path, the path length and the list of time taken for each iteration.
        """
        # If the start or goal is not in the free space, return None.
        if self.collision_check_point(self.start) or  self.collision_check_point(self.goal):
            print(f"{CYELLOW2}Start or Goal not in free space.{CEND}")
            return None, None, None

        # If the start and goal are the same, return the start node, path length 0 and None for the time taken.
        if self.start.coords == self.goal.coords:
            print(f"{CGREEN2}Start and Goal are the same.{CEND}")
            self.vsol.add(self.start)
            self.ci = 0
            return [self.start.coords], 0, None

        # Initialize the iteration counter.
        it = 0
        # Initialize the number of times the goal has been reached.
        goal_num = 0
        # Start the timer for the algorithm.
        timeStamp_StartAlgorithm = time.time()
        # Start the timer for the current simulation run.
        timeStamp_StartSolution = time.time()

        # List to store the time taken for each simulation run.
        time_taken = []
        # List to store the path length for each simulation run.
        path_lengths = []
        try:
            # Start the main loop of the algorithm.
            while True:
                # If the time limit is reached, return the final solution path.
                if time.time() - timeStamp_StartAlgorithm >= self.stop_time:
                    print(
                        f"\n\n{CITALIC}{CYELLOW2}============================================= Stopping due to time limit ============================================{CEND}"
                    )
                    path,path_node,path_length = self.final_solution()
                    if Config_.IF_GUI:
                        self.simulator.Plot_FinishLine(path=path_node, Name_Client="Client_0")
                    return path, path_length, time_taken, path_lengths

                # Increment the iteration counter.
                it += 1
                # print(it)
                # If the Edge queue and the Vertex queue are empty.
                if self.qe.empty() and self.qv.empty():
                    # Prune the search tree.
                    self.prune()
                    # Set of Sampled nodes.
                    x_sample = set()
                    # Sample m nodes.
                    while len(x_sample) < self.m:
                        x_sample.add(self.sample())

                    # Add the sampled nodes and reuse nodes to the new nodes set.
                    self.x_new = self.x_reuse | x_sample
                    # Add the new nodes to the unconnected set.
                    self.unconnected = self.unconnected | self.x_new
                    for n in self.V:
                        # Add all the nodes in the search tree to the Vertex queue.
                        self.qv.put((n.gt + self.h_hat(n), self.qv_order, n))
                        self.qv_order += 1

                while True:
                    # Run until the vertex queue is empty.
                    if self.qv.empty():
                        break
                    # Expand the next vertex.
                    self.expand_next_vertex()

                    # If the Edge queue is empty, continue.
                    if self.qe.empty():
                        continue
                    if self.qv.empty() or self.qv.queue[0][0] <= self.qe.queue[0][0]:
                        break

                # If the Edge queue is empty, continue.
                if not (self.qe.empty()):
                    # Pop the next edge from the Edge queue.
                    (vmin, xmin) = self.qe.get(False)[2]
                    # The Four conditions for adding an edge to the search tree given in the paper.
                    if vmin.gt + self.c_hat(vmin, xmin) + self.h_hat(xmin) < self.ci:
                        if vmin.gt + self.c_hat(vmin, xmin) < xmin.gt:
                            # Calculate the cost of the edge.
                            cedge = self.c(vmin, xmin)
                            if vmin.gt + cedge + self.h_hat(xmin) < self.ci:
                                if vmin.gt + cedge < xmin.gt:
                                    # Remove the edge from the search tree.
                                    rem_edge = []
                                    # If the node is in the search tree remove the edge.
                                    if xmin in self.V:
                                        # Remove the edge from the edge set.
                                        self.E.remove((xmin.parent, xmin))
                                        # # Remove the edge from the edge set for the JSON file. Done in a funny way.
                                        # self.E_vis.remove((xmin.parent.coords, xmin.coords))
                                        # A funny way to remove node xmin from the children of its parent.
                                        xmin.parent.children.remove(xmin)
                                        # Add the edge to the list of removed edges for the JSON file.
                                        rem_edge.append((xmin.parent.coords, xmin.coords))
                                        # delete edge
                                        if Config_.IF_GUI:
                                            self.simulator.Plot_DeleteEdge(
                                                Name_Client='Client_0',
                                                node_parent=xmin.parent,
                                                node_child=xmin,
                                            )
                                        # Update the parent of the node.
                                        xmin.parent = vmin
                                        # Update the cost of the edge.
                                        xmin.par_cost = cedge
                                        # Get the new gt value of the node.
                                        xmin.gt = self.gt(xmin)
                                        # Add the edge to search tree/edge set.
                                        self.E.add((xmin.parent, xmin))
                                        # # Add the edge to the search tree/edge set for the JSON file. Done in a funny way.
                                        # self.E_vis.add((xmin.parent.coords, xmin.coords))
                                        # A funny way to add node xmin to the children of its new parent.
                                        xmin.parent.children.add(xmin)
                                        # Update the gt values of the children of the node.
                                        self.update_children_gt(xmin)
                                    else:
                                        # Add the node to the search tree/vertex set.
                                        self.V.add(xmin)
                                        # Update the parent of the node.
                                        xmin.parent = vmin
                                        # Update the cost of the edge.
                                        xmin.par_cost = cedge
                                        # Get the new gt value of the node.
                                        xmin.gt = self.gt(xmin)
                                        # Add the edge to search tree/edge set. Done in a funny way.
                                        self.E.add((xmin.parent, xmin))
                                        # # Add the edge to the search tree/edge set for the JSON file. Done in a funny way.
                                        # self.E_vis.add((xmin.parent.coords, xmin.coords))
                                        self.qv_order += 1  # Why is this here?
                                        # Add the node to the Unexpanded set.
                                        self.unexpanded.add(xmin)
                                        # If the node is the goal, add it to the solution set.
                                        if xmin == self.goal:
                                            # Solution set.
                                            self.vsol.add(xmin)
                                        # A funny way to add node xmin to the children set of its parent.
                                        xmin.parent.children.add(xmin)
                                        # Remove the node from the unconnected set.
                                        self.unconnected.remove(xmin)

                                    # Create a new edge for the JSON file.
                                    new_edge = (xmin.parent.coords, xmin.coords)
                                    # draw new edge
                                    if Config_.IF_GUI:
                                        self.simulator.Plot_AddEdge(
                                            Name_Client='Client_0',
                                            node_parent=xmin.parent,
                                            node_child=xmin,
                                        )
                                    # Set the ci to max of the goal gt and the cmin. This is done so that in weird cases where the goal gt is very close to cmin and due to the float inaccuracy the goal gt is less than cmin, causing the algorithm to crash.
                                    self.old_ci=self.ci
                                    self.ci = max(self.goal.gt, self.cmin)

                                    # if the save flag is set, save the data.
                                    if self.save:
                                        self.save_data(
                                            new_edge, rem_edge, self.ci != self.old_ci
                                        )
                                    # If there is a change in the ci value. The algorithm has found a new solution.
                                    if self.ci != self.old_ci:
                                        # If the time limit is reached, return the current solution.
                                        if time.time() - timeStamp_StartAlgorithm >= self.stop_time:
                                            print(
                                                f"\n\n{CITALIC}{CYELLOW2}============================================= Stopping due to time limit ============================================{CEND}"
                                            )
                                            path,path_node, path_length = self.final_solution()
                                            if Config_.IF_GUI:
                                                self.simulator.Plot_FinishLine(path=path_node, Name_Client="Client_0")
                                            return path, path_length, time_taken,path_lengths
                                        # Print the solution.
                                        print(
                                            f"\n\n{CBOLD}{CGREEN2}================================================ GOAL FOUND {goal_num:02d} times ================================================{CEND}"
                                        )
                                        # The time taken to find the solution.
                                        print(
                                            f"{CBLUE2}Time Taken:{CEND} {time.time() - timeStamp_StartSolution}",
                                            end="\t\t",
                                        )
                                        # Append the time taken to the list of time taken.
                                        time_taken.append(time.time() - timeStamp_StartSolution)

                                        # Reset the start time for the next solution.
                                        timeStamp_StartSolution = time.time()
                                        # Get the solution path and the length of the path.
                                        solution,solution_node, length = self.final_solution()
                                        if Config_.IF_GUI:
                                            self.simulator.Plot_FinishLine(path=solution_node, Name_Client="Client_0")
                                        # Print the solution length.
                                        print(
                                            f"{CBLUE2}Path Length:{CEND} {length}{CEND}"
                                        )
                                        path_lengths.append(length)
                                        # Print the old_ci, new_ci, ci - cmin, and the difference in the ci values.
                                        print(
                                            f"{CBLUE2}Old CI:{CEND} {self.old_ci}\t{CBLUE2}New CI:{CEND} {self.ci}\t{CBLUE2}ci - cmin:{CEND} {round(self.ci - self.cmin, 5)}\t {CBLUE2}Difference in CI:{CEND} {round(self.old_ci - self.ci, 5)}"
                                        )
                                        # Print the solution path.
                                        print(f"{CBLUE2}Path:{CEND} {solution}")
                                        # Set the old_ci to the new_ci.
                                        # self.old_ci = self.ci
                                        # If the save flag is set, Dump the data.
                                        if self.save:
                                            self.dump_data(goal_num)
                                        # Increment the goal number.
                                        goal_num += 1

                    else:
                        # Reset the Edge queue and the Vertex queue.
                        self.qe = PriorityQueue()
                        self.qv = PriorityQueue()

                else:
                    # Reset the Edge queue and the Vertex queue.
                    self.qe = PriorityQueue()
                    self.qv = PriorityQueue()
        except KeyboardInterrupt:
            # If the user presses Ctrl+C, return the current solution path, the time taken to find the solution, and the path length.
            print(time.time() - timeStamp_StartSolution)
            print(self.final_solution())
            path,path_node, path_length = self.final_solution()
            if Config_.IF_GUI:
                self.simulator.Plot_FinishLine(path=path_node, Name_Client="Client_0")
            return path, path_length, time_taken, path_lengths
