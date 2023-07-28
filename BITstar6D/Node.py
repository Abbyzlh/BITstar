import numpy as np

class Node:
    def __init__(self,coords,parent=None,par_cost=None,gt=np.inf):
        self.coords=coords
        # Extract coordinates from tuple.
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]
        self.roll = coords[3]
        self.pitch = coords[4]
        self.yaw = coords[5]
        self.np_arr=np.array(list(coords))

        # Initialize parent, edge cost (par_cost), and g_t.
        self.parent = parent
        self.par_cost = par_cost
        self.gt = gt

        # Initialize the children set.
        self.children = set()



