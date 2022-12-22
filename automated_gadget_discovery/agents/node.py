class Node():
    def  __init__(self, id, action_num):
        """
        The nodes for the mcts tree structure

        Args:
            hsh (int) the id of the node
        """
        self.id = id
        self.action_num = action_num
        self.parent = None
        self.children = []
        # set of reachable unique+positively rewarded states.
        self.N = 0
        self.Q = 0
        self.sigma = 1.
        #maximum number of children
        self.num_children = 0
        self.removed_actions = []
