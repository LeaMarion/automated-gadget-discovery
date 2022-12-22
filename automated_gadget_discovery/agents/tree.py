import numpy as np
from collections import OrderedDict
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout
from agents.node import Node

class Tree():
    def __init__(self):
        """
        Builds the tree for the mcts search
        """
        #self.dict = {}
        self.type = 'tree'
        self.top_node = None
        self.current_node = self.top_node
        self.backup_node = self.top_node
        self.prev_node = self.top_node


    def reset(self, matrix):
        """
        First time call it creates the top node of the tree. Every other call it resets the tree to the starting observation (matrix)

        Args:
           matrix (np.array)  The current starting observation
        """
        if self.top_node == None:
            #hsh = self.generate_hsh(matrix)
            id = 0
            self.current_node = Node(str(id),id)
            self.backup_node = self.current_node
            self.top_node = self.current_node
            self.top_node.parents = None
            self.prev_node = self.current_node
        else:
            self.current_node = self.top_node


    def generate_hsh(self, matrix):
        """
        Generates hsh from matrix according to the python hash function. Need to convert the matrix to a tuple first.
        Args:
            matrix (np.array) the current observation in form of a matrix

        Returns:
             hsh (int) the hash of the current observation
        """
        hsh = hash(tuple(matrix.flatten()))
        return hsh

    def add_children(self,list_of_children):
        """
        Gets a list of children that is added to the tree at the current node.

        Args:
            list_of_children (list) list of children added to the current node.

        """
        self.current_node.children = OrderedDict()
        for action_index, child in enumerate(list_of_children):
            hsh = self.generate_hsh(child)
            child = Node(hsh)
            child.parent = self.current_node
            self.current_node.children.update({hsh: (child,action_index)})

    def add_child(self, action):
        child_id = str(self.current_node.id)+'_'+ str(action)
        child = Node(child_id, action)
        child.parent = self.current_node
        self.current_node.children.append(child)
        return child



    def search_child(self,child_hsh):
        """
        Given the hash code of the child this function searches for the corresponding child of the current node.

        child_hsh (int) the hsh of the child
        """
        for child in self.current_node.children:
            if child.id == child_hsh:
                return child


    def remove_current_node(self):
        """
        Given the hash code of the child, this function removes the corresponding child of the current node.

        child_hsh (int) the hsh of the child
        """
        self.current_node.parent.removed_actions.append(self.current_node.action_num)
        self.current_node.parent.children.remove(self.current_node)



