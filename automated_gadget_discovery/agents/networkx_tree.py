import numpy as np
from collections import OrderedDict
import networkx as nx
import pylab as plt
from networkx.drawing.nx_agraph import graphviz_layout
from agents.node import Node
from agents.tree import Tree
import pathlib

class NetworkxTree(Tree):
    def __init__(self,**kwargs):
        """
        Builds the tree for the mcts search
        """
        super().__init__()
        self.type = 'networkx_tree'
        self.graph = nx.DiGraph()
        self.call_counter = 0

        if 'agent_dict' in kwargs and type(kwargs['agent_dict']) is dict:
            setattr(self, 'agent_dict', kwargs['agent_dict'])
        else:
            setattr(self, 'agent_dict', {})
        if 'START_DRAW' in self.agent_dict.keys() and type(self.agent_dict['START_DRAW']) is int:
            setattr(self, 'start_draw', self.agent_dict['START_DRAW'])
        else:
            setattr(self, 'start_draw', 0)
        if 'END_DRAW' in self.agent_dict.keys() and type(self.agent_dict['END_DRAW']) is int:
            setattr(self, 'end_draw', self.agent_dict['END_DRAW'])
        else:
            setattr(self, 'end_draw', 10)
        if 'GRAPH_FOLDER' in self.agent_dict.keys() and type(self.agent_dict['GRAPH_FOLDER']) is str:
            setattr(self, 'graph_folder', self.agent_dict['GRAPH_FOLDER'])
        else:
            setattr(self, 'graph_folder', 'results/graph')
        #create folder if necessary
        pathlib.Path(self.graph_folder).mkdir(parents=True, exist_ok=True)


    def reset(self,matrix):
        """
        First time call it creates the top node of the tree. Every other call it resets the tree to the starting observation (matrix)

        Args:
           matrix (np.array)  The current starting observation
        """
        if self.top_node == None:
            #hsh = self.generate_hsh(matrix)
            id = 0
            self.current_node = Node(str(id), id)
            self.top_node = self.current_node
            self.graph.add_node(id)
            print(self.graph.node)

        else:
            self.current_node = self.top_node



    def add_child(self, action):
        """
        Generates a node with the action number as id and adds it to the tree dict.
        Args:
            action (int) the hsh of the current observation
        """
        child_id = str(self.current_node.id)+'_'+str(action)
        child = Node(child_id,action)
        child.parent = self.current_node
        self.current_node.children.append(child)
        self.graph.add_node(child_id)
        self.graph.add_edge(self.current_node.id,child_id)
        return child


    def draw_graph(self):
        """draws the graph of the tree.
        Attributes:
            type:
                ecm.Clips used for coloring
            label:
                name of the node
            learning_value:
                learning_value of the edge

        """
        plt.clf()
        #print(self.graph.edges)

        pos = graphviz_layout(self.graph, prog='dot')
        for node in self.graph.nodes:
            if node == self.current_node.id:
                nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=[node], node_color="red",node_size=100)
            else:
                nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=[node], node_color="grey", node_size=50)
        nx.draw_networkx_edges(self.graph, pos=pos)
        # plt.show()
        plt.axis('off')
        plt.savefig(self.graph_folder+"/Graph_" + str(self.call_counter) + ".png", format="PNG")
        self.call_counter+=1
