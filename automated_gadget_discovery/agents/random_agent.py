import numpy as np
import gym
import gym_quantum_computing
from agents.tree import Tree
from agents.networkx_tree import NetworkxTree
from agents.comb import Comb

class RandomAgent(Comb):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'env' in kwargs:
            setattr(self, 'env', kwargs['env'])
        else:
            raise NotImplementedError('No environment was given to the agent, add key argument env')

        self.tree = None


    def rollout(self):
        """
        Performs random rollout
        Returns:
            observation (nd.array) last observation
            reward (float) reward at the end of the rollout

        """
        observation = self.env.reset()
        done = False
        while not done:
            action = self.rng.integers(self.env.num_actions)
            observation, reward, done = self.env.step(action)
        return observation, reward


if __name__ == "__main__":
    #setting random seed
    rng = np.random.default_rng(0)

    #ENVIRONMENT SPECIFIC
    #key arguments to control the environment parameters
    #dimension of the qudit
    DIM = 2
    #number of qudits
    NUM_QUDITS = 4
    #number of operations:
    MAX_OP = 6
    env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS,  max_op=MAX_OP)

    #MCTS
    agent_dict = {
        'EXPLORATION': 0.1,
        'TREE_TYPE': 'tree',
        }

    agent = Mcts(env=env, agent_dict = agent_dict, rng = rng)
    for i in range(1):
        print(agent.rollout())

    #MCTS using a networkx tree, so that the graph can be drawn and exported as pngs
    agent_dict = {
        'EXPLORATION': 0.1,
        'TREE_TYPE': 'networkx_tree',
        'START_DRAW': 0,
        'END_DRAW': 1,
        'GRAPH_FOLDER': 'results/graph'
        }

    agent = Mcts(env=env, agent_dict = agent_dict, rng = rng)
    for i in range(1):
        print(agent.rollout())
