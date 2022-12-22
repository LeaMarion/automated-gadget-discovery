import numpy as np
from agents.comb import Comb
from agents.mcts import Mcts
from agents.random_agent import RandomAgent
from sys import stdout, getsizeof
from utils import *
import sys
import gym
import os
import gym_quantum_computing
import pathlib
from datetime import datetime



class Agent():
    def __init__(self, **kwargs):
        """
        agent class that simulates the interaction between chosen agent type and environment
        Args:
            env:
            **kwargs:
                env (class) the environment the agent interacts with, not optional
                agent_dict (dict) dictonary with agent configurations
                file_path (str) path to save the files
                patterns (list) list of patterns received from mining
                rng (gen) random number generator
        """
        if 'env' in kwargs:
            setattr(self, 'env', kwargs['env'])
        else:
            raise NotImplementedError('No environment was given to the agent, add key argument env')
        
        if 'seq_proc' in kwargs:
            setattr(self, 'seq_proc', kwargs['seq_proc'])
        else:
            raise NotImplementedError('No sequence processor was given to the agent, add key argument seq_proc')
        
        
        if 'tree' in kwargs:
            setattr(self, 'tree', kwargs['tree'])

        if 'patterns' in kwargs and type(kwargs['patterns']) is list:
            setattr(self, 'patterns', kwargs['patterns'])
        else:
            pattern = default_pattern()
            setattr(self, 'patterns', [pattern])

        if 'results_folder' in kwargs and type(kwargs['results_folder']) is str:
            setattr(self, 'results_folder', kwargs['results_folder'])
        else:
            setattr(self, 'results_folder', 'results/cycle/')
        
        if 'supplementary_folder' in kwargs and type(kwargs['supplementary_folder']) is str:
            setattr(self, 'supplementary_folder', kwargs['supplementary_folder'])
        else:
            setattr(self, 'supplementary_folder', '')

        #create folder if necessary
        pathlib.Path(self.results_folder + self.supplementary_folder).mkdir(parents=True, exist_ok=True)

        if 'rng' in kwargs:
            setattr(self, 'rng', kwargs['rng'])
        else:
            setattr(self, 'rng', np.random.default_rng())

        if 'global_run' in kwargs and type(kwargs['global_run']) is int:
            setattr(self, 'global_run', kwargs['global_run'])
        else:
            setattr(self, 'global_run', 0)

        if 'cycle' in kwargs and type(kwargs['cycle']) is int:
            setattr(self, 'cycle', kwargs['cycle'])
        else:
            setattr(self, 'cycle', 0)
            
        if 'focus' in kwargs:
            setattr(self, 'focus', kwargs['focus'])
        else:
            setattr(self, 'focus', 'pos')

        if 'agent_dict' in kwargs and type(kwargs['agent_dict']) is dict:
            setattr(self, 'agent_dict', kwargs['agent_dict'])
        else:
            setattr(self, 'agent_dict', {})

        if 'AGENT_RESET' in self.agent_dict and type(self.agent_dict['AGENT_RESET']) is bool:
            setattr(self, 'agent_reset', self.agent_dict['AGENT_RESET'])
        else:
            setattr(self, 'agent_reset', True)

        if 'AGENT_TYPE' in self.agent_dict.keys():
            self.agent_type = self.agent_dict['AGENT_TYPE']
        else:
            self.agent_type = 'mcts_empowerment'

        #currently this is a bit of a lie, since there is still a part of the data taken away, but ballpark
        if 'DATASET_SIZE' in self.agent_dict.keys():
            self.dataset_size = self.agent_dict['DATASET_SIZE']
        else:
            self.dataset_size = 10


        self.agent = self.initialization()


        self.collection_time = np.array([])

    def get_size(self, obj, seen=None):
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([self.get_size(v, seen) for v in obj.values()])
            size += sum([self.get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += self.get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([self.get_size(i, seen) for i in obj])
        return size

    def initialization(self):
        """
        generates an instance of the class of agent corresponding to the agent type
        returns the instance of the class
        """
        if self.agent_type == 'mcts':
            env = self.env
            seq_proc = self.seq_proc
            patterns = self.patterns
            agent_dict = self.agent_dict
            rng = self.rng
            agent = Mcts(env=env, seq_proc=seq_proc, patterns=patterns, agent_dict=agent_dict, rng=rng)
        elif self.agent_type == 'comb':
            env = self.env
            seq_proc = self.seq_proc
            patterns = self.patterns
            agent_dict = self.agent_dict
            rng = self.rng
            agent = Comb(env=env, seq_proc=seq_proc, patterns=patterns, agent_dict=agent_dict, rng=rng)
        elif self.agent_type == 'random':
            env = self.env
            rng = self.rng
            agent = RandomAgent(env=env)

        else:
            raise NotImplementedError('This agent type is not yet implemented')
        return agent

    def reset_agent(self):
        """
        Function that initializes the agent
        """
        self.agent = self.initialization()


    def data_collection(self):
        """
        Interaction of the agent with the environment in episodes. One episode corresponds to call of the funtion rollout.
        if 'pos' collects sequences of length MAX_LEN with a reward >= 0
        if 'neg' collects sequences of length at most MAX_LEN with a reward < 0
        Returns:
            dataset_paths (list of str) returns the locations of the datasets to be mined
        """


        #if reset is true, reset agent parameters:
        if self.agent_reset == True:
            self.agent.reset()

        #FILE PATHS: only the focus phase files are saved within their own specific folder, to structure the folders depending on the pattern the agent focuses on.
        if self.agent_type == 'comb_construction_I_max':
            path_pos = self.results_folder + self.supplementary_folder + 'pos_reward_data_cycle_' + str(
                self.cycle) + '_run_' + str(self.global_run) + '.npy'
            path_neg = self.results_folder + self.supplementary_folder + 'neg_reward_data_cycle_' + str(
                self.cycle) + '_run_' + str(self.global_run) + '.npy'
        else:
            path_pos = self.results_folder + 'pos_reward_data_cycle_' + str(self.cycle) + '_run_' + str(
                self.global_run) + '.npy'
            path_neg = self.results_folder + 'neg_reward_data_cycle_' + str(self.cycle) + '_run_' + str(
                self.global_run) + '.npy'

        #run loop over rollouts
        iter = 0
        neg_reward_observations = []
        pos_reward_observations = []
        neg_reward_hsh = []
        pos_reward_hsh = []
        num_pos_reward_observations = []
        observation_list = []
        reward_list = []
        gadget_position_list = []
        
        filename = self.results_folder+self.supplementary_folder+'collection_time_'+self.focus+'_cycle_'+str(self.cycle)+ '_run_' + str(self.global_run) + '.npy'
        #start = datetime.now()
        while True:
            observation, reward = self.agent.rollout()
            check = (observation[self.env.max_steps - 1] != np.array([0] *len(self.env.action_labels[0]))).all()
            check_dummy = (observation == np.array([[0] * len(self.env.action_labels[0])]*(self.env.max_steps))).all()
            hsh = generate_hsh(observation)
            if self.focus == 'all':
                observation_list.append(observation)
                reward_list.append(reward)
                gadget_position_list.append(np.array(self.agent.gadget_position))

            if check_dummy and reward <0:
                neg_reward_observations.append(observation)
            elif check_dummy and reward >=0:
                pos_reward_observations.append(observation)
            elif reward <= 0 and hsh not in neg_reward_hsh:
                    neg_reward_observations.append(observation)
                    neg_reward_hsh.append(hsh)
            elif reward > 0 and check and hsh not in pos_reward_hsh:
                    pos_reward_observations.append(observation)
                    pos_reward_hsh.append(hsh)

            num_pos_reward_observations.append(len(pos_reward_hsh))
            if iter%1000 == 0:
                if self.focus == 'neg':
                    print('num of episodes:', iter,', num of unique negative oberservations:', len(neg_reward_observations))
                    np.save(path_neg, neg_reward_observations)
                    dataset_path_list = [path_neg]
                    stdout.flush()
                if self.focus == 'pos':
                    print('num of episodes:', iter,', num of unique positive oberservations:', len(pos_reward_observations))
                    np.save(path_pos, pos_reward_observations)
                    dataset_path_list = [path_pos]
                    stdout.flush()
                    path_number_pos = self.results_folder + self.supplementary_folder + 'num_pos_reward_data_cycle_' + str(self.cycle) + '_run_' + str(self.global_run) + '.npy'
                    np.save(path_number_pos, num_pos_reward_observations)

            if self.focus == 'neg' and len(neg_reward_observations) == self.dataset_size:
                print('total number of episodes run to achieve negative datasetsize:', iter)
                self.collection_time = np.append(self.collection_time, iter+1)
                np.save(filename, self.collection_time)
                break
            elif self.focus == 'pos' and len(pos_reward_observations) == self.dataset_size:
                print('total number of episodes run to achieve positive datasetsize:', iter)
                self.collection_time = np.append(self.collection_time, iter+1)
                np.save(filename, self.collection_time)
                break
            elif self.focus == 'all' and len(observation_list) == self.dataset_size:
                print('total number of episodes run to achieve positive datasetsize:', iter)
                self.collection_time = np.append(self.collection_time, iter+1)
                np.save(filename, self.collection_time)
                break

            iter += 1
        


        if self.focus == 'all':
            path_gadget_position = self.results_folder + self.supplementary_folder +'all_gadget_positions_cycle_' + str(self.cycle) + '_run_' + str(self.global_run)
            path_observation = self.results_folder + self.supplementary_folder +'focus_all_observations_cycle_' + str(self.cycle) + '_run_' + str(self.global_run) + '.npy'
            path_reward = self.results_folder + self.supplementary_folder +'focus_all_rewards_cycle_' + str(self.cycle) + '_run_' + str(self.global_run) + '.npy'

            np.save(path_observation, np.array(observation_list))
            np.save(path_reward, np.array(reward_list))
            np.save(path_gadget_position, np.array(gadget_position_list))
            dataset_path_list = [path_observation]
        elif self.focus == 'pos':
            np.save(path_pos, pos_reward_observations)
            dataset_path_list = [path_pos]
        elif self.focus == 'neg':
            np.save(path_neg, neg_reward_observations)
            dataset_path_list = [path_neg]

        #save model
        if self.agent_type != 'comb_construction_I_max':
            if self.agent.tree != None:
                np.save(self.results_folder + self.supplementary_folder + 'tree_dict_cycle_' + str(self.cycle) + '_run_' + str(self.global_run) + '.npy',[self.agent.tree])
        # each round of data collection constitutes a cycle:
        return dataset_path_list

if __name__ == "__main__":
    #setting random seed
    rng = np.random.default_rng()

    # ENVIRONMENT SPECIFIC
    # key arguments to control the environment parameters
    # dimension of the qudit
    DIM = 2
    # number of qudits
    NUM_QUDITS = 4
    # number of operations:
    MAX_OP = 6
    env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS,  max_op=MAX_OP)

    #define default patterns
    pattern = default_pattern()
    patterns = [pattern]

    #result folder path
    result_folder = 'results/cycle/'

    #global seed given to the agent to save files accordingly
    global_run = 0


    # all currently implemented agents_types, the order corresponds to the MRO:
    # Comb agent
    agent_dict = {
        'AGENT_TYPE': 'comb',
        'SCHEDULE': 'prob_const',
        'THRESHOLD_DECAY': 0.1,
        'THRESHOLD': 0.1,
        'THRESHOLD_MIN': 0.1,
        'DATASET_SIZE': 10
    }

    agent = Agent(env=env, patterns=patterns, result_folder =result_folder, agent_dict=agent_dict, rng = rng, global_run = global_run)
    agent.data_collection()

    # MCTS agent data collection
    agent_dict = {
        'AGENT_TYPE': 'mcts',
        'TREE_TYPE': 'tree',
        'EXPLORATION': 0.2,
        'DATASET_SIZE': 10,
        }
    agent = Agent(env=env, result_folder =result_folder, agent_dict = agent_dict, rng = rng, global_run = global_run)
    agent.data_collection()

    # MCTS empowerment comb agent data collection
    agent_dict = {
        'AGENT_TYPE': 'mcts_comb',
        'EXPLORATION': 0.5,
        'TREE_TYPE': 'tree',
        'START_DRAW': 0,
        'END_DRAW': 10,
        'GRAPH_FOLDER': 'results/graph',
        'SCHEDULE': 'prob_const',
        'THRESHOLD_DECAY': 0.1,
        'THRESHOLD': 0.1,
        'THRESHOLD_MIN': 0.1,
        'DATASET_SIZE': 10
        }
    agent = Agent(env=env, patterns = patterns, result_folder = result_folder, agent_dict = agent_dict, rng = rng, global_run = global_run)
    agent.data_collection()


    #Comb Construction of I Max agent
    agent_dict = {
        'AGENT_TYPE': 'comb_construction_I_max',
        'FAILURE_COUNTER_THRESHOLD': 25,
        'DATASET_SIZE': 10
        }
    agent = Agent(env=env, patterns = patterns, result_folder = result_folder, agent_dict = agent_dict, rng = rng, global_run = global_run)
    agent.data_collection()

