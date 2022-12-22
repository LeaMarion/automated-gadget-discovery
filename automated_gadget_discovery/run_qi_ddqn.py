import numpy as np
import random
import torch
import sys
import argparse
import gym
import gym_quantum_computing
from sys import argv, stdout
import pathlib
import copy
from utils import *
import time
from datetime import datetime
from itertools import combinations
import hdbscan

from agents.ddqn import DDQN
from data_mining.mining import Mining
from data_mining.pattern import *
from data_mining.pattern_manager import PatternManager
from data_mining.pattern_evaluation import PatternEvaluation
from data_mining.seq_processor import SeqProcessor
from agents.jiminy import Jiminy
from sim_distance import SimDistance

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Type of device')
    parser.add_argument('--experiment', type=str, default='qi', help='Name of experiment')
    parser.add_argument('--num_config', type=int, default=0, help='Name of configuration file')
    parser.add_argument('--run', type=int, default=0, help='Seed for reproduction')
    args = parser.parse_args(argv)
    return args



args = get_args(sys.argv[1:])

config_name = 'exp_'+str(args.num_config)
run = args.run
experiment_name = args.experiment

MyGPU = args.device
print(MyGPU)
stdout.flush()
device = torch.device(MyGPU)
config = get_config(config_name+'.cfg', experiment_name)
#set number of cpu threads
torch.set_num_threads(1)

# path to the result folder
results_folder = 'results/' + experiment_name+ '/' + config_name + '/'
print(results_folder)

#create folders if necessary
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.deterministic = True


#----------methods----------------#

def state_one_hot_encoding(env):
    encoded_state = np.array([])
    for action in env.action_hist:
        if action == None:
            encoded_state = np.append(encoded_state,[0]*len(env.actions_one_hot_encoding[0]))
        else:
            encoded_state = np.append(encoded_state, env.actions_one_hot_encoding[action])
    encoded_state = encoded_state.flatten().astype(np.float32)
    observation = torch.tensor(encoded_state,dtype=torch.float,device=device)
    return observation

def exploration(cycle_num, agent, patterns):
    """
    Exploration phase within the global cycle.

    Parameters
    ----------
    cycle_num : int
        Number identifying the current cycle.
        
    agent: class with the agent (a trained agent can be loaded here, to start its learning process on a different exploration task).
    
    patterns : List
        List with patterns (classes) given to the agent for the exploration. If not empty, the agent guides the exploration using these patterns.
    
    Returns
    -------
    data_paths : List
        List with the paths to the data sets to be mined.

    """
    
    #initialize the Jiminy class for this cycle:
    INT_RESCALE = config['jiminy']['INT_RESCALE']
    EXT_RESCALE = config['jiminy']['EXT_RESCALE']
    REWARD_FCT = config['jiminy']['REWARD_FCT']
    
    jiminy = Jiminy(patterns, seq_processor, intrinsic_rescale=INT_RESCALE, extrinsic_rescale=EXT_RESCALE)
    
    #get folder to save results from this cycle's exploration
    pathlib.Path(results_folder+'exploration/').mkdir(parents=True, exist_ok=True)
    path_obs = results_folder+'data_cycle_'+str(cycle_num)+'_run_'+str(run)+'.npy'
    path_rwd = results_folder+'exploration/reward_cycle_'+str(cycle_num)+'_run_'+str(run)+'.npy'

    #get folder to save modules from this cycle's exploration
    # pathlib.Path(results_folder+'models/').mkdir(parents=True, exist_ok=True)
    
    #training
    if cycle_num > 0 and config['agent']['RESET_AFTER_MINING']: 
        agent = DDQN(config, env.num_actions, env.max_steps*len(env.actions_one_hot_encoding[0]), device)


    # loading model
    if cycle_num > 0 and config['agent']['LOAD_MODEL']:
    #if config['agent']['LOAD_MODEL']:
        #load_folder = results_folder
        load_folder = 'results/ddqn_cycle_load_model/exp_3/'
        path_model = load_folder +'models/model_cycle_'+str(cycle_num-1)+'_run_'+str(run)+'.npy'
        path_optim = load_folder+'models/optim_cycle_'+str(cycle_num-1)+'_run_'+str(run)+'.npy'
        agent.policy_net.load_state_dict(torch.load(path_model))
        agent.target_net.load_state_dict(torch.load(path_model))
        agent.optim.load_state_dict(torch.load(path_optim))
        print(agent.optim)
        agent.policy_net.eval()
        agent.target_net.eval()


    reward_list = []
    for e in range(1,NUM_EPISODES+1):
        state = env.reset()
        observation = state_one_hot_encoding(env)
        agent.policy_net.train()
        done = False
        while not done:
            action, eps = agent.act(observation)
            next_state, reward, done, _ = env.step(action)
            
            #rescaling of the extrinsic reward and evaluation of the intrinsic reward.
            reward = jiminy.extrinsic_reward_rescaling(reward, done, reward_fct=REWARD_FCT)
            # reward = jiminy.reward_modulation(next_state, reward)
                
            next_observation = state_one_hot_encoding(env)
            agent.remember(observation,
                           torch.tensor(action, device=device),
                           torch.tensor(reward, device=device),
                           next_observation,
                           torch.tensor(done, device=device,dtype=int))
            observation = next_observation
            if len(agent.memory) > config['agent']['BATCH_SIZE']:
                loss = agent.replay(config['agent']['BATCH_SIZE'])
                    
        if reward > 0:
            score = 1.
            reward_list.append(score)
        else:
            score = 0.
            reward_list.append(score)
        
        if e % 1000 == 0:
            stdout.flush()
            np.save(path_rwd, reward_list)
            
    #data_collection
    data_list = []
    #hsh list of each unique data sample
    hsh_list = []
    counter = 0
    while len(hsh_list) < DATACOLLECTION_SIZE:
        state = env.reset()
        observation = state_one_hot_encoding(env)
        agent.policy_net.train()
        done = False
        while not done:
            action, eps = agent.act(observation)
            next_state, reward, done, _ = env.step(action)
            # reward = jiminy.reward_modulation(next_state, reward)
            next_observation = state_one_hot_encoding(env)
            observation = next_observation

        hsh = generate_hsh(next_state)
        
        reward_list.append(reward)

        #checking conditions if the data_sample is added to the data list
        if hsh not in hsh_list and reward>0:
            data_list.append(next_state)
            hsh_list.append(hsh)
        
            
        counter += 1
        if counter % 20000 == 0:
            np.save(path_obs, data_list)
            print(len(hsh_list), ' unique positively (considering intrinsic reward too) rewarded sequences found in ', counter, ' episodes.')
        if counter/DATACOLLECTION_SIZE >= 2:
            print('Data collection has been interrupted after ',counter, 'episodes. ', len(hsh_list), ' sequences have been saved.')
            break
        
    np.save(path_rwd, reward_list)
    np.save(path_obs, data_list)
    # torch.save(agent.policy_net.state_dict(),
    #            results_folder+'models/policy_cycle_'+str(cycle_num)+'_run_'+str(run)+'.npy')
    # torch.save(agent.policy_net.state_dict(),
    #            results_folder+'models/target_cycle_'+str(cycle_num)+'_run_'+str(run)+'.npy')
    # torch.save(agent.optim.state_dict(),
    #            results_folder+'models/optim_cycle_'+str(cycle_num)+'_run_'+str(run)+'.npy')
        
    return path_obs

def seq_mining(cycle_num, data_paths, number_patterns=None):
    """
    Sequence mining phase within the global cycle.

    Parameters
    ----------
    cycle_num : int
        Number identifying the current cycle.
    data_paths : List
        List with the paths to the data sets to be mined.
    number_patterns : int, optional
        Number of top best mined patterns that will be added to the main dictionary and output for the agent to use.
        The default is None (all patterns output by the seqmin will be considered in the following).
    
    Returns
    -------
    patterns : List
        List with patterns (classes) given to the agent for the next exploration.

    """
    #initialize Mining class
    mining_dict = config['mining']
    
    mining = Mining(data_paths, results_folder, mining_dict=mining_dict, supplementary_folder='sequence_mining/', cycle=cycle_num, run=run, seq_proc=seq_processor)
    
    #sequence mining 
    rule_output = mining.mining()
    
    print('Patterns obtained in cycle ', cycle_num, ':\n', [p for p in rule_output.keys()])
    
    #send patterns output by the seqmin to the pattern manager for processing.
    patterns = pattern_manager.pattern_processing(list(rule_output.values()), number_patterns=number_patterns, cycle=cycle_num)
    
    return patterns

#-------------------------------------------#



""" Environment and Agent initialization"""
# ENVIRONMENT SPECIFIC
# arguments to define the environment
# dimension of the qudit
DIM = config['environment']['DIM']
# number of qudits
NUM_QUDITS = config['environment']['NUM_QUDITS']
# number of operations
MAX_OP = config['environment']['MAX_OP']
# if there is a measurement in each starting state
FIX_M = config['environment']['FIX_M']
# if rand_start True fixed_op=n determines that every nth element is fixed
FIXED_OP = config['environment']['FIXED_OP']

env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS, max_op=MAX_OP, rand_start=True, fix_measurement=FIX_M, fixed_op=FIXED_OP)
        
agent = DDQN(config, env.num_actions, env.max_steps*len(env.actions_one_hot_encoding[0]), device)


""" Initialization of SequenceProcessor, PatternManager """
#initialize sequence processor
ROOT_LENGTH = config['processor']['ROOT_LENGTH']
FEATURES_REMOVED = config['processor']['FEATURES_REMOVED']
NUM_FEATURES = len(env.action_labels[0])
SEQ_LENGTH = len(env.reset())

max_elements = [0]*NUM_FEATURES 
for feature in range(NUM_FEATURES):
    max_elements[feature] = max([label[feature] for label in env.action_labels])
 

if env.rand_start:
    INDICES_TO_MINE = [index for index in range(env.max_steps) if index not in list(range(env.max_steps))[env.fixed_op-1::env.fixed_op]]
else:
    INDICES_TO_MINE = None


seq_processor = SeqProcessor(root_length=ROOT_LENGTH, action_labels=env.action_labels,\
                             features_removed=FEATURES_REMOVED, indices_to_mine=INDICES_TO_MINE, num_features=NUM_FEATURES,\
                                 sequence_length=SEQ_LENGTH, max_value_features=max_elements)
    
  
#initialize pattern manager
pattern_manager = PatternManager(features_removed=seq_processor.features_removed)

#get general parameters
NUM_CYCLES = config["general"]["CYCLES"]
NUM_EPISODES = config["general"]["EPISODES"][0]
DATACOLLECTION_SIZE = config["general"]["DATACOLLECTION_SIZE"]
WITH_CLASSIFICATION = config["general"]["WITH_CLASSIFICATION"]

# measure runtime part 1
start_time = datetime.now()

#run cycles
for n in range(NUM_CYCLES):
    data_paths = exploration(n, agent, list(pattern_manager.pattern_dict.values()))
    print('Exploration is finished in cycle',n)
    patterns = seq_mining(n, data_paths)
    print('Sequence mining is finished in cycle',n)
    #measure runtime part 2
    cycle_time = datetime.now()
    print('The runtime after cycle number '+str(n)+' is: ',cycle_time-start_time)
    stdout.flush()



#save the final dictionary with all the patterns and their classes
np.save(results_folder + 'pattern_dict_run_'+str(run)+'.npy',pattern_manager.pattern_dict)

patterns = list(pattern_manager.pattern_dict.values())

""" Initialization and run (if applicable) of the classification """

if WITH_CLASSIFICATION and patterns:
    
    #Initialize similarity measure
    INDICES_ENV = [2,5,8]
    SIM_MEASURE = config["classification"]["SIM_MEASURE"]
    MIN_SAMPLES = config["classification"]["MIN_SAMPLES"]
    INITSET_SIZE = config["classification"]["INITSET_SIZE"]
    WEIGHTS = config["classification"]["WEIGHTS"]
    sup_folder = 'gadget_clustering/'+SIM_MEASURE+'/'
    
    #populate and get the dictionary with the one-hot encoding of each featured action.
    one_hot_dict = {}
    for action, encoding in enumerate(env.actions_one_hot_encoding):
        one_hot_dict[str(list(env.action_labels[action])[:-FEATURES_REMOVED])] = encoding[:-4]
        
    sim = SimDistance(seq_proc=seq_processor, patterns=patterns, results_folder=results_folder, supplementary_folder=sup_folder, run=run,\
                      indices_env_placements=INDICES_ENV, sim_measure=SIM_MEASURE, one_hot_dict=one_hot_dict, max_initset_size=INITSET_SIZE, weights_bit_blocks=WEIGHTS)
    sim.distance_matrix_population() #compute distance matrix
    
    #initialize clusterer
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=MIN_SAMPLES)
    clusterer.fit(sim.pattern_distance_matrix)
    labels = clusterer.labels_
    prob = clusterer.probabilities_
    
    np.save(results_folder + sup_folder + 'labels_run_{}.npy'.format(run), labels)
    np.save(results_folder + sup_folder + 'prob_labels_run_{}.npy'.format(run), prob)
        
elif WITH_CLASSIFICATION and not patterns:
    print('There are not patterns to classify.')
    
