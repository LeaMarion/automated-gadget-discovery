import numpy as np
import random
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


from data_mining.pattern import *
from data_mining.pattern_manager import PatternManager
from data_mining.pattern_evaluation import PatternEvaluation
from data_mining.seq_processor import SeqProcessor

from clustering.sim_distance import SimDistance

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Type of device')
    parser.add_argument('--experiment', type=str, default='qi', help='Name of experiment')
    parser.add_argument('--num_config', type=int, default=0, help='Name of configuration file')
    parser.add_argument('--num_agents', type=int, default=3, help='Number of agents from which to collect gadgets')
    parser.add_argument('--clustering_method', type=str, default='utility', help='Clustering method, either utility or context')
    args = parser.parse_args(argv)
    return args



args = get_args(sys.argv[1:])

config_name = 'exp_'+str(args.num_config)
experiment_name = args.experiment
NUM_AGENTS = args.num_agents
CLUSTERING_METHOD = args.clustering_method

print(config_name)

config = get_config(config_name+'.cfg', experiment_name)
# path to the result folder
# results_folder = 'results/' + experiment_name+ '/' + config_name + '/'
results_folder = 'results/' + experiment_name+ '/' + config_name + '/'
print(results_folder)

#create folders if necessary
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

stdout.flush()

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

for ag in range(NUM_AGENTS):
    rule_output = np.load(results_folder + 'sequence_mining/' +'final_rules_cycle_0_run_'+str(ag)+'.npy', allow_pickle=True).item()  
    #add patterns to the pattern manager.
    pattern_manager.pattern_processing(list(rule_output.values()), agent=ag)
    
#save the final dictionary with all the patterns and their classes
np.save(results_folder + 'pattern_dict_cycle_0_runs_'+str([ag for ag in range(NUM_AGENTS)])+'.npy',pattern_manager.pattern_dict)

print('\nClustering gadgets from file: ', results_folder + 'pattern_dict_cycle_0_runs_'+str([ag for ag in range(NUM_AGENTS)])+'.npy', '\n')

patterns = list(pattern_manager.pattern_dict.values())

""" Initialization and run (if applicable) of the classification """


if CLUSTERING_METHOD == 'context' and patterns:
    
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
        
    sim = SimDistance(seq_proc=seq_processor, patterns=patterns, results_folder=results_folder, supplementary_folder=sup_folder, \
                      indices_env_placements=INDICES_ENV, sim_measure=SIM_MEASURE, one_hot_dict=one_hot_dict, max_initset_size=INITSET_SIZE, weights_bit_blocks=WEIGHTS)
    sim.distance_matrix_population() #compute distance matrix
    
    #initialize clusterer
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=MIN_SAMPLES)
    clusterer.fit(sim.pattern_distance_matrix)
    labels = clusterer.labels_
    prob = clusterer.probabilities_
    
    np.save(results_folder + sup_folder + 'labels_'+SIM_MEASURE+'.npy', labels)
    np.save(results_folder + sup_folder + 'prob_labels_'+SIM_MEASURE+'.npy', prob)

elif CLUSTERING_METHOD == 'utility' and patterns:
    
    sup_folder = 'gadget_clustering/utility/'
    MIN_SAMPLES = config["classification"]["MIN_SAMPLES"]
    
    sim = SimDistance(seq_proc=seq_processor, patterns=patterns, results_folder=results_folder, supplementary_folder=sup_folder)
    
    #initialize clusterer
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=MIN_SAMPLES)
    
    #utility based clustering
    clusterer.fit(sim.utility_based_distance())

    labels = clusterer.labels_
    prob = clusterer.probabilities_
   
    np.save(results_folder + sup_folder + 'labels_utility.npy', labels)
    np.save(results_folder + sup_folder + 'prob_labels_utility.npy', prob)