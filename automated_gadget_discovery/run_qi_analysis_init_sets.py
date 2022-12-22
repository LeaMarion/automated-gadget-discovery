import numpy as np
import random
from utils import get_config
from utils import render_circuit
import gym
import gym_quantum_computing


MAIN_PATH = 'results/'
EXPERIMENT = 'qi'
NUM_CONFIG = 3
NUM_AGENTS = 3
AG_LIST = [0,1,2]
config = get_config('exp_'+str(NUM_CONFIG) +'.cfg', EXPERIMENT)

dist_measure = config['classification']['SIM_MEASURE']

results_folder = MAIN_PATH + EXPERIMENT + '/exp_'+str(NUM_CONFIG) + '/'
sup_folder = 'gadget_clustering/Hamming/'


pattern_dict = np.load(results_folder + 'pattern_dict_cycle_0_runs_'+str(AG_LIST)+'.npy', allow_pickle=True).item()
patterns = [p for p in pattern_dict.values()]


#Initialize env
DIM = 2
NUM_QUDITS = 4
MAX_OP = 9
FIX_M = False
FIXED_OP = 3

env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS, max_op=MAX_OP, rand_start=False, fix_measurement=FIX_M, fixed_op=FIXED_OP)

cut_to_full_label = {}
#create entries.
for label in env.action_labels:
    if str(label[:-1]) not in cut_to_full_label:
        cut_to_full_label.update({str(label[:-1]): []})
#populate entries.
for label in env.action_labels:
    cut_to_full_label[str(label[:-1])].append(label)
    
#create a dictionary from action labels to actions. (This is useful in case actions have features)
labels_to_actions =  {str(None):None}
for i, action_label in enumerate(env.action_labels):
    labels_to_actions.update({str(action_label): i})

for i, p in enumerate(patterns):
    print('\nPattern #',i, ':', p.name)
    init_set = np.ndarray.tolist(np.load(results_folder + sup_folder +'init_set_pattern_'+str(i)+'.npy', allow_pickle=True))
    
    av_length = np.mean([len(circuit) for circuit in init_set])
    print('Average circuit length', av_length)
    
    for L in range(3,9):
        print('L='+str(L)+':', np.sum([1 if len(circuit) == L else 0 for circuit in init_set]))
        

    if i == 0:
        
        #without gadget
        for circuit in init_set[1:5]:
            all_reward = []
            env.reset()
            # print('\n',circuit)
            for element in circuit:
                random.shuffle(cut_to_full_label[str(np.array(element))])
                action = cut_to_full_label[str(np.array(element))][0]
                _,reward,_,_ = env.step(labels_to_actions[str(action)])
                all_reward.append(reward)
                # print(action, all_reward)
            
            render_circuit(circuit, reward, 4, len(circuit))
            
