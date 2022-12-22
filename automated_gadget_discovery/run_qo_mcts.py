from agent import Agent
from data_mining.mining import Mining
from data_mining.pattern_manager import PatternManager
from data_mining.pattern_evaluation import PatternEvaluation
from data_mining.seq_processor import SeqProcessor
from data_mining.pattern import *
import gym
from sys import argv, stdout
import os, sys
import argparse
from utils import *
import warnings
import numpy as np
import concurrent.futures
from datetime import datetime
from itertools import product, combinations
from PIL import Image, ImageDraw, ImageFont
import pathlib
import gym_quantum_computing
#import gym_entangled_ions
from itertools import product


#sys.path.append(os.path.join(os.path.dirname(__file__), '/scratch/c7051107/gym-optical-tables/gym_optical_tables/envs/'))
#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/leamt/gym-optical-tables/gym_optical_tables/envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/leamarion/Git/gym-optical-tables/gym_optical_tables/envs'))
from optical_tables_env import OpticalTablesEnv



def get_args(argv):
    """
    Passes the arguments from the runfile to the main file or passes the default values to the main file.
    Returns a namespace object that makes each element callable by args.name.
    Args:
        argv  (list) list of arguments that are passed through with a --name

    Returns:
        args  (namespace) namespace of all arguments passed through
    """
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='qo', help='name of the experiment')
    parser.add_argument('--num_config', type=str, default=0, help='number of the configuration file')
    parser.add_argument('--run', type=int, default=0, help='how often rerun to reproduce the result')
    args = parser.parse_args(argv)
    return args

def load_environment(env_dict):
    """
    Loads the gym environment with the corresponding dictionary specified in the config file
    Args  env_dict (dict): dict with all the environment hyperparameters
    Returns: env (object) the gym environment that
    """
    name = env_dict['NAME']
    if name == 'quantum-computing-v0':
        # dimension of the qudit
        DIM = env_dict['DIM']
        # number of qudits
        NUM_QUDITS = env_dict['NUM_QUDITS']
        # number of operations:
        MAX_OP = env_dict['MAX_OP']
        # environment rand start
        RAND_START = env_dict['RAND_START']
        env = gym.make(name, dim=DIM, num_qudits=NUM_QUDITS, max_op=MAX_OP, rand_start = RAND_START)
        num_features = len(env.action_labels[0])
        setattr(env, 'num_features', num_features)
        return env

    elif name == 'optical-tables-v0':
        ENV_PARAMS = env_dict['ENV_PARAMS']
        env = OpticalTablesEnv(**ENV_PARAMS)
        setattr(env, 'rand_start', False)
        setattr(env, 'max_steps', env.episode_length)
        print(env.action_space)
        if not hasattr(env, 'action_labels'):
            label_list = [np.array([x+1]) for x in range(env.num_actions)]
            setattr(env, 'action_labels', label_list)
            action_list = list(range(env.num_actions))
            setattr(env, 'action_list', action_list)
            num_features = len(label_list[0])
            setattr(env, 'num_features', num_features)
        else:
            action_list = list(range(env.num_actions))
            setattr(env, 'action_list', action_list)
            num_features = len(env.action_labels[0])
            setattr(env, 'num_features', num_features)
        return env

    else:
        raise NotImplementedError("This environment is not yet implemented")




def exploration(cycle_num, patterns, agent_num=0):
    """
    Exploration phase within the global cycle.

    Parameters
    ----------
    cycle_num : int
        Number identifying the current cycle.
    patterns : List
        List with patterns (classes) given to the agent for the exploration. If not empty, the agent guides the exploration using these patterns.
    agent_num : int, optional
        Number of the configuration for the given agent (changed in case we want to have different configs for the different cycles). The default is 0.

    Returns
    -------
    data_paths : List
        List with the paths to the data sets to be mined.

    """
    
    if cycle_num == 0:
        #initialize agent that explores for the first time.
        agent = Agent(env=env, cycle=cycle_num, agent_dict=config['exploration']['agent_'+str(agent_num)], results_folder=results_folder,\
                      supplementary_folder='exploration/', patterns=[], seq_proc=seq_processor, rng=rng, global_run=RUN)
        #agent collects data while interacting with the environment
        data_paths = agent.data_collection()

    else:
        #initialize agent that explores using the information of previous cycles (guided exploration).
        agent = Agent(env=env, cycle=cycle_num, agent_dict=config['exploration']['agent_'+str(agent_num)], results_folder=results_folder,\
                      supplementary_folder='exploration/', patterns=patterns, seq_proc=seq_processor, rng=rng, global_run=RUN)
        #agent collects data while interacting with the environment
        data_paths = agent.data_collection()
        
    return data_paths

def seq_mining(cycle_num, data_paths, number_patterns=None, config_num=0):
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
    config_num : int, optional
        Number of the configuration (changed in case we want to have different mining configs for the different cycles).
        The default is 0.

    Returns
    -------
    patterns : List
        List with patterns (classes) given to the agent for the focus phase and the next exploration.

    """
    #initialize Mining class
    mining_dict = config['mining']

    print(env)
    mining = Mining(data_paths, results_folder, mining_dict=mining_dict, supplementary_folder='sequence_mining/', cycle=cycle_num, run=RUN, env=env, seq_proc=seq_processor)
    #sequence mining 

    rule_output = mining.mining()
    print(rule_output)
    #send patterns output by the seqmin to the pattern manager for processing.
    patterns = pattern_manager.pattern_processing(list(rule_output.values()), number_patterns=number_patterns, cycle=cycle_num)
    return patterns

def focus(global_cycle_num, patterns, root_length, features_removed, agent_num=0):
    """
    Prepares the parameters and runs the focus phase in parallel processes for the different patterns.

    Parameters
    ----------
    global_cycle_num : int
        Number identifying the current global cycle.
    patterns : List
        List with patterns (classes) from the mining, given to the agent for the focus phase.
    agent_num : int, optional
        Number of the configuration for the given agent (changed in case we want to have different configs for the different cycles). 
        The default is 0.

    Returns
    -------
    evaluation : gen
        Results from the focus phase (several runs of the focus phase for each pattern).

    """
    
    #generate a random seed for any concurrent processes
    seed_seq = rng.bit_generator._seed_seq
    child_states = seed_seq.spawn(len(patterns))
    
    # num_cycles for all concurrent processes. How many times the focus phase is repeated for each pattern.
    num_runs_focus = config['focus']['agent_'+str(agent_num)]['NUM_RUNS_FOCUS']
    runs_focus_list = [num_runs_focus]*len(patterns)
    
    #information for the global process (this refers to the full cycle exploration-sm-focus)
    global_cycle_num_list = [global_cycle_num]*len(patterns)
    global_run_list = [RUN]*len(patterns)

    #parallel processing of the patterns
    with concurrent.futures.ProcessPoolExecutor() as executor:
        evaluation = executor.map(cycle_focus_phase, runs_focus_list, global_cycle_num_list, patterns, child_states, global_run_list)
    return evaluation


def cycle_focus_phase(num_runs_focus, global_cycle_num, pattern, seed, global_run, agent_num=0):
    """
    Runs the focus phase (several runs) for one pattern.

    Parameters
    ----------
    num_runs_focus : int
        Total number of runs for each pattern.
    global_cycle_num : int
        Number identifying the current global cycle.
    pattern : Class
        Class containing the information of the pattern that is being evaluated.
    seed : np.random._bit_generator.SeedSequence
        Generator of the seed for each run of the focus phase ('pos' and 'neg' run with the same generator).
    global_run : int
        Indicates the global agent number (run number of the entire interaction process).
    agent_num : int, optional
        Number of the configuration for the given agent (changed in case we want to have different configs for the different cycles). 
        The default is 0.
        
    Returns
    -------
    evaluation_list : List (of lists, 2 levels)
        List containing the info from the focus phase of one pattern.
        Each entry of the list contains the data (with the form [pattern_name, I, F, C, 'pos'/'neg']) of one run of the focus phase.

    """

    rng = np.random.default_rng(seed)
    seed_seq = rng.bit_generator._seed_seq
    #spawn seeds to generate independent cycles
    child_states = seed_seq.spawn(num_runs_focus)
    #initialize list to store results from all the focus cycles.
    evaluation_list = []
    for i in range(num_runs_focus):
        for pos_neg_focus in ['pos','neg']:
            #same seed for both positive and negative sets
            rng = np.random.default_rng(child_states[i])
            #initialize focus agent
            focus = Agent(env=env, patterns=[pattern], cycle=global_cycle_num, agent_dict=config['focus']['agent_' + str(agent_num)], focus=pos_neg_focus,\
                          results_folder=results_folder, supplementary_folder='focus/pattern_'+str(pattern.id)+'/', seq_proc=seq_processor, rng=rng, global_run=global_run)
            # agent collects dataset
            data_focus_path_list = focus.data_collection()
            # evaluation of the patterns in pattern_list with respect to the set obtained in the focus phase.


            pattern_evaluation = PatternEvaluation(seq_proc=seq_processor, pattern_list=[pattern], file_path=data_focus_path_list[0])
            result_evaluation = pattern_evaluation.evaluation(pos_neg_focus)[0]
            #store the results of this focus cycle for the given pattern.
            evaluation_list.append(result_evaluation)

    return evaluation_list

def evaluation_processing(cycle_num, evaluation):
    """
    Process all the data (all patterns and all focus runs) from the focus phase.
    Calls the pattern_manager to update the information of the stored patterns and include focus_I and its std.
    
    Parameters
    ----------
    cycle_num : int
        Number identifying the current global cycle.
    evaluation : gen
        Output from the method focus().

    """
    #extract and store data from the focus process in a data dictionary.
    data_dict = {}
    for ele in evaluation:
        for i in range(len(ele)):
            pattern = ele[i][0]
            values = ele[i][1:4]
            pos_neg = ele[i][4]
            if str(pattern) in data_dict.keys():
                if pos_neg == 'pos':
                    data_dict[str(pattern)]['pos'] = np.append(data_dict[str(pattern)]['pos'], values[0])
                if pos_neg == 'neg':
                    data_dict[str(pattern)]['neg'] = np.append(data_dict[str(pattern)]['neg'], values[0])
    
            else:
                data_dict.update({str(pattern): {'pos':np.array([]), 'neg':np.array([])}})
                if pos_neg == 'pos':
                    data_dict[str(pattern)]['pos'] = np.append(data_dict[str(pattern)]['pos'], values[0])
                if pos_neg == 'neg':
                    data_dict[str(pattern)]['neg'] = np.append(data_dict[str(pattern)]['neg'], values[0])
    
    #save data_dict in supplementary folder of the focus phase
    np.save(results_folder + 'focus/' + 'I_arrays_dict_cycle_{}_run_{}.npy'.format(cycle_num, RUN), data_dict)
    
    #update the pattern manager dictionary with this data 
    for pattern in data_dict.keys():
        pattern_manager.focus_update(pattern, data_dict[str(pattern)]['pos'], data_dict[str(pattern)]['neg'])

def render_circuit(circuit = [], num_circuit = 0, num_qudits = 0, num_ops = 6, num_cycle = 0):
    pattern_eval = PatternEvaluation(seq_proc = seq_processor, env = env)
    reward, info = pattern_eval.check_reward(circuit)


    width = num_ops * 50 + 100
    height = num_qudits * 50 + 100
    im = Image.new('RGB', (width, height), (194, 194, 198))
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), str(info), (255, 255, 255))
    draw.polygon(((0, 0), (width, height), (width, 0)), fill=(194, 194, 198))
    draw.polygon(((0, 0), (width, height), (0, height)), fill=(194, 194, 198))

    if reward == 1.0:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)


    x_pos = 60 + 50 * num_ops
    y_pos = 10
    draw.rectangle((x_pos, y_pos, x_pos + 30, y_pos + 30), fill=color)
    y_start = 70
    for i in range(num_qudits):
        draw.line(((0, y_start + 10 + 50 * i), (width, y_start + 10 + 50 * i)), fill=(255, 255, 255), width=5)


    elements = [env.read_toolbox()[element] for j, element in enumerate(circuit)]
    print(elements)

    for j, ele in enumerate(elements):
        x_pos = 50 + 50 * j
        if ele[0] == 'H':
            color = (255, 122, 122)
            pos_1 = (int(ele[5])-1)*50
            draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

        elif ele[0] == 'B':
            pos_1 = (int(ele[3])-1)*50
            pos_2 = (int(ele[6])-1)*50
            color = (95, 173, 233)
            draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)
            draw.ellipse((x_pos, y_start + pos_2, 20 + x_pos, y_start + 20 + pos_2), fill=color)
            draw.line(((x_pos + 10, y_start + 10 + pos_1), (x_pos + 10, y_start + 10 + pos_2)), fill=color, width=5)

        elif ele[0] == 'D':
            color = (242, 206, 121)
            pos_1 = (int(ele[3])-1) * 50
            draw.rectangle((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

        elif ele[0] == 'R':
            color = (133, 246, 159)
            pos_1 = (int(ele[5])-1) * 50
            draw.rectangle((x_pos, y_start + pos_1 - 20, 20 + x_pos, y_start + 40 + pos_1), fill=color)
    im.save(results_folder+"cycle_"+str(num_cycle)+"_"+str(info)+"_circuit_num_"+str(num_circuit)+".jpg")
    #im.show()
    

if __name__ == "__main__":
    #get the experiment name, the configuration file name and the seed.
    args = get_args(argv[1:])
    EXPERIMENT = args.experiment
    CONFIG = 'exp_'+str(args.num_config)
    RUN = args.run
    RENDER_GADGET = False
    print('EXPERIMENT:',CONFIG,RUN)
    stdout.flush()
    
    
    #path to the result folder
    results_folder = 'results/'+EXPERIMENT+'/'+CONFIG+'/'#+'/run_'+str(RUN)+'/'
    
    #load the configuration file
    config = get_config(CONFIG+'.cfg',EXPERIMENT)
    
    # set seed
    rng = np.random.default_rng()
    
    #LOAD ENVIRONMENT
    env_dict = config['environment']
    env = load_environment(env_dict)

    
    #CYCLE CONFIGURATIONS
    NUM_CYCLES = config['general']['CYCLES']

    config_num = 0
    
    #initialize sequence processor
    # initialize sequence processor
    ROOT_LENGTH = config['processor']['ROOT_LENGTH']
    FEATURES_REMOVED = config['processor']['FEATURES_REMOVED']
    NUM_FEATURES = len(env.action_labels[0])
    SEQ_LENGTH = env.max_steps

    max_elements = [0] * NUM_FEATURES
    for feature in range(NUM_FEATURES):
        max_elements[feature] = max([label[feature] for label in env.action_labels])

    if env.rand_start:
        INDICES_TO_MINE = [index for index in range(env.max_steps) if
                           index not in list(range(env.max_steps))[env.fixed_op - 1::env.fixed_op]]
    else:
        INDICES_TO_MINE = None

    seq_processor = SeqProcessor(root_length=ROOT_LENGTH, action_labels=env.action_labels, \
                                 features_removed=FEATURES_REMOVED, indices_to_mine=INDICES_TO_MINE,
                                 num_features=NUM_FEATURES, \
                                 sequence_length=SEQ_LENGTH, max_value_features=max_elements)

    # initialize pattern manager
    pattern_manager = PatternManager(features_removed=seq_processor.features_removed)

    # measure runtime part 1
    now = datetime.now()


    for n in range(0,NUM_CYCLES):
        data_paths = exploration(n, list(pattern_manager.pattern_dict.values()))
        print('Exploration is finished in cycle',n)
        patterns = seq_mining(n, data_paths[0])
        print('Sequence mining is finished in cycle',n)


    #save the final dictionary with all the patterns and their classes
    np.save(results_folder + 'pattern_dict_run'+str(RUN)+'.npy',pattern_manager.pattern_dict)

    # measure runtime part 2
    later = datetime.now()
    print(later-now)
    stdout.flush()

    if RENDER_GADGET:
        for i, pattern in enumerate(pattern_manager.pattern_dict.items()):
            circuit = [int(elem-1) for elem in pattern[1].action_list]
            render_circuit(circuit = circuit, num_circuit= i, num_qudits=4, num_ops=len(circuit), num_cycle= n)

