import numpy as np
import itertools
import gym
import gym_quantum_computing

from data_mining.pattern_manager import PatternManager
from data_mining.seq_processor import SeqProcessor


class Jiminy(object):

    def __init__(self, patterns, seq_proc, **kwargs):
        """
        Class that computes the intrinsic reward to guide the agent in its search. 
        It also contains methods to rescale the extrinsic reward. 
        
        patterns (list) : patterns that are to be consider in the guidance. Each element in this list is an instance of the class Pattern.
        seq_proc: class with the sequence processor.
        
        **kwargs
        intrinsic_rescale (int) : negative integer (default -10). Rescales the reward if the agent creates a sequence that contains a pattern from self.patterns.
        extrinsic_rescale (int): positive integer (default 5). Rescales the extrinsic reward.
        
        """
        
        if 'intrinsic_rescale' in kwargs and type(kwargs['intrinsic_rescale']) is int:
            setattr(self, 'intrinsic_rescale', kwargs['intrinsic_rescale'])
        else:
            setattr(self, 'intrinsic_rescale', -10)
        
        if 'extrinsic_rescale' in kwargs and type(kwargs['extrinsic_rescale']) is int:
            setattr(self, 'extrinsic_rescale', kwargs['extrinsic_rescale'])
        else:
            setattr(self, 'extrinsic_rescale', 5)
        
        
        if len(patterns) != 0:
            #extract each pattern as an np.array with the environment notation.
            self.patterns = [pattern.action_list for pattern in patterns]
            self.cohesion_list = [pattern.C for pattern in patterns]
        else:
            self.patterns = patterns
        #sequence processor
        self.seq_proc = seq_proc
        
    
    def intrinsic_reward(self, sequence):
        """
        Computes the intrinsic reward.

        Parameters
        ----------
        sequence : (np.array or list). Sequence of actions to be analysed.

        Returns
        -------
        None: no patterns were given to compute the intrinsic reward.
        1: the given sequence does not contain any of the patterns.
        intrinsic_rescale: negative value (float), indicates that the sequence contains at least one of the patterns.

        """
        if self.patterns != []:
            for index, pattern in enumerate(self.patterns):
                if self.pattern_in_sequence(pattern, sequence):
                    if self.oneseq_cohesion(pattern, sequence) >= self.cohesion_list[index]:
                        return 1.*self.intrinsic_rescale
            return 1.
                
        else:
            return None
    
    def reward_modulation(self, sequence, reward):
        """
        Computes the final reward.

        Parameters
        ----------
        sequence : (np.array or list). Sequence of actions to be analysed.
        reward : (float). Extrinsic reward, given by the env. for the input sequence.
            
        Returns
        -------
        Final reward (float), combining the extrinsic and the intrinsic reward.
            
        """
        irwd = self.intrinsic_reward(sequence)
        
        if reward <= 0 and irwd: #the reward that gives a stronger penalty is selected.
            return min(reward, irwd)
        elif reward > 0 and irwd: #extrinsic reward if patterns are not in the sequence, intrinsic otherwise.
            print('STUPID!!')
            return reward if irwd == 1. else irwd

        else:
            return reward
        
    def extrinsic_reward_rescaling(self, reward, done, reward_fct='unscaled'):
        """
        Rescales the extrinsic reward.

        Parameters
        ----------
        reward_fct : (str). Choice of rescaling. It can be: final, unscaled, intermediate, lopsided, negative_weighted.
            
        reward : (int). The reward given by the env.
            
        done : (int). Signal from the env that indicates whether the sequence (episode) is finished.

        Returns
        -------
        Rescaled reward.
        
        """
        if reward_fct == 'final':
            return self.final_reward(reward, done)
        elif reward_fct == 'unscaled':
            return reward
        elif reward_fct == 'intermediate':
            return self.lower_intermediate_reward(reward, done)
        elif reward_fct == 'lopsided':
            return self.lopsided(reward, done)
        elif reward_fct == 'negative_weighted':
            return self.negative_weighted(reward, done)
        else:
            raise NotImplementedError('Reward function not yet implemented')
            
    #-------helper methods -------
    
    def pattern_in_sequence(self, pattern, sequence):
        """
        Checks whether the pattern is contained in the sequence or not.
        
        Parameters
        ----------
        pattern : (list of np.arrays)
            
        sequence : np.array. Sequence to be analyzed.

        Returns
        -------
        True/False: whether the pattern is contained in the sequence.

        """
        seq = np.ndarray.tolist(self.seq_proc.feature_removal([sequence])[0])
        
        P = np.ndarray.tolist(self.seq_proc.feature_removal([np.array(pattern)])[0])

        pos = 0
        for element in seq:
            if pos < len(P) and element == P[pos]:
                pos += 1
        
        return pos == len(P)
    
    def oneseq_cohesion(self, pattern, sequence):
        """
        Computes the cohesion of a pattern considering only one sequence.

        Parameters
        ----------
        pattern : (list of np.arrays)
            
        sequence : (list or np.array). Sequence that contains the pattern.

        Returns
        -------
        Cohesion (float) : cohesion of the pattern in the given sequence.

        """
        
        processed_sequence = self.seq_proc.feature_removal([sequence])[0]
        seq = (processed_sequence if processed_sequence is list else np.ndarray.tolist(processed_sequence))
        
        processed_pattern = self.seq_proc.feature_removal([np.array(pattern)])[0]
        P = np.ndarray.tolist(processed_pattern)
        
        L_S_I = min([max(c) - min(c) + 1 for c in itertools.product(*[[i for i, j in enumerate(seq) if j == element] for element in P]) if (list(c) == sorted(c) and len(c) == len(set(c)))])
        
        return len(pattern) / L_S_I
        
    
    #methods to rescale the extrinsic reward.
    def final_reward(self, reward, done):
        if done:
            reward *= self.extrinsic_rescale
        return reward
    
    def lower_intermediate_reward(self, reward, done, inter_rescale_factor=0.2):
        if done:
            reward *= self.extrinsic_rescale
        else:
            reward *= inter_rescale_factor
        return reward
    
    def lopsided(self, reward, done):
        if done:
            reward *= self.extrinsic_rescale
        elif reward < 0:
            reward *= self.extrinsic_rescale
        return reward
    
    def negative_weighted(self, reward, done):
        if reward < 0:
            reward *= self.extrinsic_rescale
        return reward


if __name__ == "__main__":
    
    #file path to get the mined rules:
    main_path = 'C:/Users/ali_9/Desktop/DATA/results/'
    EXPERIMENT = 'ddqn_one-hot_mod-env'
    CONFIG = 12
    CYCLE = 0
    RUN = 1
    mined_data_path = main_path + EXPERIMENT + '/' + 'exp_' + str(CONFIG) + '/final_rules_cycle_'+str(CYCLE)+'_run_'+str(RUN)+'.npy'
    
    #load output from the mining
    rule_output = np.load(mined_data_path, allow_pickle=True).item()
    
    #initialize environment
    DIM = 2
    NUM_QUDITS = 4
    MAX_OP = 3
    FIX_M = False
    
    env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS, max_op=MAX_OP, rand_start=True, fix_measurement=FIX_M)

    #initialize sequence processor
    ROOT_LENGTH = 2
    FEATURES_REMOVED = 1
    
    seq_processor = SeqProcessor(env=env, root_length=ROOT_LENGTH, features_removed=FEATURES_REMOVED)
    
    #initialize pattern manager
    pattern_manager = PatternManager(features_removed=seq_processor.features_removed)
    
    #send patterns output by the seqmin to the pattern manager for processing.
    patterns = pattern_manager.pattern_processing(list(rule_output.values()))
    
    #initialize intrinsic motivator
    jiminy = Jiminy(patterns, seq_processor)
    
    #generation of a sequence
    state = env.reset()
    done = False
    while not done:
        action = np.random.randint(env.num_actions)
        next_state, reward, done, _ = env.step(action)
        print('reward by the env', reward)
        state = next_state
        reward = jiminy.reward_modulation(state, reward)
        print('final rwd', reward)
        
   