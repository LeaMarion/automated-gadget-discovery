import numpy as np
import gym
import gym_quantum_computing
from utils import *
from data_mining.pattern_evaluation import PatternEvaluation

class Comb():
    def __init__(self, **kwargs):
        """
        Implementation of the quantum-comb-inspired "agent".
        """
        if 'env' in kwargs:
            setattr(self, 'env', kwargs['env'])
        else:
            raise NotImplementedError('No environment was given to the agent, add key argument env')
        
        if 'seq_proc' in kwargs:
            setattr(self, 'seq_proc', kwargs['seq_proc'])
        else:
            raise NotImplementedError('No sequence processor was given to the agent, add key argument seq_proc')
        
        if 'patterns' in kwargs and type(kwargs['patterns']) is list:
            setattr(self, 'patterns', kwargs['patterns'])
        else:
            setattr(self, 'patterns', [])

        if 'agent_dict' in kwargs and type(kwargs['agent_dict']) is dict:
            setattr(self, 'agent_dict', kwargs['agent_dict'])
        else:
            setattr(self, 'agent_dict', {})

        if 'rng' in kwargs:
            setattr(self, 'rng', kwargs['rng'])
        else:
            setattr(self, 'rng', np.random.default_rng())

        if 'SCHEDULE' in self.agent_dict.keys() and type(self.agent_dict['SCHEDULE']) is str:
            setattr(self, 'schedule', self.agent_dict['SCHEDULE'])
        else:
            setattr(self, 'schedule', 'prob_const')

        if 'SCHEDULE_LENGTH' in self.agent_dict.keys() and type(self.agent_dict['SCHEDULE_LENGTH']) is int:
            setattr(self, 'schedule_length', self.agent_dict['SCHEDULE_LENGTH'])
        else:
            setattr(self, 'schedule_length', 20000)

        if 'THRESHOLD' in self.agent_dict.keys() and type(self.agent_dict['THRESHOLD']) is float:
            setattr(self, 'threshold', self.agent_dict['THRESHOLD'])
        else:
            setattr(self, 'threshold', 0.1)

        if 'THRESHOLD_DECAY' in self.agent_dict.keys() and type(self.agent_dict['THRESHOLD_DECAY']) is float:
            setattr(self, 'threshold_decay', self.agent_dict['THRESHOLD_DECAY'])
        else:
            setattr(self, 'threshold_decay', 0.1)

        if 'THRESHOLD_MIN' in self.agent_dict.keys() and type(self.agent_dict['THRESHOLD_MIN']) is float:
            setattr(self, 'threshold_min', self.agent_dict['THRESHOLD_MIN'])
        else:
            setattr(self, 'threshold_min', 0.1)


        self.call_counter = 0
        self.step_counter = 0
        self.rollout_counter = 0
        self.pattern_evaluation = PatternEvaluation(seq_proc=self.seq_proc)
        self.action_gen = None
        self.pattern_list = []
        if self.patterns != []:
            self.pattern_action_list = [self.patterns[i].action_list for i in range(len(self.patterns))]
            self.pattern_list = [[str(ele[ele != 0]) for ele in pattern.action_list] for pattern in self.patterns]
            self.pattern_to_actions_dict = self.generate_pattern_to_action_dict()
        if self.schedule == 'sigmoid':
            self.schedule_function = self.inverted_sigmoid(y_end=self.threshold_min, y_start=self.threshold, length=self.schedule_length, skew=self.threshold_decay)

        #the np.array that will function as an observation to the agent
        self.action_label_history = np.zeros(shape=(self.env.max_steps,len(self.env.action_labels[0])),dtype=int)


    def rollout(self):
        """
        Performs the entire rollout of a comb agent:
        (1) Sequence Selection: Choses a sequence of actions from the patterns received from the mining process
        (2) Selection: Chooses actions until the episode terminates.

        Returns:

            observation (np.ndarray) the sequence that was performed until "done" signal from environment
            reward (float) the corresponding reward for that sequence

        """
        observation = self.env.reset()
        history = []
        self.call_counter = 0
        self.step_counter = 0
        index = self.pattern_selection()
        self.pattern_action_choice = self.pattern_to_actions_dict[index]


        done = False
        while not done:
            history, observation, reward, done = self.selection(observation, history)
            self.step_counter += 1
        self.rollout_counter +=1
        observation = self.action_label_history
        return observation, reward

    def generate_pattern_to_action_dict(self):
        """
        Generates a dictionary of the actions the agent can take given the index of a pattern in the given pattern list.
        Returns:
            pattern_to_actions_dict (dict) a dictionary of the actions the agent can that given a particular pattern
        """
        pattern_to_actions_dict = {}
        for iter, pattern in enumerate(self.patterns):
            action_list = []
            for ele in pattern.action_list:
                action_list.append([])
                index = np.where(ele == 0)[0]
                for iter2, label in enumerate(self.env.action_labels):
                    if (np.delete(ele,index)==np.delete(label,index)).all():
                        action_list[-1].append(iter2)
            pattern_to_actions_dict.update({iter: action_list})
        return pattern_to_actions_dict



    def pattern_selection(self):
        """
        This function selects a pattern from the list of patterns uniformly random.
        """
        index = self.rng.integers(len(self.patterns))
        return index


    def selection(self, observation, history):
        """
        This function selects an action guided by previously found patterns.
        Args:
            observation (np.ndarray) The starting state observation.
            history (list) List of obseravtions encountered in rollout.

        Returns:
            history (list) the history of selected observations
            observation (np.ndarray) the current observation
            reward (float) the current reward
            done (bool) the episode termination truth value

        """
        history.append(observation)
        action = self.policy()
        observation, reward, done, _ = self.env.step(action)
        action_label = self.env.action_labels[action]
        self.action_label_history[self.step_counter] = action_label
        return history, observation, reward, done

    def policy(self):
        """
        depending on the probability value p the agent either chooses an action according to the chosen pattern or it chooses a random action
        :param call:
        Returns
        """
        prob = self.rng.uniform(0,1)
        if prob > self.threshold or self.call_counter >= len(self.pattern_action_choice):
            action = self.choose_action()
        else:
            action = self.comb_action()
            self.call_counter+=1
        if self.schedule == 'prob_const':
            pass
        elif self.schedule == 'exp_decay':
            if self.threshold > self.threshold_min:
                self.threshold = self.threshold*self.threshold_decay
        elif self.schedule == 'sigmoid':
            if self.rollout_counter < self.schedule_length:
                self.threshold = self.schedule_function[self.rollout_counter]
            else:
                self.threshold = self.threshold_min

        else:
            raise NotImplementedError('schedule is not yet implemented: choose prob_const or exp_decay')
        return action

    def choose_action(self):
        action = self.rng.integers(self.env.num_actions)
        return action

    def comb_action(self):
        actions = self.pattern_action_choice[self.call_counter]
        action = self.rng.choice(actions)
        return action

    ###%------------------helper functions------------------------------%
    def inverted_sigmoid(self,x_start=-100, x_end=100, y_end=0.25, y_start=0.75, length=50000, skew=0.5):
        """
        This function defines a sigmoid schedule
        :param x_start: defines the beginning of the x-value interval the function is evaluated on
        :param x_end: defines the end of the x-value interval the function is evaluated on
        :param y_start: defines the value of the function at x_start
        :param y_end: defines the value of the function at x_end
        :param length: defines the length of the schedules
        :param skew: describes the decay of the sigmoid function

        """
        x = np.linspace(x_start, x_end, length)
        z = (y_start - y_end) / (1 + np.exp(skew * x)) + y_end
        return z

    def reset(self):
        """
        Resets the threshold and the rollout counter so that schedule is started from the beginning.
        """
        self.rollout_counter = 0
        if 'THRESHOLD' in self.agent_dict.keys() and type(self.agent_dict['THRESHOLD']) is float:
            setattr(self, 'threshold', self.agent_dict['THRESHOLD'])
        else:
            setattr(self, 'threshold', 0.1)


if __name__ == "__main__":
    #setting random seed
    rng = np.random.default_rng(0)

    #ENVIRONMENT SPECIFIC
    #key arguments to control the environment parameters
    # dimension of the qudit
    DIM = 2
    # number of qudits
    NUM_QUDITS = 4
    # number of operations:
    MAX_OP = 6
    env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS,  max_op=MAX_OP)

    #we give the same pattern to the agent as it receive if the keywork patterns is not given to the agent
    #pattern_list = [[['G1_3', 'M1_4'], 0., 0., 0.]]
    pattern = Pattern(['G1_3', 'M1_4'], 0, 0., 0., 0.)
    #pattern = default_pattern()
    patterns = [pattern]


    # full example dict for comb agent
    agent_dict = {
        'SCHEDULE':'sigmoid',
        'SCHEDULE_LENGTH': 100,
        'THRESHOLD_DECAY': 0.5,
        'THRESHOLD': 0.9,
        'THRESHOLD_MIN': 0.2,
        'DATASET_SIZE': 10
        }

    # agent example with all usable keyword arguments, env is not optional, all other keywords are optional
    agent = Comb(env=env, patterns = patterns, agent_dict=agent_dict, rng = rng)
    for i in range(1000):
        rollout = agent.rollout()
        #print(rollout)