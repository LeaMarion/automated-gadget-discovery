import numpy as np
import gym
import gym_quantum_computing
from agents.tree import Tree
from agents.networkx_tree import NetworkxTree
from agents.comb import Comb
from data_mining.pattern_evaluation import PatternEvaluation
from data_mining.pattern_manager import PatternManager
from data_mining.seq_processor import SeqProcessor


class Mcts(Comb):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if 'env' in kwargs:
            setattr(self, 'env', kwargs['env'])
        else:
            raise NotImplementedError('No environment was given to the agent, add key argument env')

        if 'agent_dict' in kwargs and type(kwargs['agent_dict']) is dict:
            setattr(self, 'agent_dict', kwargs['agent_dict'])
        else:
            setattr(self, 'agent_dict', {})

        if 'TREE_TYPE' in self.agent_dict.keys() and type(self.agent_dict['TREE_TYPE']) is str:
            setattr(self, 'tree_type', self.agent_dict['TREE_TYPE'])
        else:
            setattr(self, 'tree_type', 'tree')

        if 'EXPLORATION' in self.agent_dict.keys() and type(self.agent_dict['EXPLORATION']) is float:
            setattr(self, 'exploration', self.agent_dict['EXPLORATION'])
        else:
            setattr(self, 'exploration', 1.0)

        if 'GAMMA' in self.agent_dict.keys() and type(self.agent_dict['GAMMA']) is float:
            setattr(self, 'gamma', self.agent_dict['GAMMA'])
        else:
            setattr(self, 'gamma', 1.0)

        if 'TEMPERATURE' in self.agent_dict.keys() and type(self.agent_dict['TEMPERATURE']) is float:
            setattr(self, 'temperature', self.agent_dict['TEMPERATURE'])
        else:
            setattr(self, 'temperature', 1.0)

        if 'TREE_POLICY_TYPE' in self.agent_dict.keys() and type(self.agent_dict['TREE_POLICY_TYPE']) is str:
            setattr(self, 'tree_policy_type', self.agent_dict['TREE_POLICY_TYPE'])
        else:
            setattr(self, 'tree_policy_type', 'uct_softmax')

        if 'EMPOWERMENT' in self.agent_dict.keys() and type(self.agent_dict['EMPOWERMENT']) is bool:
            setattr(self, 'empowerment', self.agent_dict['EMPOWERMENT'])
        else:
            setattr(self, 'empowerment', False)

        if 'TREE_UNCERTAINTY' in self.agent_dict.keys() and type(self.agent_dict['TREE_UNCERTAINTY']) is bool:
            setattr(self, 'tree_uncertainty', self.agent_dict['TREE_UNCERTAINTY'])
        else:
            setattr(self, 'tree_uncertainty', False)

        if 'CUT_TREE' in self.agent_dict.keys() and type(self.agent_dict['CUT_TREE']) is bool:
            setattr(self, 'cut_tree', self.agent_dict['CUT_TREE'])
        else:
            setattr(self, 'cut_tree', False)

        if 'FORGETTING' in self.agent_dict.keys() and type(self.agent_dict['FORGETTING']) is float:
            setattr(self, 'forgetting', self.agent_dict['FORGETTING'])
        else:
            setattr(self, 'forgetting', 0.0)

        if 'BOREDOM' in self.agent_dict.keys() and type(self.agent_dict['BOREDOM']) is bool:
            setattr(self, 'boredom', self.agent_dict['BOREDOM'])
        else:
            setattr(self, 'boredom', False)

        if 'BOREDOM_THRESHOLD' in self.agent_dict.keys() and type(self.agent_dict['BOREDOM_THRESHOLD']) is int:
            setattr(self, 'boredom_threshold', self.agent_dict['BOREDOM_THRESHOLD'])
        else:
            setattr(self, 'boredom_threshold', 100)


        #default value for number of expansion for MCTS, is adapted for MCTS empowerment via horizon
        self.num_expansions = 1


        if self.empowerment:
            self.exploration *= 1/np.sqrt(2)
            if 'HORIZON' in self.agent_dict.keys() and type(self.agent_dict['HORIZON']) is int:
                setattr(self, 'horizon', self.agent_dict['HORIZON'])
            else:
                setattr(self, 'horizon', self.env.max_steps)


        if self.tree_type == 'tree':
            self.tree = Tree()
            self.draw = False
        elif self.tree_type == 'networkx_tree':
            agent_dict = self.agent_dict
            self.tree = NetworkxTree(agent_dict=agent_dict)
            self.draw = True
        else:
            raise NotImplementedError('This tree type is not yet implemented.')
        self.rollout_counter = 0
        self.step_counter = 0
        self.pattern_encounters = 0
        if self.pattern_list != []:
            self.min_pattern_length = min([len(pattern) for pattern in self.pattern_list])
        else:
            self.min_pattern_length = 0


    def action_selection(self, hsh):
        """
        Receives the current node as input and chooses the next node.
        Args:
             hsh (int) The hash of the current node.
        Returns:
             hsh (int) The hash of the selected node.
        """
        pass

    def rollout(self):
        """
        Performs the entire mcts rollout:
        (1) Selection: Chooses actions until a leaf node is reached.
        (2) Expansion: Expanded the tree starting from the leaf node
        (3) Simulation: Chooses random actions until the episode is terminated.
        (4) Backpropergation: propergates the reward through the tree

        """
        self.step_counter = 0

        observation = self.env.reset()
        info = None
        #start from a different state

        pre_step_actions = []
        for action in pre_step_actions:
            _, _, _, info = self.env.step(action)
            start_info = info

        #the np.array that will function as an observation to the agent
        self.action_label_history = np.zeros(shape=(self.env.max_steps-len(pre_step_actions),len(self.env.action_labels[0])),dtype=int)
        self.tree.reset(observation)
        self.tree.current_node.num_children = len(self.env.action_list)
        reward = 0
        done = False

        observation, reward, done, info = self.selection(observation, reward, done, info)
        observation, reward, done, info = self.expansion(observation, reward, done, info)
        observation, reward, done, info = self.simulation(observation, reward, done, info)


        self.backpropergation(reward)
        if self.tree.type == 'networkx_tree':
            self.draw_tree()
        self.rollout_counter += 1
        observation = self.action_label_history

        adapt_reward = False
        if reward > 0 and adapt_reward:
            for start_elem in start_info['SRVs']:
                for end_elem in info['SRVs']:
                    sorted_start_elem = np.sort(start_elem)
                    sorted_end_elem = np.sort(end_elem)
                    for i in range(len(end_elem)):
                        if sorted_end_elem[i] > sorted_start_elem[i]:
                            reward = 1
                            break
                        else:
                            reward = 0
        return observation, reward


    def selection(self, observation, reward, done, info):
        """
        This function selects the tree policy which corresponds to the sequence of nodes selected until a leaf node is reached.
        Returns:
            history (list) the history of selected observations.
            observation (np.ndarray) the current observation
            reward (float) the current reward
            done (bool) the episode termination truth value

        """
        pattern_history = []
        while True:
            removed_actions = self.tree.current_node.removed_actions
            if len(self.tree.current_node.children) != len(self.env.action_list)-len(removed_actions) or done:
                return observation, reward, done, info
            else:
                node, action = self.tree_policy()
                self.tree.current_node = node
                observation, reward, done, info = self.env.step(action)
                action_label = self.env.action_labels[action]
                self.action_label_history[self.step_counter] = action_label
                pattern_history, reward, done = self.pattern_guidance(action,reward,done,pattern_history)
                self.step_counter += 1


    def expansion(self, observation, reward, done, info):
        """
        Adds the current chosen node as a child to the tree.
        """
        if self.empowerment:
            self.num_expansions = self.horizon - self.step_counter

        expansion_counter = 1
        while True:
            if expansion_counter > self.num_expansions or done:
                return observation, reward, done, info
            else:
                action_list = self.env.action_list.copy()
                removed_actions = self.tree.current_node.removed_actions
                children = self.tree.current_node.children

                if self.empowerment:
                    expansion_counter += 1
                else:
                    if len(children) > 0:
                        expansion_counter = 0
                    else:
                        expansion_counter += 1

                if len(self.tree.current_node.children) < self.env.num_actions - len(removed_actions):
                    action_list = list(range(self.env.num_actions))
                    for child in children:
                        action_list.remove(child.action_num)
                    for action in removed_actions:
                        action_list.remove(action)
                # choose randomly any of the remaining actions
                action = self.rng.choice(action_list)
                child = self.tree.add_child(action)
                self.tree.current_node = child
                self.unique = True
                observation, reward, done, info = self.env.step(action)
                action_label = self.env.action_labels[action]
                self.action_label_history[self.step_counter] = action_label
                self.tree.current_node.num_children = len(self.env.action_list)
                if done:
                    self.tree.current_node.sigma = 0
                self.step_counter += 1




    def simulation(self, observation, reward, done, info):
        """
        Chooses random actions until the episode terminates and returns the reward of the episode.
        Args:
            int (obs) The hash of the current observation.
        Returns:
            observation (np.array) the current observation
            reward (float) the reward of the current episode
            done (bool) episode termination truth value
        """
        while not done:
            action_list = self.env.action_list.copy()
            action = self.rng.choice(action_list)
            observation, reward, done, info = self.env.step(action)
            action_label = self.env.action_labels[action]
            self.action_label_history[self.step_counter] = action_label
            self.step_counter+=1
        return observation, reward, done, info


    def backpropergation(self, reward):
        """
        Backpropergates the reward through starting from the current chosen leaf node
        Args:
            reward (float) Reward issued by the environment at the end of the episode
        """
        backprop = True
        while backprop:
            self.tree.current_node.N += 1
            if self.empowerment:
                if reward > 0 and self.unique:
                    #print(self.tree.current_node.id)
                    self.tree.current_node.Q += reward
                    if self.boredom:
                        if self.tree.current_node.Q > self.boredom_threshold:
                            #print(self.boredom_threshold)
                            self.tree.current_node.Q = 0

            else:
                self.tree.current_node.Q += reward
            reward *= self.gamma

            if self.tree.current_node.parent != None:
                self.tree.current_node = self.tree.current_node.parent
            else:
                backprop = False

        if self.tree_uncertainty:
            #define leaf node as node that leads to an environment done=True
            while self.tree.current_node != self.tree.top_node:
                self.tree.current_node = self.tree.current_node.parent
                new_sigma = 0.
                factor = 0.
                for child in self.tree.current_node.children:
                    new_sigma += child.N*child.sigma
                    factor += child.N
                new_sigma += self.tree.current_node.num_children-len(self.tree.current_node.children)
                factor += self.tree.current_node.num_children-len(self.tree.current_node.children)

                new_sigma = new_sigma/factor
                self.tree.current_node.sigma = new_sigma

    def pattern_guidance(self, action, reward, done, pattern_history):
        """
        This function checks if the interaction history aligns with previous interactions.
        Then stops the interaction and issues a negative reward and sets done to True.
        If self.cut_tree = True. The tree such that this pattern cannot be taken again.

        Args:
            action (int) the action taken in the environment
            pattern_history (list) the action label history of the actions so far taken in this rollout

        Returns:
            done (bool) this value checks if the interaction with the environment will be terminated
            reward (float) the
            pattern_history (list) the action label history of the actions so far taken in this rollout appended by the
                                   most recent action label

        """
        if self.pattern_list != []:
            label = self.env.action_labels[action]
            label = label[0:len(label) - self.patterns[0].features_removed]
            pattern_history.append(str(label))

            if len(pattern_history) >= self.min_pattern_length:
                for pattern in self.pattern_list:
                    subseq = self.pattern_evaluation.is_subsequence(pattern, pattern_history)
                    if subseq:
                        print('FOUND!!')
                        print(pattern)
                        print(pattern_history)
                        print(self.tree.current_node.parent.action_num)
                        print(action)

                        self.pattern_encounters += 1
                        reward = -1
                        done = True
                        if self.cut_tree:
                            self.tree.remove_current_node()
                        return pattern_history, reward, done
        return pattern_history, reward, done

    def tree_policy(self):
        """
        Interface function between the policy and the different choices for the policy (uct, softmax uct, etc.) so that other functionalities can be added here.
        """
        node = self.tree.current_node
        node, action = self.selected_tree_policy(node)
        return node, action



    def selected_tree_policy(self, node):
        """
        Selects an action according to the selected tree policy type.

        Args:
            node (obj) current node
        Returns:
            node (obj) selected node
            action (int) selected action

        """
        if self.tree_policy_type == 'uct_softmax':
            node, action = self.uct_softmax(node)

        elif self.tree_policy_type == 'uct':
            node, action = self.uct(node)
        else:
            raise NotImplementedError('This tree policy type is not yet implemented.')
        return node, action




#%--------------------------helper_functions--------------------------------------------

    def uct_softmax(self, node):
        """
        Selects the next action according to the UCT. And expands the tree whenever a
        new state (action sequence) is encountered.

        Returns:
            action (int) The hash of the selected observation.
        """

        self.unique = False
        N = node.N
        N_values = np.array([])
        Q_values = np.array([])
        sigma = np.array([])
        for child in node.children:
            N_values = np.append(N_values, child.N)
            Q_values = np.append(Q_values, child.Q)
            sigma = np.append(sigma, child.sigma)

        uct = Q_values / N_values + sigma * self.exploration * np.sqrt(2*np.log(N) / N_values)
        softmax_values = np.exp(self.temperature*uct)
        softmax_values = softmax_values/np.sum(softmax_values)
        index = self.rng.choice(range(len(uct)), p=softmax_values)
        node = node.children[index]
        action = node.action_num
        return node, action

    def uct(self, node):
        """
        Selects the next action according to the UCT. And expands the tree whenever a
        new state (action sequence) is encountered.

        Returns:
            action (int) The hash of the selected observation.
        """

        self.unique = False
        N = node.N
        N_values = np.array([])
        Q_values = np.array([])
        sigma = np.array([])
        for child in node.children:
            N_values = np.append(N_values, child.N)
            Q_values = np.append(Q_values, child.Q)
            sigma = np.append(sigma, child.sigma)


        uct = Q_values / N_values + sigma * self.exploration * np.sqrt(2*np.log(N) / N_values)
        index = np.argmax(uct)
        node = node.children[index]
        action = node.action_num
        return node, action


    def draw_tree(self):
        if self.draw:
            if self.rollout_counter in range(self.tree.start_draw, self.tree.end_draw):
                self.tree.draw_graph()


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
    #PATTERNS
    seq_processor = SeqProcessor(env=env)
    pattern_manager = PatternManager(env=env, seq_proc=seq_processor)
    patterns = np.load('results/sequence_mining/exp_1/EXPLORATION=0.5/final_rules_cycle_0_file1.npy', allow_pickle=True)
    patterns = patterns.item()
    print(len(patterns))
    for pattern in patterns:
        name, I, F, C, CONF = patterns[pattern]
        pattern_manager.pattern_update(name, I, F, C)

    pattern_list = pattern_manager.all_pattern_ranking(10)
    pattern_list = [pattern[0] for pattern in pattern_list]
    pattern_length = [len(pattern.action_list) for pattern in pattern_list]
    patterns = pattern_list


    #MCTS
    agent_dict = {
        'EXPLORATION': 0.1,
        'TREE_TYPE': 'tree',
        'GAMMA': 0.6,
        'TEMPERATURE': 5.0,
        'TREE_POLICY_TYPE': 'uct_softmax',
        'EMPOWERMENT':True,
        'HORIZON': 6,
        'TREE_UNCERTAINTY':True,
        'CUT_TREE': False
        }

    agent = Mcts(env=env, agent_dict = agent_dict, rng = rng, patterns = patterns, seq_proc=seq_processor)
    for i in range(10000):
        #print(i)
        agent.rollout()
        #print(agent.pattern_encounters)


    #MCTS using a networkx tree, so that the graph can be drawn and exported as pngs
    agent_dict = {
        'EXPLORATION': 0.1,
        'TREE_TYPE': 'networkx_tree',
        'GAMMA': 0.6,
        'TEMPERATURE': 5.0,
        'TREE_POLICY_TYPE': 'uct_softmax',
        'EMPOWERMENT': False,
        'HORIZON': 4,
        'TREE_UNCERTAINTY': False,
        'CUT_TREE': False,
        'START_DRAW': 0,
        'END_DRAW': 100,
        'GRAPH_FOLDER': 'results/graph'
        }

    agent = Mcts(env=env, agent_dict = agent_dict, rng = rng, seq_proc=seq_processor)
    for i in range(0):
        print(agent.rollout())

