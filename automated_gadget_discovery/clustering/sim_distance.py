import numpy as np
import pathlib
import time
import warnings
from sys import stdout
import hdbscan
from itertools import product

import gym
import gym_quantum_computing

from data_mining.seq_processor import SeqProcessor

class SimDistance():
    def __init__(self, **kwargs):
        """
        Class that computes the distance matrix between pairs of gadgets.
        For that, it first gets the initialization set of each gadget, and then computes the similarity measure between sets.
        
        **kwargs:
                seq_proc: class with the sequence processor.
                patterns (list) : patterns that are considered for the evaluation. Each element in this list is an instance of the class Pattern.
                results_folder (str) path of the folder where results from the exchanger will be saved.
                supplementary_folder (str) path where supplementary information from the exchanger is stored.
                run (int) id for the current run of the experiment (several cycles).
                indices_env_placements (list/np.array): optional. It indicates which indices of a sequence correspond to elements placed by the env.
                sim_measure (str): default: Hamming. which similarity measure is used.
                    Options: 'Hamming', 'Levenshtein', 'Tally'
                one_hot_dict (dict): optional. Dictionary with one-hot encoding of each action.
                    If provided, the keys should be str of lists (featured actions) or str of int (non-featured actions). Both keys and values should already be displayed without the features that are to be ignored (features_removed).
                    If not provided, it is automatically computed considering that each feature will be encoded in the #bits indicated in the corresponding
                    entry of seq_proc.max_value_features. 
                max_initset_size (int): size of the init sets.
                weights_bit_blocks (list/np.array): in case the bit encoding is split into features, this list establishes the weight of each feature in the computation of the distance between sequences.
                    
                """
        
       
        if 'seq_proc' in kwargs:
            setattr(self, 'seq_proc', kwargs['seq_proc'])
        else:
            raise NotImplementedError('No sequence processor was given to the agent, add key argument seq_proc')
        
        if 'patterns' in kwargs and type(kwargs['patterns']) is list:
            setattr(self, 'patterns', kwargs['patterns'])
        else:
            setattr(self, 'patterns', None)
            
        if 'results_folder' in kwargs and type(kwargs['results_folder']) is str:
            setattr(self, 'results_folder', kwargs['results_folder'])
        else:
            setattr(self, 'results_folder', None)
            
        if 'supplementary_folder' in kwargs and type(kwargs['supplementary_folder']) is str:
            setattr(self, 'supplementary_folder', kwargs['supplementary_folder'])
        else:
            setattr(self, 'supplementary_folder', '')
            
        if 'run' in kwargs and type(kwargs['run']) is int:
            setattr(self, 'run', kwargs['run'])
        else:
            setattr(self, 'run', 0)
            
        if 'indices_env_placements' in kwargs:
            setattr(self, 'indices_env_placements', kwargs['indices_env_placements'])
        else:
            setattr(self, 'indices_env_placements', [None])
            
        if 'sim_measure' in kwargs and type(kwargs['sim_measure']) is str:
            setattr(self, 'sim_measure', kwargs['sim_measure'])
        else:
            setattr(self, 'sim_measure', 'Hamming')
            
        if 'one_hot_dict' in kwargs and type(kwargs['one_hot_dict']) is dict:
            setattr(self, 'one_hot_dict', kwargs['one_hot_dict'])
        else:
            print('\nOne_hot_encoding will be computed automatically considering the max_value_features of the seq_proc.')
            setattr(self, 'one_hot_dict', None)
        
        if 'max_initset_size' in kwargs and type(kwargs['max_initset_size']) is int:
            setattr(self, 'max_initset_size', kwargs['max_initset_size'])
        else:
            setattr(self, 'max_initset_size', 1200)
            
        if 'weights_bit_blocks' in kwargs:
            setattr(self, 'weights_bit_blocks', kwargs['weights_bit_blocks'])
        else:
            setattr(self, 'weights_bit_blocks', None)
            
    
        if self.results_folder:
            #create folder if necessary
            pathlib.Path(self.results_folder+self.supplementary_folder).mkdir(parents=True, exist_ok=True)

        
        if self.patterns:
            #distance matrix between patterns.
            self.pattern_distance_matrix = np.zeros([len(self.patterns), len(self.patterns)])
            #dictionary that stores the init sets of the patterns (to avoid computing them several times). 
            self.init_set_dict = {id: None for id in [pattern.id for pattern in self.patterns]}
        
        #create a dictionary from action labels to actions. (This is useful in case actions have features)
        self.labels_to_actions =  {str(None):None}
        for i, action_label in enumerate(self.seq_proc.action_labels):
            self.labels_to_actions.update({str(action_label): i})
            
        #set weights_bit_blocks in case the user did not specify any
        if self.weights_bit_blocks == None:
            self.weights_bit_blocks = [1/(self.seq_proc.num_features - self.seq_proc.features_removed)] * (self.seq_proc.num_features - self.seq_proc.features_removed)
        else:
            if np.sum(self.weights_bit_blocks) != 1:
                print('\nThe weigths do not add up to 1. They will be reset to be equally distributed.')
                self.weights_bit_blocks = [1/(self.seq_proc.num_features - self.seq_proc.features_removed)] * (self.seq_proc.num_features - self.seq_proc.features_removed)
            
    def distance_matrix_population(self):
        """
        Populates the distance matrix.
        
        For each pair of patterns (i, j), it computes the matrix with elements:
            M_ij: similarity measure between the init sets of pattern i and pattern j (M_ij = M(I_i, I_j))
        
        """
        
        #populate the context dictionary and the distance matrix.
        for i, j in zip(np.triu_indices(len(self.patterns),1)[0], np.triu_indices(len(self.patterns),1)[1]):
            t1 = time.perf_counter()
            print('\nCurrent analyzed patterns:', self.patterns[i].name, self.patterns[j].name)
            
            self.pattern_distance_matrix[i, j] = self.sim_measure_init_sets(self.patterns[i], self.patterns[j])
            self.pattern_distance_matrix[j, i] = self.pattern_distance_matrix[i, j]
            
            t2 = time.perf_counter()
            print('Time (in min) for patterns',i,j,':  ',(t2-t1)/60)
            stdout.flush()
            
            #save distance matrix.
            np.save(self.results_folder+self.supplementary_folder+'distance_matrix_{}.npy'.format(self.sim_measure), self.pattern_distance_matrix)
          
          
            
    def sim_measure_init_sets(self, pattern_i, pattern_j):
        """
        Computes the similarity measure between the init set of pattern_i and the init set of pattern_j.

        Parameters
        ----------
        pattern_i (j) : pattern class. Class with the information of the pattern (in which cycle it was obtained, its name, its action list, etc.).
        
        Returns
        -------
        Similarity measure between the init sets of the two patterns M(I_i, I_j).
        
        """
        
        #get initialization sets if they are not already computed.
        if self.init_set_dict[pattern_i.id] is None:
            self.init_set(pattern_i)
            
        if self.init_set_dict[pattern_j.id] is None:
            self.init_set(pattern_j)
        
        lengths = np.array([len(seq) for seq in self.init_set_dict[pattern_i.id]])
        print('\nLengths in init_set 1:', np.unique(lengths))
        lengths2 = np.array([len(seq) for seq in self.init_set_dict[pattern_j.id]])
        print('Lengths in init_set 2:', np.unique(lengths2))
       
        #initialize matrix of pairwise distances (each distance will be the distance between two sequences)
        d = np.full([len(self.init_set_dict[pattern_i.id]), len(self.init_set_dict[pattern_j.id])], np.inf)
        
        #populate d matrix
        for index, sequence in enumerate(self.init_set_dict[pattern_i.id]):
            d[index] = self.distance(sequence, self.init_set_dict[pattern_j.id])
            
        #save d matrix.
        np.save(self.results_folder+self.supplementary_folder+'d_matrix_patt_{}_{}.npy'.format(pattern_i.id,pattern_j.id), d)
        
        #compute the minimum over rows and columns and delete the np.inf placeholders (only the comparable sequences contribute to the mean)
        del_row = np.delete(np.min(d,axis=0), np.where(np.min(d,axis=0) == np.inf))
        del_col = np.delete(np.min(d,axis=1), np.where(np.min(d,axis=1) == np.inf))
        
        print('#points for mean, patt. 1:', len(del_col))
        print('#points for mean, patt. 2:', len(del_row))
        
        #compute the mean over the min values. If there is no comparable sequence, the min is set to the maximum value.
        M_ij = np.mean(np.append((del_row if len(del_row) != 0 else 1),\
                                 (del_col if len(del_col) != 0 else 1)))
            
        
        print('Distance Mij:', M_ij)
        
        return M_ij
    
    def distance(self, sequence, init_set):
        """
        Computes the distance between sequence and each of the sequences in init_set.

        Parameters
        ----------
        sequence : (list of lists) sequence to be evaluated.
        init_set : (list of sequences, where each sequence is a list) initialization set.
            
        Returns
        -------
        distance_row : (np.array) distances computed according to the chosen similarity measure.
            
        """
        #array with lengths of the sequences of the init set 
        lengths = np.array([len(seq) for seq in init_set])
        w = np.where(lengths == len(sequence))[0]
        
        distance_row = np.full(len(init_set), np.inf)
        
        if len(w) != 0:
            for index, second_seq in enumerate(init_set[w[0]:w[-1]+1]):
                
                if self.sim_measure == 'Hamming':
                    distance_row[w[index]] = self.hamming(sequence, second_seq)
                elif self.sim_measure == 'Levenshtein':
                    distance_row[w[index]] = self.levenshtein(sequence, second_seq)
                elif self.sim_measure == 'Tally':
                    distance_row[w[index]] = self.tally(sequence, second_seq)
        else:
            warnings.warn('In this init_set, there are not sequences of the same length as the input one. The min. distance will be set to 1 in this row.')
        
        return distance_row
    
    def utility_based_distance(self):
        """
        Computes the distance between pairs of patterns by computing the utilities of each pattern
        (how many elements of each type are contained in the pattern).
        
        Note: this is not a classification based on context. It is based on the information of the pattern elements.

        Returns
        -------
        distance : (np.array) Distance matrix between pairs of patterns
        """
        
        distance = np.zeros((len(self.patterns), len(self.patterns)))
        
        for i, j in product(np.arange(len(self.patterns)),np.arange(len(self.patterns))):
            distance[i,j] = self.tally(self.patterns[i].action_list, self.patterns[j].action_list)
        
        return distance
        
    #----------helper methods-----------#
    
    def init_set(self, pattern):
        """
        Computes and stores the initialization set of the given pattern.

        Parameters
        ----------
        pattern : class of the pattern

        Adds to dictionary
        ------------------
        init_set : list of lists (Each inner list can have a different length)
            Initialization set.

        """
        
        init_set = []
        
        #get raw data
        if self.results_folder:
            raw_data = np.load(self.results_folder + 'data_cycle_'+str(pattern.cycle[0])+'_run_'+str(pattern.agent[0])+'.npy') #sequences with environment notation.
            np.random.shuffle(raw_data)
        else:
            raw_data = []
            warnings.warn('No folder was given. The initialization set cannot be computed.')
        
        #get initialization sequences
        for sequence in raw_data:
            in_sequence, init_sequence = self.pattern_in_sequence(pattern.action_list, sequence)
            #add sequence to the context dictionary if original pattern is in it.
            if in_sequence:
                init_set.append(init_sequence)
        
        #trim init set to init_set size chosen by the user
        init_set = init_set[:self.max_initset_size]
        
        #(ordered by sequence length in ascending order)
        init_set.sort(key=len)
        
        #save initialization set
        np.save(self.results_folder+self.supplementary_folder+'init_set_pattern_{}.npy'.format(pattern.id), np.asarray(init_set, dtype=object))
        
        #add initialization set to the dictionary.
        self.init_set_dict[pattern.id] = init_set
        
    
    def pattern_in_sequence(self, pattern, sequence):
        """
        Checks whether the pattern is contained in the sequence or not.
        
        Parameters
        ----------
        pattern : (list of np.arrays)
            
        sequence : (np.array). Sequence to be analyzed.

        Returns
        -------
        True/False: (bool) whether the pattern is contained in the sequence.
        init_sequence (list of lists): The sequence as it was before the first element of the pattern is introduced.

        """
        #TODO: (maybe?) add code so that if the pattern appears twice or more in the sequence, all the states prior to each placement are added.
        
        init_sequence = []
        
        seq = np.ndarray.tolist(self.seq_proc.feature_removal([sequence])[0])
        P = np.ndarray.tolist(self.seq_proc.feature_removal([np.array(pattern)])[0])
        
        pos = 0
        for i, element in enumerate(seq):
            
            if pos < len(P) and element == P[pos] and i not in self.indices_env_placements:
                pos += 1
            
            if pos == 0 or i in self.indices_env_placements:
                init_sequence.append(element)
        
        return pos == len(P), init_sequence
    
    def hamming(self, sequence, second_sequence, norm_bit_blocks=[2,2,4], weights=None):
        """
        Hamming distance (normalized over the sequence length) between the two given sequences (of equal length).
        
        Parameters
        ----------
        sequence : (list of lists) 
        second_sequence : (list of lists)
        norm_bit_blocks : (list) max value that each block of bits can have
                Default: None (if there are no blocks)
        weights : (list) weight assigned to each block of bits (this is defined by the user, depending on the prior knowledge)
                Default: [1] (if there is no prior knowledge)
                
        Returns
        -------
        (int) Hamming distance
            
        """
        one_hot_seq1 = self.one_hot_encoding(sequence)
        one_hot_seq2 = self.one_hot_encoding(second_sequence)
        
        if norm_bit_blocks:
            if weights == None:
                weights = self.weights_bit_blocks
                
            #sum the contribution of each element of the sequence (each row) to the distance
            #before the sum: array with the binary difference split into rows (sequence length), and columns (#bits per element)
            sum_per_element = np.sum(np.reshape((one_hot_seq1 + one_hot_seq2) % 2, \
                                                (len(sequence), np.sum(norm_bit_blocks))), axis=0)
            
            #split the array of the sum into the given blocks
            c = 0
            distance = 0
            for i, index_slice in enumerate(np.cumsum(norm_bit_blocks)):
                distance += 1/len(sequence) * 1/norm_bit_blocks[i] * weights[i] * np.sum(sum_per_element[c:index_slice]) 
                c = np.copy(index_slice)
            
            return distance
        else:
            return np.sum((one_hot_seq1 + one_hot_seq2) % 2)
            
        
    def one_hot_encoding(self, sequence):
        """
        Computes the one hot encoding of the full sequence.

        Parameters
        ----------
        sequence : (list) sequence to encode. 
            It has the 'features_removed' features already removed.
           
        Returns
        -------
        encoded_sequence : (np.array) one-hot encoding of the sequence.
            
        """
        encoded_sequence = np.array([])
        if self.one_hot_dict:
            for element in sequence:
                encoded_sequence = np.append(encoded_sequence, self.one_hot_dict[str(element)])    
        else:
            for element in sequence:
                if self.seq_proc.num_features == 1:
                    #takes into account that the min value of 'element' is 0.
                    encoded_sequence = np.append(encoded_sequence, self.get_string(element+1, self.seq_proc.max_value_features+1)) 
                else:
                    for i, feature in enumerate(element):
                        encoded_sequence = np.append(encoded_sequence, self.get_string(feature, self.seq_proc.max_value_features[i]))
        
        return encoded_sequence
    
    def get_string(self, digit_toencode, string_length):
        """
        Computes the one-hot encoding of 'string_length' bits with all '0' except for the '1'
        corresponding to the value of digit_toencode.

        Parameters
        ----------
        digit_toencode : (int) value to encode as one-hot (string with '0' everywhere except for the position corresponding
                                                           to the value of the digit, in which the bit is '1')
            
        string_length : (int). string_length should be >= digit_toencode.
            
        Returns
        -------
        string : (list). One-hot encoding of digit_toencode in string_length bits.
            
        """
        string = [0] * string_length
        string[digit_toencode-1] = 1
        
        return string
    
    def levenshtein(self, sequence, second_seq):
        """
        Levenshtein distance between the two given sequences.
        
        Parameters
        ----------
        sequence : (list of lists) 
        second_sequence : (list of lists)
            
        Returns
        -------
        (int) Levenshtein distance

        """
        a = ''.join(str(i) for element in sequence for i in element)
        b = ''.join(str(j) for element in second_seq for j in element)
        
        def min_dist(s1, s2):
            """
            Recursive function to compute the levenshtein distance.
            """
            
            if s1 == len(a) or s2 == len(b):
                return len(a) - s1 + len(b) - s2
        
            # no change required
            if a[s1] == b[s2]:
                return min_dist(s1 + 1, s2 + 1)
        
            return 1 + min(
                min_dist(s1, s2 + 1),      # insert character
                min_dist(s1 + 1, s2),      # delete character
                min_dist(s1 + 1, s2 + 1))  # replace character
        
        return min_dist(0,0)
    
    def tally(self, sequence, second_seq):
        """
        Euclidean distance of the vectors containing the tally of each type of element in the sequence.

        Parameters
        ----------
        sequence : (list of lists) 
        second_sequence : (list of lists)
            
        Returns
        -------
        (int) Euclidean distance of the given embedding.

        """
        
        a = self.seq_proc.sample_tally(self.seq_proc.data_formatting(sequence)) 
        b = self.seq_proc.sample_tally(self.seq_proc.data_formatting(second_seq))
        
        return np.sqrt(np.sum((a-b)**2))
    
    
    
    
    
if __name__ == "__main__":
    
    #file path to get the data set and the patterns:
    #main_path = 'D:/DATA/results/'
    main_path = 'results/'
    EXPERIMENT = 'cycle'
    CONFIG = 34
    RUN = 2
    
    results_folder = main_path + EXPERIMENT + '/' + 'exp_' + str(CONFIG) + '/'
    
    #Get patterns.
    patterns = []
    for RUN in range(8):
        pattern_dict = np.load(results_folder + 'pattern_dict_run_'+str(RUN)+'.npy', allow_pickle=True).item()      
        patterns += [p for p in pattern_dict.values()]
    print('\n',len(patterns), 'patterns in pattern_dict.\n')
    
    
    #Initialize env
    DIM = 2
    NUM_QUDITS = 4
    MAX_OP = 6
    FIX_M = False
    FIXED_OP = 3
    
    env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS, max_op=MAX_OP, rand_start=True, fix_measurement=FIX_M, fixed_op=FIXED_OP)
    #get action labels from env
    action_labels = env.action_labels
    #populate and get the dictionary with the one-hot encoding of each featured action.
    one_hot_dict = {}
    for action, encoding in enumerate(env.actions_one_hot_encoding):
        one_hot_dict[str(list(action_labels[action])[:-1])] = encoding[:-4]
    
    #Initialize sequence processor
    ROOT_LENGTH = 2
    FEATURES_REMOVED = 1
    INDICES_TO_MINE = [0,1,3,4,6,7]
    EXTRACT_LABELS = 'checked_cartesian'
    processor_file_path = results_folder + 'data_cycle_0_run_0.npy'#just to compute the characteristics of the sequences.
    seq_processor = SeqProcessor(file_path=processor_file_path, root_length=ROOT_LENGTH, action_labels=action_labels, features_removed=FEATURES_REMOVED, indices_to_mine=INDICES_TO_MINE, extract_labels=EXTRACT_LABELS)
    
    #Initialize similarity measure
    sup_folder = 'utility_clustering/'
    INDICES_ENV = [2,5,8]
    SIM_MEASURE = 'Hamming'
    INFO = 'utility'
    sim = SimDistance(seq_proc=seq_processor, patterns=patterns, results_folder=results_folder, supplementary_folder=sup_folder, run=RUN, indices_env_placements=INDICES_ENV, sim_measure=SIM_MEASURE, one_hot_dict=one_hot_dict)
    
    
    #initialize clusterer
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=2, min_cluster_size=5)
    
    if INFO == 'context':
        sim.distance_matrix_population() #compute distance matrix
    
        #load a distance matrix instead
        # pattern_distance_matrix = np.load(results_folder + 'gadget_clustering/initset_2000/dataset_sample_1/'+'distance_matrix_Hamming_run_'+str(RUN)+'.npy')
    
        clusterer.fit(sim.pattern_distance_matrix)
        # clusterer.fit(pattern_distance_matrix) #to use a loaded distance matrix
        
        labels = clusterer.labels_
        prob = clusterer.probabilities_
        
        # np.save(results_folder + sup_folder + 'labels_run_{}.npy'.format(RUN), labels)
        # np.save(results_folder + sup_folder + 'prob_labels_run_{}.npy'.format(RUN), prob)
        
    elif INFO == 'utility':
        #utility based clustering
        clusterer.fit(sim.utility_based_distance())
    
        labels = clusterer.labels_
        prob = clusterer.probabilities_
        
        # np.save(results_folder + sup_folder + 'labels_utility_run_{}.npy'.format(RUN), labels)
        # np.save(results_folder + sup_folder + 'prob_labels_utility_run_{}.npy'.format(RUN), prob)
        
        np.save(results_folder + sup_folder + 'labels_utility_all.npy', labels)
        np.save(results_folder + sup_folder + 'prob_labels_utility_all.npy', prob)
        