import numpy as np
import itertools
import warnings

class PatternEvaluation():
    
    def __init__(self, **kwargs):
        
        """Class that evaluates the interestingness, support and cohesion of a pattern, given a data set.
        
        Args:
            
            **kwargs:
                seq_proc: class with the sequence processor.
                pattern_list (list): list of patterns to evaluate (with either the encoding from mining, e.g. ['M2_4, G2_5'], or the agent's encoding of actions).
                file_path (str): path of the file that contains the data set.
                 
        """
        
        if 'seq_proc' in kwargs:
            setattr(self, 'seq_proc', kwargs['seq_proc'])
        else:
            raise NotImplementedError('No sequence processor was given to the PatternEvaluation class, add key argument seq_proc')
        
        if 'pattern_list' in kwargs and type(kwargs['pattern_list']) is list:
            setattr(self, 'pattern_list', kwargs['pattern_list'])
        else:
            setattr(self, 'pattern_list', None)
            
        if 'file_path' in kwargs and type(kwargs['file_path']) is str:
            setattr(self, 'file_path', kwargs['file_path'])
        else:
            setattr(self, 'file_path', None)

        #TODO: this was changed, now they are separately converted, such that the output has the same format as the input
        #if self.pattern_list:
        #    self.patterns = self.conversion(self.pattern_list)#if needed, patterns in pattern_list are converted to mining encoding.

        if self.file_path:
            raw_data = np.load(self.file_path) #with agent's encoding of actions
            #process the raw data with the sequence processor.
            self.sequence_set = self.seq_proc.full_data_processing(raw_data)

        if 'max_indices' in kwargs and type(kwargs['max_indices']) is list:
            setattr(self, 'max_indices', kwargs['max_indices'])
        else:
            setattr(self, 'max_indices', [4, 2, 3, 3])

        if 'env' in kwargs:
            setattr(self, 'env', kwargs['env'])
        else:
            self.env = None
    

    def evaluation(self, pos_neg='pos'):
        """Evaluates the interestingness of the pattern P, from the set of sequences that is imported to the class.
        
        Input
        --------
        pos_neg (str): (default: 'pos') the type of data set that is evaluated. Either positively ('pos') or negatively ('neg') rewarded sequences.
        
        Output
        --------
        I (float): interestingness
        F (float): support
        C (float): cohesion
        
        """
        
        evaluation_list = []
        for P in self.pattern_list:
            P_conv = self.single_conversion(P.action_list)
            Nk_P = self.N(P_conv, self.sequence_set) #subset of sequences that contain pattern P.
            evaluation_list.append([P.name,self.interestingness(P_conv, self.sequence_set, Nk_P), self.support(self.sequence_set,Nk_P), self.cohesion(P_conv, Nk_P), pos_neg])
        return evaluation_list


    def conversion(self, patterns):
        """ Checks if the input patterns have the mining encoding, and if not, it converts agent's encoding back to the mining encoding. """

        if type(patterns[0][0][0])!=str: # if agent's encoding of actions..
            return [self.seq_proc.data_formatting([operation[:len(operation)-self.seq_proc.features_removed] for operation in P]) for P in patterns]
        else:
            return patterns

    def single_conversion(self, pattern):
        """ Checks if the input patterns have the mining encoding, and if not, it converts agent's encoding back to the mining encoding. """

        if type(pattern[0][0]) != str:  # if agent's encoding of actions..
            return self.seq_proc.data_formatting( [operation[:len(operation) - self.seq_proc.features_removed] for operation in pattern])
        else:
            return pattern

    def is_subsequence(self, P, seq):
        """
        Checks whether a subsequence P (pattern) is contained in a sequence seq.
    
        Parameters
        ----------
        P : list
            Potential subsequence.
        seq : list
            Sequence, which may include the pattern `P` as subsequence.
    
        Returns
        ----------
        bool
            Whether a subsequence `P` is contained in the sequence `seq`.
        """
        P = P if isinstance(P, list) else list([P])

        pos = 0
        for element in seq:
            if pos < len(P) and element == P[pos]:
                pos += 1
        
        return pos == len(P)




    def N(self,P, seq_list):
        """
        Computes the subset `Nk_P` of sequences that contain the subsequence P, from a set S_k with label k.
    
        Parameters
        ----------
        P : list
            Potential subsequence.
        seq_list : list
            List of sequences.
    
        Returns
        ----------
        list
            List of sequences that contain `P` as subsequence.
    
        """
    
        return [seq for seq in seq_list if self.is_subsequence(P, seq)]


    def cohesion(self,P, Nk_P):
        """
        Computes the Cohesion of a pattern `P`, by evaluating the mean `Length of the Shortest Interval` in `Nk_P`.
    
        Parameters
        ----------
        P : list
            Potential subsequence.
        Nk_P : list
            List of sequences containing the subsequence `P`.
    
        Returns
        ----------
        float
            Average `Length of the Shortest Interval` for `P` in `Nk_P`.
    
        """
        
        
        if len(Nk_P) == 0:
            return float(0)  # Arbitrary choice, when a pattern is completely absent
        else:
    
            local_Nk = [Nk_P].copy() if isinstance(Nk_P[0],
                                                   np.ndarray) == True else Nk_P.copy()  # Ensures that Cohesion works for 1+ sequences
            
            L_S_I = [min([
                max(c) - min(c) + 1 for c in itertools.product(
                    *[[i for i, j in enumerate(seq) if j == element] for element in P]
                ) if (list(c) == sorted(c) and len(c) == len(set(c)))]
            ) for seq in local_Nk]  # Computes, for each sequence in Nk_P, the Length of the Shortest Interval
    
            return len(P) / np.mean(L_S_I)




    def support(self,S_k, Nk_P):
        """
        Computes the Support of a pattern `P`,
        given by the ratio of sequences `Nk_P` that contain `P` in the class labeled `k`", i.e. `S_k`."
    
        Parameters
        ----------
        S_k : list
            List of sequences for the class with label `k`.
        Nk_P : list
            List of sequences containing the subsequence `P` in the class `S_k`.
    
        Returns
        ----------
        float
            Support of pattern `P` in class `k`.
    
        """
        if len(Nk_P) / len(S_k) > 0.9:
            print(len(Nk_P) / len(S_k))
        return len(Nk_P) / len(S_k)




    def interestingness(self,P, S_k, Nk_P):
        """
        Computes the Interestingness of a pattern `P`,
        given by the product of its Support and Cohesion in the class labeled `k`.
    
        Parameters
        ----------
        P   : list
            Subsequence
        S_k : list
            List of sequences for the class with label `k`.
        Nk_P : list
            List of sequences containing the subsequence `P` in the class `S_k`.
    
        Returns
        ----------
        float
            Interestingness of pattern `P` in class `k`.
    
        """
        return self.support(S_k, Nk_P) * self.cohesion(P, Nk_P)
            
    def confidence(self, P, S, Nk_P):
        """
        Computes the Confidence of a pattern `P`,
        given by the percentage of sequences that contain it, across all classes.
    
        Parameters
        ----------
        P   : list
            Subsequence
        S : list
            List of all sequences, ignoring all labels.
        Nk_P : list
            List of sequences containing the subsequence `P` in the class `S_k`.
    
        Returns
        ----------
        float
            Confidence of pattern `P` in class `k`.
    
        """
    
        return len(Nk_P) / len(self.N(P, S))



    def check_reward(self, gadget):
        """
        Given a gadget checks whether the gadget is rewarded.

        Args:
            gadget (np.ndarray): Input array describing gadget.

        Returns:
            reward (int): The reward associated w/ circuit.
        """
        if self.env == None:
            raise NotImplementedError(
                'Reward filter requires the self.environment as a key argument, alternatively set reward_filter to False')

        self.env.reset()
        # remove zeros from circuit
        for i, element in enumerate(gadget):
            _, reward, done, info = self.env.step(element)
        return reward, info['SRVs']




    def expand_gadgets(self, gadget):
        """
        Expands gadget of the format ['1,23,4',...] to the format of an action sequence.

        Parameters
        ----------
        gadget : list
            List of elements, represented as ['1,23,4',...]

        See Also:
        ----------
        filter_byREWARD(.)

        Returns
        ----------
        list
            List of all possible action sequences that are compatible with the partial pattern in the input.
        """

        action_lists = []
        for elem in gadget:
            action_lists.append([])
            elem = np.array([int(part) for part in elem.split(',')])
            for label in self.env.action_labels:
                if elem == label[:len(elem)]:
                    action_lists[-1].append(self.env.action_labels.index(label))

        action_sequences = []
        for combinations in itertools.product(*action_lists):
            action_sequences.append(list(combinations))

        return action_sequences