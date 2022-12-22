import numpy as np
import random
import warnings
from itertools import product

class SeqProcessor():
    
    def __init__(self, **kwargs):
        
        """Class that processes the sequences obtained by the agent. All the sequences in the data set are transformed to make them ready for 
        mining and/or evaluation. The user choices include: in case of actions that consist of different features, how many features are considered, which of these features
        describe the main element types...
        
        Args:
            
            **kwargs:
                file_path: file path of data to mine 
                root_length (int): optional (default:1) how many features are considered to be the main root information.
                features_removed (int): optional (default: 0), which assumes that no element is ignored.
                                Decides how much information we want to ignore, from the leaves of the tree defining the element of the sequence to the root.
    
                                + 0: Full information (useless for sequence mining).
                                + 1: We ignore the detailed information on the operation (4th digit).
                                + 2: We ignore where the element acts on (3rd and 4th digits). Only distinguishes between G1, G2, M1, M2.
                
                indices_to_mine (list): optional (default:None). If given, this list contains the indices of the sequence elements that are to be mined. 
                                        Useful if one mines only the actions performed by the agent when they are interspersed by elements placed by the environment.
                
                num_features (int) (*): (default:1). In case of featured actions, how many features define each action.
                max_value_features (int/list) (*): Max value each feature can take.
                extract_labels (str): mode with which action labels are extracted.
                                'simple' (fixed for non-featured actions): list of arrays where each array is simply the action number.
                                'cartesian' (default for featured actions): cartesian product of all possible feature values (from minimum to maximum value in data set).
                                'checked_cartesian': in addition to cartesian, it checks that all the combinations are actually in the data set.
                                'user_input': the labels are input by the user directly.
                                
                action_labels (list of np.arrays) (*): labels of the actions. Each label is a description of the action in terms of its features.
                                These can be input by the user, or automatically extracted by the sequence processor (see extract_labels, generate_labels).
                                (e.g. action = 0, label = [1,1,1,1]. Action with 4 features, each of which has value 1.)
                
                sequence_length (int) (*): length of the sequences.
                
                (*) args that are automatically computed by analyzing the data provided in file_path.
        """
        
        if 'file_path' in kwargs:
            setattr(self, 'file_path', kwargs['file_path'])
        else:
            setattr(self, 'file_path', None)
            
        if 'root_length' in kwargs and type(kwargs['root_length']) is int:
            setattr(self, 'root_length', kwargs['root_length'])
        else:
            setattr(self, 'root_length', 1)
        #TODO: change default value of root_length to be equal to num_features (?).
        #root_length influences element_types, which enters in mining in sort_rules (and clustering) (within sample_tally).
        
        if 'features_removed' in kwargs and type(kwargs['features_removed']) is int:
            setattr(self, 'features_removed', kwargs['features_removed'])
        else:
            setattr(self, 'features_removed', 0)

        if 'indices_to_mine' in kwargs and type(kwargs['indices_to_mine']) is list:
            setattr(self, 'indices_to_mine', kwargs['indices_to_mine'])
        else:
            setattr(self, 'indices_to_mine', None)
        
        if 'num_features' in kwargs and type(kwargs['num_features']) is int:
            setattr(self, 'num_features', kwargs['num_features'])
        else:
            setattr(self, 'num_features', 1)
        
        if 'max_value_features' in kwargs:
            setattr(self, 'max_value_features', kwargs['max_value_features'])
        else:
            setattr(self, 'max_value_features', None)
            
        if 'extract_labels' in kwargs and type(kwargs['extract_labels']) is str:
            setattr(self, 'extract_labels', kwargs['extract_labels'])
        else:
            setattr(self, 'extract_labels', 'cartesian')
            
        if 'action_labels' in kwargs:
            setattr(self, 'action_labels', kwargs['action_labels'])
        else:
            setattr(self, 'action_labels', None)
            print('\nNo action_labels were given to the sequence processor. They will be automatically computed.')
        
        if 'sequence_length' in kwargs and type(kwargs['sequence_length']) is int:
            setattr(self, 'sequence_length', kwargs['sequence_length'])
        else:
            setattr(self, 'sequence_length', None)
            
        
        if self.file_path:
            data = np.load(self.file_path)
            self.sequence_length = (len(data[0][self.indices_to_mine]) if self.indices_to_mine else len(data[0]))
            self.num_features = len(data[0][0])
            self.max_value_features = self.get_max_value(data)
            #label generation
            if not self.action_labels:
                if self.num_features == 1:
                    self.action_labels = self.generate_labels(data, 'simple')
                else:
                    self.action_labels = self.generate_labels(data, self.extract_labels)
                    
        else:
            warnings.warn('No file path was introduced. \n "num_features" is set to '+str(self.num_features)+'.\n')
            if not self.sequence_length:
                raise NotImplementedError('User must input "sequence_length".')
            if not self.action_labels:
                raise NotImplementedError('User must input "action_labels".')
            if not self.max_value_features:
                raise NotImplementedError('User must input "max_value_features".')
                

        if self.root_length > self.num_features:
            self.root_length = 1
            warnings.warn("Root_length is too long, reset to 1")
            
        if self.features_removed >  self.num_features - self.root_length:
            self.features_removed =  self.num_features - self.root_length
            warnings.warn("Too many features were removed. Number of removed features set to maximal number of digits that can be removed")

        #for featured actions: determine which features are the main ones according to the choice of root_length.
        self.element_types = self.feature_root_info()
        
        #dictionary from the cut label (where some features have been removed) to the possible original labels it may have come from.
        #(only) useful for environments with featured actions.
        
        if self.num_features > 1:
            self.cut_to_full_label = {}
            #create entries.
            for label in self.action_labels:
                if str(label[:-self.features_removed]) not in self.cut_to_full_label:
                    self.cut_to_full_label.update({str(label[:-self.features_removed]): []})
            #populate entries.
            for label in self.action_labels:
                self.cut_to_full_label[str(label[:-self.features_removed])].append(label)
        
    
    def mining_preprocessing(self, raw_data, input_size):
        """
        Preprocessing for mining without clustering.

        Parameters
        ----------
        raw_data : np.array of arrays
            Data set to be mined.
        input_size : int
            Maximum size of data set that is passed to sequence mining.

        Returns
        -------
        classes : list
            List with data set to be mined.
        input_data : list
            List with data set to be mined.

        """
        if self.indices_to_mine:
            print('ERROR HERE',self.indices_to_mine)
            raw_data = [sample[self.indices_to_mine] for sample in raw_data]
        
        data = self.full_data_processing(raw_data)
        
        input_data = random.sample(list(data), k=(input_size if input_size<len(data) else len(data)))
        
        classes = [input_data]
    
        return classes, input_data    
    
    def full_data_processing(self, data_set):
        """
        Given a data set, it processes the sequences: (i) it transforms them to the sequence mining notation and (ii) removes
        the features chosen by the user.
        
        Input
        -------
        data_set (np.array): data set containing the sequences that are to be mined and/or evaluated.
            
        Returns
        -------
        Processed data set (list)

        """
        return [self.data_formatting(sample) for sample in self.feature_removal(data_set)]
    
    
    def data_formatting(self,sample):
        """Transforms environment notation to sequence mining notation (with all digits). """
        
        new_sample = [[] for el in sample]
        for cont, el in enumerate(sample):
            new_el = ','.join([str(n) for n in el])
            new_sample[cont] = new_el
        
        return new_sample
    
    
    def feature_removal(self, data_set):
        """
        Removes the information of the last features_removed digits of the elements of each sequence in the data set.
        
        Default: features_removed=0 (for environments with no featured actions), so the data set remains untouched.
        
        """

        if self.features_removed == 0:
            return data_set
        else:
            return [sample[:, :-self.features_removed] for sample in data_set]
        
    #------ methods to infer information about the sequences directly from data set ------#
    
    def feature_root_info(self):
        """
        Computes the list of element types, given the root_length (how many features are considered to be the main/root information).

        Returns
        -------
        element_types : List
            List with the element types that are considered to be the root information.
            (e.g. ['1,1,', '1,2,', '2,1,', '2,2,'], corresponding to ['G1', 'G2', 'M1', 'M2'] in the quantum computing env).

        """
        
        element_types = []

        for label in self.action_labels:
            label_root_str = (',').join([str(component) for component in label[:self.root_length]])
            if label_root_str not in element_types:
                element_types.append(label_root_str)
                
        return element_types
        
    def generate_labels(self, data, mode):
        """
        Generate labels from data set. Used for featured actions. It first extracts the maximum
        value of each feature and then generates the cartesian product of all possible labels.

        Parameters
        ----------
        data : np.array
            Data set with all sequences to be mined.
        mode: str
            Type of label extraction

        Returns
        -------
        labels : list of np.arrays
            List with labels.

        """
        #TODO: right now, the minimum value of the actions can be 0. Change this if we want to enforce it to be 1 (in order to leave 0 for random)
        #TODO: In case we change it, remember to change it in sim_distance (inside one-hot-encoding) as well .
        
        if mode == 'simple':
            return [np.array([x]) for x in range(self.max_value_features+1)]
        
        else:
           
            min_action_value =  min(data.flatten())
            
            labels = [np.array(c) for c in product(*[np.arange(min_action_value, i+1) for i in self.max_value_features])]
            
            if mode == 'cartesian':
                return labels
            elif mode == 'checked_cartesian':
                return self.check_labels(labels, data)
            
        
    def check_labels(self, labels, data):
        """
        Checks that all labels in the list actually appear in the data set.

        Parameters
        ----------
        labels : list of np.arrays.
            List with labels generated in self.generate_labels.
        data : np.array
            Data set with all sequences to be mined.

        Returns
        -------
        checked_labels : list of np.arrays
            List with labels that actually are in the data set.
        """
        
        checked_labels = []
        for label in labels:
            remove = True
            for seq in data:
                if (label == seq).all(1).any():
                    remove = False

            if not remove:
                checked_labels.append(label)
                
        return checked_labels
        
    def get_max_value(self, data):
        """
        Gets the maximum value of each feature.

        Parameters
        ----------
        data : np.array
            Data set with all sequences to be mined.

        Returns
        -------
        max_elements : max value of each feature.
            If num_features = 1, returns int with the maximum value in data
            If num_features >1, returns list with max value of each feature in each entry.

        """
        if self.num_features == 1:
            max_elements =  max(data.flatten())
        else:
            max_elements = [0]*len(data[0][0])
        
            for feature in range(self.num_features):
                max_elements[feature] = max([max(seq[:,feature]) for seq in data])
    
        return max_elements
    
    def sample_tally(self, sample):
        """
        Counts the number of elements per type, between (G1 G2 M1 M2).
    
        Parameters
        ----------
        sample : list
            List of elements. Each element is a list with 4 integers, as for the tree representation of the allowed operations.
    
        Returns
        ----------
        numpy.array
            Array with Counts for each type of element, between (G1 G2 M1 M2).
        """
    
        elements_Count = np.zeros(shape=len(self.element_types))
        for el in sample:
            numbers = el.split(',')
            root = ','.join([str(n) for n in numbers[:self.root_length]])
            elements_Count[self.element_types.index(root)] += 1
    
        return elements_Count