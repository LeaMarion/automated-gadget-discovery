import numpy as np
import warnings

from data_mining.pattern import Pattern

class PatternManager():
    def __init__(self, features_removed, **kwargs):
        
        """
        Class that manages the patterns obtained in the mining process.
        It contains the functions that update, compute and store the pattern information.
        
        args:
            features_removed (int) : for actions with features, how many features are removed (ignored in mining and/or evaluation).
            
        """
        
        self.features_removed = features_removed
        
        #Dictionary that stores the patterns and their classes (which contains info about their F, C, I).
        self.pattern_dict = {}
        
        
    def pattern_update(self, P, current_F, current_C, current_I, cycle=0, agent=0):
        """Function that adds the pattern to the dictionary and updates its information if it is already there.
    
        Input
        --------
        P (list): pattern definition e.g. ['G2_ab', 'G2_ac']
        current_F (float): current support of pattern P
        current_C (float): current cohesion of pattern P
        current_I (float): current interestingness of pattern P
    
    
        """
        id = len(self.pattern_dict)
        pattern = Pattern(P, id, current_F, current_C, current_I, self.features_removed)
        
        if str(pattern.name) in self.pattern_dict:
            print('Pattern '+str(pattern.name)+' is already in the dictionary. Found by agents:', self.pattern_dict[str(pattern.name)].agent)
            print ('Values of support, cohesion and interestingness are those corresponding to the first agents dataset and will not be updated. ')
            self.pattern_dict[str(pattern.name)].agent.append(agent)
            
        else:
            self.pattern_dict[str(pattern.name)] = pattern
            self.pattern_dict[str(pattern.name)].cycle.append(cycle)
            self.pattern_dict[str(pattern.name)].agent.append(agent)
            
        return pattern
            
        
    def focus_update(self, P, Iarray_pos, Iarray_neg):
        """Function that updates information on pattern P with the results from the focus phase evaluation.
    
        Input
        --------
        P (list): pattern definition e.g. ['G2_ab', 'G2_ac']
        Iarray_pos (np.array): array with the I of the positively rewarded data sets of the focus cycles.
        Iarray_neg (np.array): array with the I of the negatively rewarded data sets of the focus cycles.
    
        """

        if P in self.pattern_dict:
            if np.mean(Iarray_neg) != 0:
                self.pattern_dict[str(P)].focus_ratio_I = np.mean(Iarray_pos)/np.mean(Iarray_neg)
                self.pattern_dict[str(P)].std_focus_ratio_I = np.sqrt((np.std(Iarray_pos)) ** 2 / (np.mean(Iarray_pos)) ** 2 \
                                                                + (np.std(Iarray_neg)) ** 2 / (np.mean(Iarray_neg)) ** 2)
            else:
                self.pattern_dict[str(P)].focus_ratio_I = 'Inf'
                
        else:
            print(self.pattern_dict)
            print('Pattern '+str(P)+' is not in the dictionary yet. Information from the focus phase cannot be added.')

    def all_pattern_ranking(self, pattern_dict=None, threshold_number=None):
        """
         Function that ranks the patterns in pattern_list according to their interestingness.
         It outputs the threshold_number best patterns.

         Input
         --------
         pattern_dict (dict): (default self.pattern_dict) dictionary of patterns to be ranked.
         threshold_number (int): (default None) number of top best patterns that the function outputs.

         Output
         --------
         ranking (list): list of patterns, sorted in descending order according to Interestingness.

         """
        if pattern_dict == None:
            pattern_dict = self.pattern_dict

        ranking = [[pattern_dict[p], pattern_dict[p].I] for p in pattern_dict.keys()]
    
        # Rank patterns in descending order according to the interestingness (specified with the function Sort_func)
        ranking.sort(reverse=True, key=self.func_sort)
    
        return ranking[:threshold_number]

    def all_pattern_focus_ranking(self, threshold_number):
        """
         Function that ranks the patterns in pattern_list according to their interestingness.
         It outputs the threshold_number best patterns.

         Input
         --------
         threshold_number (int): number of top best patterns that the function outputs.

         Output
         --------
         ranking (list): list of patterns, sorted in descending order according to the focus ratio of Interestingness.

         """

        ranking = [[self.pattern_dict[p], self.pattern_dict[p].focus_ratio_I] for p in self.pattern_dict.keys()]

        # Rank patterns in descending order according to the interestingness (specified with the function Sort_func)
        ranking.sort(reverse=True, key=self.func_sort)

        # TODO: include maybe other threshold (e.g. with I>I_threshold or some function of the reliability)
        return ranking[:threshold_number]
    
    def func_sort(self, entry):
        """
        Function that specifies the criteria for the ranking. In this case, patterns are sorted by interestingness.
    
        It assumes that each entry of the list to be sorted has the form: [pattern_name, interestingness]
        """
        return entry[1]
    
    
    def pattern_processing (self, mined_rules, number_patterns=None, cycle=0, agent=0):
        """ Function that processes the output of the mining: it updates the pattern dictionary, it ranks the patterns given in mined_rules
        and it outputs the most interesting patterns with the notation that the agent can read.
        
        Input
        ------
        mined_rules (list): list with the rules (patterns), where each entry is of the form [pattern_name, F, C, I, confidence].
        number_patterns (int): number of top best mined patterns that will be added to the main dictionary and output for the agent to use. \
            (default=None, all mined patterns are added) 
        cycle (int): cycle number (default=0)
                            
        Output
        ------
        pattern_list (list): list with the patterns (classes) that the agent will use. 
        
        """
        
        # Rank patterns in mined_rules in descending order according to the interestingness.
        mined_rules.sort(reverse=True, key=lambda x:x[3])

        pattern_list=[]
        for pattern in mined_rules[:number_patterns]:
            pattern = self.pattern_update(pattern[0], pattern[1], pattern[2], pattern[3], cycle=cycle, agent=agent)
            pattern_list.append(self.pattern_dict[str(pattern.name)])
        
        return pattern_list


    def add_removed_features_to_name(self, name):
        """
        Adds the removed features to the name of the pattern
        Args:
            name (str) name string of pattern
        Returns:
            new_name (str) adds removed feature to all string elements of the name
        """
        print(name)
        for ele in name:
            print('ele', ele)
        if type(name[0])==str:
            new_name = [el+',0'*self.features_removed for el in name]
            return new_name

        return name


