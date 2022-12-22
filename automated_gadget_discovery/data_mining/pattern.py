import numpy as np

class Pattern():
    def __init__(self, name, id, F,C,I, features_removed):
        """
        Class defined for each pattern that is obtained from the sequence mining (function: full_SM).
        It stores the pattern name (e.g. ['G2_ab', 'G2_ac']) and the current information of the pattern, i.e. support (F), cohesion (C) and interestingness (I).
        
        name (list) : output from sequence mining (function: full_SM)
        id (int): integer to identify the pattern
        F (float) : support of the pattern
        C (float) : cohesion of the pattern
        I (float) : interestingness of the pattern
        features_removed (int) : for actions with features, how many features are removed (ignored in mining and/or evaluation).
        
        """

        #Identification of the pattern
        self.id = id
        self.cycle = [] #list of cycles in which pattern appeared
        self.agent = [] #agents that found the pattern
        self.features_removed = features_removed

        #Name of the pattern, as feature vector
        name = self.add_removed_features_to_name(name)
        self.name = name

        #Notation of the pattern as actions from the environment
        self.action_list = [self.correspondence(el) for el in self.name]

        #Information about the pattern at the current cycle.
        self.F = F
        self.C = C
        self.I = I
        self.focus_ratio_I = 0.
        self.std_focus_ratio_I = 0.




    def correspondence(self,element):
        """
        Returns the action notation (the one the agent can read) that corresponds to an element of the pattern.
        
        """
        #TODO: not general, does  not work if it is shifted from  zero, but currently not used
        if type(element[0]) == str:  # if the notation is not the agent's notation
            numbers = element.split(',')
            label = np.array([int(num) for num in numbers])
            return label

        return element


    def add_removed_features_to_name(self, name):
        """
        Adds the removed features to the name of the pattern
        Args:
            name (str) name string of pattern
        Returns:
            new_name (str) adds removed feature to all string elements of the name
        """
        if type(name[0])==str:
            new_name = [el+',0'*self.features_removed for el in name]
            return new_name

        return name
