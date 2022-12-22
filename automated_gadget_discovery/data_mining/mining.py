import numpy as np
import random
from sys import argv, stdout
import os
import pathlib
import warnings

from data_mining.seq_min_methods import SeqMinMethods
from data_mining.seq_processor import SeqProcessor


class Mining(SeqMinMethods):
    def __init__(self, file_path, results_folder, **kwargs):
        """
        Mining class that mines the data obtained by the agent
        Args:
            file_path (str) retrieve the data the agent generated to be mined.
            results_folder (str) path of the folder where results from mining will be saved.
            
        **kwargs:
            cycle (int) cycle number
            run (int) id for the current run of the experiment (several cycles).
            supplementary_folder (str) path where supplementary information from the mining is stored.
            mining_dict (dict) dictonary with mining configurations
                
            Main parameters to be tuned (in mining_dict):
                
                MIN_L (int) minimal length of the sequences to be mined
                MIN_SUP: (float) minimum support threshold
                MIN_COH: (float) minimum cohesion threshold
                MIN_CONF: (float) minimum confidance threshold
            
                COVERAGE_FILTER: (bool) turns on coverage filter
                MAX_COVERAGE: (int) max # of times a sequence can be covered by a rule (the lower, the more it cuts)
                EMERGENCE_FILTER: (bool) turns on emergence filter
                FILTER_RATIO: (list) List ([min_ratio, max_ratio]) with the minimum and maximum fraction of rules that we allow to cut
                
        """

        super().__init__(**kwargs)

        self.file_path = file_path
        self.results_folder = results_folder
        
        if 'supplementary_folder' in kwargs and type(kwargs['supplementary_folder']) is str:
            setattr(self, 'supplementary_folder', kwargs['supplementary_folder'])
        else:
            setattr(self, 'supplementary_folder', '')
            
        #create folder if necessary
        pathlib.Path(self.results_folder+self.supplementary_folder).mkdir(parents=True, exist_ok=True)

        
        if 'cycle' in kwargs and type(kwargs['cycle']) is int:
            setattr(self, 'cycle', kwargs['cycle'])
        else:
            setattr(self, 'cycle', 0)
        
        if 'run' in kwargs and type(kwargs['run']) is int:
            setattr(self, 'run', kwargs['run'])
        else:
            setattr(self, 'run', 0)
            
        if 'mining_dict' in kwargs and type(kwargs['mining_dict']) is dict:
            setattr(self, 'mining_dict', kwargs['mining_dict'])
        else:
            setattr(self, 'mining_dict', {})

        #HYPERPARAMETRIZATION
        if 'HYPERPAR' in self.mining_dict:
            self.with_hyperpar = self.mining_dict['HYPERPAR']
        else:
            self.with_hyperpar = True
            
        if 'NUM_PATTERNS_RANGE' in self.mining_dict:
            self.num_patterns_range = self.mining_dict['NUM_PATTERNS_RANGE']
        else:
            self.num_patterns_range = [1,20]
            
        #CLUSTERING parameters
        if 'CLUSTER_ON' in self.mining_dict['clustering']:
            self.cluster_on = self.mining_dict['clustering']['CLUSTER_ON']
        else:
            self.cluster_on = True
            
        if 'INPUT_SIZE' in self.mining_dict['clustering']:
            self.input_size = self.mining_dict['clustering']['INPUT_SIZE']
        else:
            self.input_size = 300000
            
        if 'UTILITIES' in self.mining_dict['clustering'] and len(self.mining_dict['clustering']['UTILITIES'])==len(self.seq_proc.element_types):
            self.utilities = np.array(self.mining_dict['clustering']['UTILITIES'])
        else:
            self.utilities = np.array([1]*len(self.seq_proc.element_types))

            
        if 'MIN_SIZE' in self.mining_dict['clustering']:
            self.min_size = self.mining_dict['clustering']['MIN_SIZE']
        else:
            self.min_size = 35000
            
        if 'MIN_SAMPLES' in self.mining_dict['clustering']:
            self.min_samples = self.mining_dict['clustering']['MIN_SAMPLES']
        else:
            self.min_samples = 20
            
        if 'ENCODING' in self.mining_dict['clustering']:
            self.encoding = self.mining_dict['clustering']['ENCODING']
        else:
            self.encoding = 'vec_resource'
            
        #SEQUENCE MINING parameters
        if 'MIN_SUP' in self.mining_dict['seqmining']:
            self.min_sup = self.mining_dict['seqmining']['MIN_SUP']
        else:
            self.min_sup = 0.01

        if 'MIN_COH' in self.mining_dict['seqmining']:
            self.min_coh = self.mining_dict['seqmining']['MIN_COH']
        else:
            self.min_coh = 0.6
        
        if 'FACTOR_MIN_INT' in self.mining_dict['seqmining']:
            self.factor_min_int = self.mining_dict['seqmining']['FACTOR_MIN_INT']
        else:
            self.factor_min_int = 1

        self.min_int = self.factor_min_int*self.min_sup * self.min_coh

        if 'MIN_CONF' in self.mining_dict['seqmining']:
            self.min_conf = self.mining_dict['seqmining']['MIN_CONF']
        else:
            self.min_conf = 0.75

        if 'M' in self.mining_dict['seqmining']:
            self.M = self.mining_dict['seqmining']['M']
        else:
            self.M = 1
            
        if 'MIN_L' in self.mining_dict['seqmining']:
            self.min_l = self.mining_dict['seqmining']['MIN_L']
        else:
            self.min_l = 2




            
        #FILTERS parameters.
        
        #by length
        if 'CRITERION' in self.mining_dict['filters']:
            self.criterion = self.mining_dict['filters']['CRITERION']
        else:
            self.criterion = 'all' #'all' effectively deactivates this filter.
        
        if 'MIN_LENGTH' in self.mining_dict['filters']:
            self.min_length = self.mining_dict['filters']['MIN_LENGTH']
        else:
            self.min_length = 2
            
        #by coverage
        if 'COVERAGE_FILTER' in self.mining_dict['filters']:
            self.coverage_filter = self.mining_dict['filters']['COVERAGE_FILTER']
        else:
            self.coverage_filter = True
        
        if 'MAX_COVERAGE' in self.mining_dict['filters']:
            self.max_coverage = self.mining_dict['filters']['MAX_COVERAGE']
        else:
            self.max_coverage = 2
        
        #by emergence
        if 'EMERGENCE_FILTER' in self.mining_dict['filters']:
            self.emergence_filter = self.mining_dict['filters']['EMERGENCE_FILTER']
        else:
            self.emergence_filter = True
            
        if not self.cluster_on: #automatically turns off the filter by emergence if clustering is deactivated.
            self.emergence_filter = False
        
        if 'FILTER_RATIO' in self.mining_dict['filters']:
            self.filter_ratio = self.mining_dict['filters']['FILTER_RATIO']
        else:
            self.filter_ratio = [0.33,0.66]
            
        if 'MIN_R_CL' in self.mining_dict['filters']:
            self.min_r_cl = self.mining_dict['filters']['MIN_R_CL']
        else:
            self.min_r_cl = 0
            
        #by reward
        if 'REWARD_FILTER' in self.mining_dict['filters']:
            self.reward_filter = self.mining_dict['filters']['REWARD_FILTER']
        else:
            self.reward_filter = False
            
        if 'MIN_RATIO_REWARDS' in self.mining_dict['filters']:
            self.min_ratio_rewards = self.mining_dict['filters']['MIN_RATIO_REWARDS']
        else:
            self.min_ratio_rewards = 0.4
        
        #VISUALIZATION parameters.
        if 'L_VIEW' in self.mining_dict:
            self.l_view = self.mining_dict['L_VIEW']
        else:
            self.l_view = 2


    def mining(self):    
        """
        Full process that manages the sequence mining:
            1. Loads and preprocess input data
            2. Runs sequence mining. User choice: (i) with/without hyperparametrization, (ii) with/without clustering.
            3. Saves data (rules, mining parameters)

        Returns
        -------
        rule_output : dict
            Dictionary with rules.

        """
        raw_data = np.load(self.file_path)
        
        classes, S = self.seq_proc.mining_preprocessing(raw_data, input_size=self.input_size)
        
        if self.with_hyperpar:
            rule_output = self.sm_hyperpar(classes, S, num_rule_range=self.num_patterns_range)
            #save final parameters
            np.save(self.results_folder+self.supplementary_folder+'optimized_SM_parameters_cycle_{}_run_{}.npy'.format(self.cycle, self.run), {'F':self.min_sup, 'C':self.min_coh, 'I':self.min_int, 'Factor_I':self.factor_min_int, 'Conf':self.min_conf, 'M':self.M })
            #reset parameters to original values
            self.parameter_reset()
        else:
            if self.cluster_on:
                rule_output = self.seqmin_with_clustering(np.load(self.file_path))
            else:
                rule_output = self.seqmin(classes, S)
            np.save(self.results_folder+self.supplementary_folder+'SM_parameters_cycle_{}_run_{}.npy'.format(self.cycle, self.run), {'F':self.min_sup, 'C':self.min_coh, 'I':self.min_int, 'Factor_I':self.factor_min_int, 'Conf':self.min_conf, 'M':self.M })

        #save rules
        np.save(self.results_folder+self.supplementary_folder+'final_rules_cycle_{}_run_{}.npy'.format(self.cycle, self.run), rule_output)
           
        return rule_output
    
    def parameter_reset(self):
        self.min_sup = self.mining_dict['seqmining']['MIN_SUP']
        self.min_coh = self.mining_dict['seqmining']['MIN_COH']
        self.min_int = self.factor_min_int*self.min_sup * self.min_coh
        self.min_conf = self.mining_dict['seqmining']['MIN_CONF']
        self.M = 1
    
    def seqmin(self, classes, S):
        """
        Sequence mining (independent of clustering).

        Parameters
        ----------
        classes : list with classes to mine. The default is to have one class containing the data set to mine.
            
        S : list without the division in classes.

        Returns
        -------
        rule_output : dict
            dictionary with the final rules after all the filters.

        """
        raw_rules = self.sequence_mining(S, classes, self.min_sup, self.min_coh, self.factor_min_int, self.min_conf, m=self.M)
        filtered_Rules = self.filter_rules(S, classes, raw_rules, filter_ratio=self.filter_ratio, resource=self.utilities, min_R_cl=self.min_r_cl,
                                        min_L=self.min_length, criterion=self.criterion, coverage_filter=self.coverage_filter, max_coverage=self.max_coverage,
                                        emergence_filter=self.emergence_filter,
                                        reward_filter=self.reward_filter, min_ratio_Rewards=self.min_ratio_rewards, L_view=self.l_view)
        
        rule_output = {}
        for rule in filtered_Rules:
            P, P_dict = self.describe_rule(rule[0], S, classes[rule[1]], 4)
            rule_output.update({str(P):[P,P_dict['Support'],P_dict['Cohesion'],P_dict['Interest'],P_dict['Confidence']]})
        
        return rule_output
    
    def seqmin_with_clustering(self, o_r_raw):
        """
        Sequence mining full algorithm.
        
        Input
        ---------
        o_r_raw (np.array, from .npy file): raw data to mine

        Output
        ---------
        rule_output (dict): dictionary with the final rules after all the filters.
        
        Each entry (for each pattern P) is of the form: 
            key: str(P)
            value: (list) [P, F(P), C(P), I(P), confidence(P)], where F, C, I are support, cohesion and interestingness, respectively.
        """
        if self.seq_proc.indices_to_mine:
            o_r_raw = [sample[self.seq_proc.indices_to_mine] for sample in o_r_raw]
            
        o_r = [self.seq_proc.data_formatting(sample) for sample in o_r_raw]
    
        while True:
    
            input_data = np.asarray(random.sample(list(o_r), k=(self.input_size if self.input_size<len(o_r) else len(o_r))))
            clusters, _ = self.clustering(input_data, u=self.utilities, min_size=self.min_size, min_samples=self.min_samples, encoding=self.encoding)
            
            if len(clusters) == 2:

                classes, S = self.preprocessing(clusters, max_cluster_size=30000)
                raw_rules = self.sequence_mining(S, classes, self.min_sup, self.min_coh, self.factor_min_int, self.min_conf, m=self.M)
                
                N_nonEmpty_clusters = len(set([r[1] for r in raw_rules if len(r[0]) >= self.min_l]))
    
                if len(clusters) == N_nonEmpty_clusters:
                    break
                else:
                    print("Some clusters do not have " + str(self.min_l) + "-rules with settings (" + str(
                        round(self.min_sup, 3)) + " " + str(round(self.min_int, 3)) + ")\n")
                    print("_________________________________________")
                    self.min_sup *= 0.95
                    self.min_int *= 0.95
                    self.M *= 0.95
            else:
                print("\n Clustering found " + str(
                    len(clusters)) + " clusters, instead of 2.\nWe will randomly sample new data and cluster again.\n")
                pass
    
        filtered_Rules = self.filter_rules(S, classes, raw_rules, filter_ratio=self.filter_ratio, resource=self.utilities, min_R_cl=self.min_r_cl,
                                        min_L=self.min_length, criterion=self.criterion, coverage_filter=self.coverage_filter, max_coverage=self.max_coverage,
                                        emergence_filter=self.emergence_filter,
                                        reward_filter=self.reward_filter, min_ratio_Rewards=self.min_ratio_rewards, L_view=self.l_view)
   
        #final_Rules = self.present_rules(S, classes, filtered_Rules, self.utilities, L_view=self.l_view, print_output=True,
        #                               arrange="by_resource", encoding=self.encoding, detailed=self.detailed, readable=True,
        #                               permutations=self.permutations)
    
        # For the moment, it outputs a dictionary of patterns without any visualization nor equivalence considered.
        rule_output = {}
        for rule in filtered_Rules:
            P, P_dict = self.describe_rule(rule[0], S, classes[rule[1]], 4)
            rule_output.update({str(P):[P,P_dict['Support'],P_dict['Cohesion'],P_dict['Interest'],P_dict['Confidence']]})
        
        return rule_output
    
    def sm_hyperpar(self, classes, S, num_rule_range, sup_opt_threshold=300):
        """
        This function adjusts the parameters min_sup, min_coh, min_conf of the sequence mining to obtain # of rules in the specified range.
        
        Input
        -------
        o_r_raw (np.array, from .npy file): raw data to mine
        num_rule_range: (list) [min_#rules, max_#rules] full_seqmin(.) should output more than min_#rules and less than max_#rules rules.
        sup_opt_threshold: (int) initial threshold to start optimizing the support (cohesion and confidence are fixed and support is varied
                               until the output number of rules is below this threshold).
        Output
        ---------
        rule_output (dict): dictionary with the final rules after all the filters.
        
        Each entry (for each pattern P) is of the form: 
            key: str(P)
            value: (list) [P, F(P), C(P), I(P), confidence(P)], where F, C, I are support, cohesion and interestingness, respectively.
        """
        
        alternate = 1
        print('\nInitial parameters: MIN_F='+str(self.min_sup)+', MIN_C='+str(self.min_coh)+', MIN_CONF='+str(self.min_conf))
        print('\nseqmin(.) should output between '+str(num_rule_range[0])+' and '+str(num_rule_range[1])+' rules. \n')
        
        while True:
            
            if self.cluster_on:
                rule_output = self.seqmin_with_clustering(np.load(self.file_path))
            else:
                rule_output = self.seqmin(classes, S)
            
            if len(rule_output) > sup_opt_threshold:
                self.min_sup += 0.01
                self.min_int = self.factor_min_int*self.min_sup * self.min_coh
                print('\n**Upper threshold:'+str(sup_opt_threshold)+' rules.** '+str(len(rule_output))+' rules were selected, support is increased to__MIN_F='+str(self.min_sup)+'\n')
            
            else:
                if len(rule_output) < num_rule_range[0]:
                    self.min_sup -= 0.003
                    self.min_int = self.factor_min_int*self.min_sup * self.min_coh
                    print('\n**Too few rules were selected.** Support is decreased to__MIN_F='+str(self.min_sup)+'\n')
                    
                    if self.min_sup < 0.00001: #stopping condition
                        self.min_sup += 0.003 #reset to previous values, which are the ones that will be saved (used in the last mining round).
                        self.min_int = self.factor_min_int*self.min_sup * self.min_coh 
                        warnings.warn('\nMining process has been stopped. MIN_F is too low for mining. \n The output corresponds to mining with MIN_F='+str(self.min_sup)+', which yielded '+str(len(rule_output))+' patterns.')
                        return rule_output
                    
                elif len(rule_output) > num_rule_range[1]:
                    if alternate:
                        self.min_sup += 0.005
                        self.min_int = self.factor_min_int*self.min_sup * self.min_coh
                        alternate = (alternate + 1)%2
                        print('\n**Too many rules were selected.** Support is increased to__MIN_F='+str(self.min_sup)+'\n')
                        
                    else:
                        self.min_coh += 0.01
                        self.min_conf += (0.01 if self.min_conf <= 0.99 else 0)
                        self.min_int = self.factor_min_int*self.min_sup * self.min_coh
                        alternate = (alternate + 1)%2
                        print('\n**Too many rules were selected.** Cohesion and confidence are increased to__MIN_C='+str(self.min_coh)+'___MIN_CONF='+str(self.min_conf)+'\n')

                else:
                    print('\nFinal parameters: MIN_F='+str(round(self.min_sup,3))+', MIN_C='+str(self.min_coh)+', MIN_I='+str(round(self.min_int,3))+', MIN_CONF='+str(self.min_conf)+'. \n Total #rules: '+str(len(rule_output))+'\n')

                    return rule_output
                
    
           

if __name__ == "__main__":
    
    #FILE PATH
    file_path = 'D:/DATA/results/ddqn_cycle_exchanger/exp_1/data_cycle_0_run_0.npy'
    #result folder path
    result_folder = 'results/sequence_mining/test/'
    
    #Initialize sequence processor
    ROOT_LENGTH = 2
    FEATURES_REMOVED = 1
    INDICES_TO_MINE = [0,1,3,4,6,7]
    EXTRACT_LABELS = 'checked_cartesian'
    seq_processor = SeqProcessor(file_path=file_path, root_length=ROOT_LENGTH, features_removed=FEATURES_REMOVED, indices_to_mine=INDICES_TO_MINE, extract_labels=EXTRACT_LABELS)
    
    mining_dict = { 'HYPERPAR': True,
                    'NUM_PATTERNS_RANGE': [1,10],
                    'clustering':
                               {'CLUSTER_ON':False,
                                'INPUT_SIZE':80000,
                                'UTILITIES':[1,1,1,1],
                                'MIN_SIZE':35000},
                    'seqmining':
                                {'MIN_SUP':0.14,
                                 'MIN_COH':0.8,
                                 'MIN_CONF':0.99},
                                
                    'filters':
                                {'CRITERION': 'discard_long',
                                 'COVERAGE_FILTER':True,
                                 'MAX_COVERAGE':2,
                                 'EMERGENCE_FILTER':False,
                                 'FILTER_RATIO':[0.33,0.66],
                                 'REWARD_FILTER':False},        
                    'visualization':
                               {'L_VIEW':2}
                                }
    
    
    mining = Mining(file_path, result_folder, mining_dict=mining_dict, seq_proc=seq_processor)
        
    rule_output = mining.mining()
    