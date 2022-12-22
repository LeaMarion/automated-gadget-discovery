import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
from tabulate import tabulate
from operator import itemgetter
import random
import sys

from utils import get_config
import gym
import gym_quantum_computing

#-------------- IF NEEDED, INITIALIZE HERE THE ENV --------------------

#Initialize env
DIM = 2
NUM_QUDITS = 4
MAX_OP = 9
FIX_M = False
FIXED_OP = 3

env = gym.make('quantum-computing-v0', dim=DIM, num_qudits=NUM_QUDITS, max_op=MAX_OP, rand_start=False, fix_measurement=FIX_M, fixed_op=FIXED_OP)

cut_to_full_label = {}
#create entries.
for label in env.action_labels:
    if str(label[:-1]) not in cut_to_full_label:
        cut_to_full_label.update({str(label[:-1]): []})
#populate entries.
for label in env.action_labels:
    cut_to_full_label[str(label[:-1])].append(label)
    
#create a dictionary from action labels to actions. (This is useful in case actions have features)
labels_to_actions =  {str(None):None}
for i, action_label in enumerate(env.action_labels):
    labels_to_actions.update({str(action_label): i})

#-----------

class ResultsAnalysis(object):

    def __init__(self, results_folder, config, **kwargs):
        """
        Class for the analysis of the results

        Parameters
        ----------
        results_folder : str
            Path to the main folder where the results of the cycles are saved.
        config : cfg
            cfg file with the parameters chosen by the user for the full algorithm.
        **kwargs : 
            num_agents (int): (default 1) number of runs.
            plot_format (str): (default 'png', with dpi=200) chosen format for saving the plots.
            feature_dict (dict): in case of featured observations, this dictionary provides a translation from the used notation to a readable notation.
                The dictionary should be of the form: {feature index (int): readable notation (str)}
            color_list (list): list with colors for each cycle in the gadgets plot.
            with_env (bool): (default: false) whether there is an environment that is initialized outside of this class (needed e.g. in gadget_clustering to get the avg. reward of init sequences)

        """
        
        if 'num_agents' in kwargs and type(kwargs['num_agents']) is int:
            setattr(self, 'num_agents', kwargs['num_agents'])
        else:
            setattr(self, 'num_agents', 1)
        
        if 'plot_format' in kwargs and type(kwargs['plot_format']) is str:
            setattr(self, 'plot_format', kwargs['plot_format'])
        else:
            setattr(self, 'plot_format', 'png')
        
        if 'feature_dict' in kwargs and type(kwargs['feature_dict']) is dict:
            setattr(self, 'feature_dict', kwargs['feature_dict'])
        else:
            setattr(self, 'feature_dict', None)
        
        if 'color_list' in kwargs and type(kwargs['color_list']) is list:
            setattr(self, 'color_list', kwargs['color_list'])
        else:
            setattr(self, 'color_list', None)
            
        if 'with_env' in kwargs and type(kwargs['with_env']) is bool:
            setattr(self, 'with_env', kwargs['with_env'])
        else:
            setattr(self, 'with_env', False)
    
        
        self.results_folder = results_folder
        self.plots_folder = self.results_folder + 'analysis-and-plots/'
        pathlib.Path(self.plots_folder).mkdir(parents=True, exist_ok=True) 
        
        #get parameters from config 
        self.num_cycles = config['general']['CYCLES']
        self.episodes = config['general']['EPISODES'][0]
        self.data_collection_size = config['general']['DATACOLLECTION_SIZE']
        self.int_rescale = config['jiminy']['INT_RESCALE']
        
        self.seqmin_hyperpar = config['mining']['HYPERPAR']
        # self.p_threshold = config['exchanger']['PROB_THRESHOLD']
        
        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_cycles)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.tab20)
        if not self.color_list:
            self.color_list = [cmap.to_rgba(i+1) for i in range(self.num_cycles)]
        
    def learning_performance(self, agent_id=None, num_trials_avg=1000, cycle=0):
        """
        Analysis of the agents' performance during the learning process.

        Parameters
        ----------
        agent_id : int or list, optional
            Which agents (runs) are to be analysed. The default is None, which considers all runs.
        num_trials_avg : int, optional
            Number of trials to be averaged per point in the plot. The default is 1000.
        cycle : int
            Cycle number from which to extract the reward file. The default is 0.

        """
        
        #create a list with the identifiers of the analysed agents.
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None

        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
        
        
        plt.figure(figsize=(5.9,1.8))
        
        
        for ag in agent_list:
            #Load files with evolution of rewards.
            rwd_array = np.load(self.results_folder + 'exploration/' + 'reward_cycle_'+str(cycle)+'_run_'+str(ag)+'.npy')[:self.episodes]
            
            #Compute mean and std over num_trials_avg trials
            mean = np.zeros(int(self.episodes / num_trials_avg))
            std = np.zeros(int(self.episodes / num_trials_avg))
            
            for block in range(int(self.episodes / num_trials_avg)):
                mean[block] = np.mean(rwd_array[block*num_trials_avg : block*num_trials_avg + num_trials_avg])
                std[block] = np.std(rwd_array[block*num_trials_avg : block*num_trials_avg + num_trials_avg])
                    
            #Add subplot
            plt.subplot(1, len(agent_list), ag+1)
            plt.fill_between(np.arange(len(mean)),mean+std, mean-std, alpha = 0.2)
            plt.plot(mean, color='k')
            
            
            plt.title('Agent '+str(ag),fontsize=7)
            if ag == 0:
                plt.ylabel('Fraction of correct circuits',fontsize=7)
                
            plt.xlabel('Episode (x$10^3$)',fontsize=7)
            plt.axhline(y=1, color='red', linestyle='-', linewidth=0.5) #max reward
            
            #ticks
            ticks = np.arange(0, self.episodes+1, self.episodes/2)
            plt.xticks(ticks / num_trials_avg, labels=[int(t/1000) for t in ticks])
            plt.yticks([0, 0.5, 1])
            
            #set font size
            axes = plt.gca()
            axes.tick_params(axis='both', which='major', labelsize=7)
            
        plt.tight_layout()
        if agent_id:
            plt.savefig(self.plots_folder + 'learning_performance_avg_'+str(num_trials_avg)+'_trials_run_'+str(agent_id)+'.'+str(self.plot_format), dpi = 200)
        else:
            plt.savefig(self.plots_folder + 'learning_performance_avg_'+str(num_trials_avg)+'_trials.'+str(self.plot_format), dpi = 200)
        
    
    def data_collection_performance(self, agent_id=None, num_trials_avg=1000):
        """
        Analysis of the agents' performance during the data collection phase.

        Parameters
        ----------
        agent_id : int or list, optional
            Which agents (runs) are to be analysed. The default is None, which considers all runs.
        num_trials_avg : int, optional
            Number of trials to be averaged per point in the plot. The default is 1000.

        """
        #create a list with the identifiers of the analysed agents. 
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None
            
        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
            
        plt.figure(figsize=(12.5,10))
        
        #Initialize lists to save data for bar plot.
        cycle_failures = []
        cycle_failures_err = []
        cycle_pat_encounters = []
        cycle_pat_encounters_err = []
        
        for cycle in range(self.num_cycles):
            #Initialize array to stack performances of all agents under consideration.
            fail_array = np.zeros(2*self.data_collection_size)
            pat_array = np.zeros(2*self.data_collection_size)
            min_length = []
            
            for ag in agent_list:
                #Load files.
                reward_list = np.load(self.results_folder + 'exploration/' + 'reward_cycle_'+str(cycle)+'_run_'+str(ag)+'.npy')
                failure_array = 1-np.sign(np.sign(np.array(reward_list[self.episodes:]))+1)
                pattern_encountered = 1.*(self.int_rescale == np.array(reward_list[self.episodes:]))
                #Add loaded list to the array.
                fail_array = np.vstack((fail_array, np.concatenate((failure_array,np.zeros(len(fail_array)-len(failure_array))))))
                pat_array = np.vstack((pat_array, np.concatenate((pattern_encountered,np.zeros(len(pat_array)-len(pattern_encountered))))))
                #Add length of the actual data_collection list.
                min_length.append(len(reward_list[self.episodes:]))
            
            #Add data for bar plot
            min_length = min(min_length)
            cycle_failures.append(np.mean(fail_array[1:,:min_length]))
            cycle_failures_err.append(np.std(fail_array[1:,:min_length]))
            cycle_pat_encounters.append(np.mean(pat_array[1:,:min_length]))
            cycle_pat_encounters_err.append(np.std(pat_array[1:,:min_length]))
            
            #Mean over the different agents
            fail_array = np.mean(fail_array[1:,:min_length],axis=0)
            pat_array = np.mean(pat_array[1:,:min_length],axis=0)
            # std_agents = np.std(rwd_array[1:,:min_length],axis=0)
            
            #Compute mean and std over num_trials_avg trials
            mean = np.zeros(int(min_length / num_trials_avg))
            std = np.zeros(int(min_length / num_trials_avg))
            
            mean_pat = np.zeros(int(min_length / num_trials_avg))
            std_pat = np.zeros(int(min_length / num_trials_avg))
            
            for block in range(int(min_length / num_trials_avg)):
                mean[block] = np.mean(fail_array[block*num_trials_avg : block*num_trials_avg + num_trials_avg])
                std[block] = np.std(fail_array[block*num_trials_avg : block*num_trials_avg + num_trials_avg])
                
                mean_pat[block] = np.mean(pat_array[block*num_trials_avg : block*num_trials_avg + num_trials_avg])
                std_pat[block] = np.std(pat_array[block*num_trials_avg : block*num_trials_avg + num_trials_avg])
            
            print('\n In cycle ', cycle, ', ',round(100*np.mean(mean_pat)/np.mean(mean),2) ,'% of the failures were due to the agent using a known pattern.')
            
            
            #Add subplot
            if self.num_cycles > 5:
                plt.subplot(round(self.num_cycles / 2),2,cycle+1)
            else:
                plt.subplot(self.num_cycles,1,cycle+1)
            plt.fill_between(np.arange(len(mean)),mean+std, mean-std, alpha = 0.2)
            plt.plot(mean, color='k', label='All failures')
            
            plt.fill_between(np.arange(len(mean_pat)),mean_pat+std_pat, mean_pat-std_pat, alpha = 0.2)
            plt.plot(mean_pat, color='r', label='Known pattern used')
            
            plt.title('Cycle '+str(cycle),fontsize=7)
            plt.ylabel('Fraction of failures',fontsize=7)
            plt.xlabel('Training epoch $e$',fontsize=7)
            plt.legend(fontsize=7)
            
            #ticks
            ticks = np.arange(0, min_length+1, min_length/2)
            plt.xticks(ticks / num_trials_avg, labels=[int(t) for t in ticks])
            
            #set font size
            axes = plt.gca()
            axes.tick_params(axis='both', which='major', labelsize=7)
            
        plt.tight_layout()
        
        if agent_id:
            plt.savefig(self.plots_folder + 'data_collection_performance_avg_'+str(num_trials_avg)+'_trials_run_'+str(agent_id)+'.'+str(self.plot_format), dpi = 200)
        else:
            plt.savefig(self.plots_folder + 'data_collection_performance_avg_'+str(num_trials_avg)+'_trials.'+str(self.plot_format), dpi = 200)
        
        #bar plot
        x = np.arange(self.num_cycles)
        width = 0.25  # the width of the bars
        fig, ax = plt.subplots(figsize=(self.num_cycles*0.8, 4))
        ax.bar(x, cycle_failures, width, color='k', label='All failures')
        ax.bar(x, cycle_pat_encounters,width, color='0.5', label='Known pattern used')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Fraction of failures',fontsize=7)
        ax.set_xlabel('Cycle',fontsize=7)
        ax.set_xticks(x)
        ax.legend(fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        fig.tight_layout()
        
        if agent_id:
            plt.savefig(self.plots_folder + 'failures_datacollection_run_'+str(agent_id)+'.'+str(self.plot_format), dpi = 200)
        else:
            plt.savefig(self.plots_folder + 'failures_datacollection.'+str(self.plot_format), dpi = 200)
 
    def gadgets(self, agent_id=None, with_plot=True):
        """
        Analysis of the gadgets obtained by the agents.

        Parameters
        ----------
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.
    
        with_plot : bool, optional
            Whether to plot the gadgets in a bar plot or not. The default is True.

        """
        
        #create a list with the identifiers of the analysed agents. 
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None
            
        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
            
        for ag in agent_list:
            print('\nAgent #', ag, '\n')
            #Load files.
            pattern_dict = np.load(self.results_folder + 'pattern_dict_run_'+str(ag)+'.npy', allow_pickle=True).item()
            
            if self.seqmin_hyperpar:
                for cycle in range(self.num_cycles):
                    param_dict = np.load(self.results_folder + 'sequence_mining/' + 'optimized_SM_parameters_cycle_' + str(cycle) + '_run_'+ str(ag) +'.npy', allow_pickle=True).item()
                    print('Final parameters in cycle '+str(cycle)+':\n', 'MIN_SUP', round(param_dict['F'],4), 'MIN_COH', param_dict['C'], 'MIN_I', round(param_dict['I'],4), 'MIN_CONF', param_dict['Conf'],'\n')
                    
            #Print table with patterns
            table_list = []
            print_cycles = []
            for p in pattern_dict.values():
                translated_name = self.translate(p.name)
                if p.cycle[0] < self.num_cycles:
                    print_cycles.append(p.cycle)
                    table_list.append([translated_name, p.I, p.F, p.C, p.cycle])

                
            print(tabulate(table_list, headers=['Pattern', 'I', 'Support', 'Cohesion', 'In cycles']))
            
            #Create csv file with ranked patterns
            table_list.sort(reverse=True, key=lambda x:x[1])
            #table_list.sort(reverse=True, key=itemgetter(4, 1))
            f = open(self.plots_folder + "gadgets_run_"+str(ag)+".csv","w+")
            f.write('Pattern;Interestingness;Support;Cohesion;In cycles \n')
            for pattern_list in table_list:
                f.write(str(pattern_list[0])+';'+ str(pattern_list[1])+';'+ str(pattern_list[2])+';'+ str(pattern_list[3])+';'+ str(pattern_list[4])+'\n')
            f.close()
            
            if with_plot:
                
                x = np.arange(len(table_list))
                width = 0.25  # the width of the bars
                
                #get color of bars depending on the cycle at which the pattern was obtained.
                bar_color = []
                for pattern_data in table_list:
                    bar_color.append(self.color_list[pattern_data[4][0]])
                
                #labels for legend that depend on the color of the bar
                colors = {}
                for cycle in range(self.num_cycles):
                    colors['Cycle '+str(cycle)] = self.color_list[cycle]
                labels = list(colors.keys())
                handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
                
                #plot
                fig, ax = plt.subplots(figsize=(len(table_list)*0.6, 4))
                ax.bar(x, [pattern_data[1] for pattern_data in table_list], width, color=bar_color)

                # Add some text for labels, title and custom x-axis tick labels, etc.
                ax.set_ylabel('Interestingness',fontsize=7)
                ax.set_title('Gadgets',fontsize=7)
                ax.set_xticks(x)
                ax.set_xticklabels([pattern_data[0] for pattern_data in table_list], rotation=30)
                ax.legend(handles, labels, fontsize=7)
                ax.tick_params(axis='both', which='major', labelsize=7)
                
                fig.tight_layout()
                plt.savefig(self.plots_folder + 'gadgets_run_'+str(ag)+'.'+str(self.plot_format), dpi = 200)
    
    
            
    def gadget_clustering(self, min_samples=None, dist_measure=None, agent_id=None, sup_folder=None):
        """
        Analysis of the results output by the gadget clustering.

        Parameters
        ----------
        min_samples: int, default=None. Parameter used in HDBSCAN. If None, value is taken from config file.
        dist_measure: str, default=None. Distance used to compute the distance matrix. If None, value is taken from config file
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.
            
        sup_folder: (str) supplementary folder where the cluster labels are saved.
        
        """
        #get parameters
        if min_samples == None:
            min_samples = config['classification']['MIN_SAMPLES']
        if dist_measure == None:
            dist_measure = config['classification']['SIM_MEASURE']
        if sup_folder == None:
           sup_folder = 'gadget_clustering/' + dist_measure + '/'
        
        #create a list with the identifiers of the analysed agents. 
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None
            
        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
        
         
        #Load files.
        pattern_dict = np.load(self.results_folder + 'pattern_dict_cycle_0_runs_'+str(agent_list)+'.npy', allow_pickle=True).item()
        
        cluster_labels = np.load(self.results_folder + sup_folder +'labels_'+dist_measure+'.npy')
        prob_cluster_labels = np.load(self.results_folder + sup_folder + 'prob_labels_'+dist_measure+'.npy')
        
        #Save classes into a csv file
        table_list = []
        for i, p in enumerate(pattern_dict.values()):
            #Pattern name
            translated_name = self.translate(p.name)
            
            #Analysis of its init set
            init_set = np.ndarray.tolist(np.load(self.results_folder + sup_folder +'init_set_pattern_'+str(i)+'.npy', allow_pickle=True))
            av_length = np.mean([len(circuit) for circuit in init_set])
            if self.with_env:
                all_reward = []
                for circuit in init_set:
                    env.reset()
                    for element in circuit:
                        random.shuffle(cut_to_full_label[str(np.array(element))])
                        action = cut_to_full_label[str(np.array(element))][0]
                        _,reward,_,_ = env.step(labels_to_actions[str(action)])
                        
                    all_reward.append(reward)
            
            if self.with_env:
                table_list.append([translated_name, cluster_labels[i], prob_cluster_labels[i], av_length, np.mean(all_reward), p.I, p.F, p.C, p.agent[0]])
                
            else:
                table_list.append([translated_name, cluster_labels[i], prob_cluster_labels[i], av_length, "None", p.I, p.F, p.C])
                    
        #Create csv file with ranked patterns according to their class
        
        table_list.sort(reverse=True, key=lambda x:x[1])
        f = open(self.plots_folder +  "gadget_clusters_all_info_"+str(dist_measure)+"_distance_runs_"+str(agent_list)+".csv","w+")
        f.write('Gadget;Cluster;Prob;InitSet Length;InitSet Reward;Interestingness;Agent \n')
        for pattern_list in table_list:
            f.write(self.to_latex(pattern_list[0])+';'+ 'C'+str(pattern_list[1])+';'+str(pattern_list[2])+';'+str(pattern_list[3])+';'+str(pattern_list[4])+';'+str(pattern_list[5])+';'+str(pattern_list[8])+'\n')
        f.close()
        
        
        #Plot
        distance_matrix = np.load(self.results_folder + sup_folder + 'distance_matrix_'+dist_measure+'.npy')
        
        labels = [self.translate(p.name) for p in pattern_dict.values()]
        #labels = [self.translate(p.name) for p in pattern_dict]#this line is for when we plot reclustering (then, comment out line above)

        array_toplot = np.round(distance_matrix,3)
        
        plt.figure()
        fig, ax = plt.subplots()
        
        im = ax.imshow(array_toplot,cmap='Spectral')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(yticks = np.arange(len(labels)), 
               yticklabels=labels,
               xticks = np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.tick_params(axis='both', which='major', labelsize=6)
        
        plt.xlabel('Gadget', fontsize=7)
        plt.ylabel('Gadget', fontsize=7)
        
        fig.tight_layout()
        plt.savefig(self.plots_folder + dist_measure+'_distance_matrix.'+str(self.plot_format), dpi = 200)
        
    def utility_clusters(self, agent_id=None, sup_folder=None):
        """
        Analysis of the results output by the utility clustering.

        Parameters
        ----------
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.
            
        sup_folder: (str) supplementary folder where the cluster labels are saved.
        
        """
        
        #get parameters
        if sup_folder == None:
           sup_folder = 'gadget_clustering/utility/'
        
        #create a list with the identifiers of the analysed agents. 
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None
            
        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
        
        patterns = []
        
        #Load files together in one single list (same order as in sim_distance)
        pattern_dict = np.load(results_folder + 'pattern_dict_cycle_0_runs_'+str(agent_list)+'.npy', allow_pickle=True).item()      
        patterns += [p for p in pattern_dict.values()]
            
        cluster_labels = np.load(self.results_folder + sup_folder +'labels_utility.npy')
        prob_cluster_labels = np.load(self.results_folder + sup_folder + 'prob_labels_utility.npy')
        
        #Save classes into a csv file
        table_list = []
        for i, p in enumerate(patterns):
            #Pattern name
            translated_name = self.translate(p.name)
        
            table_list.append([translated_name, cluster_labels[i], prob_cluster_labels[i], p.I, p.F, p.C, p.agent[0]])
                
        #Create csv file with ranked patterns according to their class
        table_list.sort(reverse=True, key=lambda x:x[1])
        f = open(self.plots_folder +  "utility_clusters_runs_"+str(agent_list)+".csv","w+")
        f.write('Gadget;Cluster;Prob;Interestingness;Support;Cohesion\n')
        for pattern_list in table_list:
            f.write(self.to_latex(pattern_list[0])+';C'+ str(pattern_list[1])+';'+str(pattern_list[2])+';'+str(pattern_list[3])+';'+str(pattern_list[4])+';'+str(pattern_list[5])+'\n')
        f.close()
        
        
    def init_set_percentages(self, min_samples=None, dist_measure=None, agent_id=None, sup_folder=None):
        """
        Computes the percentage of sequences in the init set of patterns of a given cluster that start with a given operation.

        Parameters
        ----------
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.
            
        sup_folder: (str) supplementary folder where the cluster labels are saved.
        
        """
        #get parameters
        
        if dist_measure == None:
            dist_measure = config['classification']['SIM_MEASURE']
        if sup_folder == None:
            sup_folder = 'gadget_clustering/' + dist_measure + '/'
        
        #create a list with the identifiers of the analysed agents. 
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None
            
        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
        
        if not self.with_env:
            print('Add environment')
            
        possible_labels = []
        for label in env.action_labels:
            if list(label)[:-1] not in possible_labels:
                possible_labels.append(list(label)[:-1])
        
        
        for ag in agent_list:
            
            #Load files.
            pattern_dict = np.load(self.results_folder + 'pattern_dict_cycle_0_runs_'+str(agent_list)+'.npy', allow_pickle=True).item()
            
            cluster_labels = np.load(self.results_folder + sup_folder +'labels_'+dist_measure+'.npy')
            
            #Save classes into a csv file
            table_list = []
            for i, p in enumerate(pattern_dict.values()):
                
                #Pattern name
                translated_name = self.translate(p.name)
                
                #Analysis of its init set
                init_set = np.ndarray.tolist(np.load(self.results_folder + sup_folder +'init_set_pattern_'+str(i)+'.npy', allow_pickle=True))
                
                values_for_each_label = [0] * len(possible_labels)
                
                for circuit in init_set:
                    if len(circuit) > 3:
                        label_index = [index for index, label in enumerate(possible_labels) if circuit[0] == label][0]
                        
                        values_for_each_label[label_index] += 1
                        
                table_list.append([translated_name, cluster_labels[i]]+[value / len(init_set) for value in values_for_each_label])
                    
            operation_names = [self.translate(self.list_to_name(label)) for label in possible_labels]
            string_names = ''
            for name in operation_names:
                string_names += name[0] + ';'
                
            #Create csv file with ranked patterns according to their class
            
            table_list.sort(reverse=True, key=lambda x:x[1])
            f = open(self.plots_folder +  "gadget_clusters_first_operation_runs_"+str(agent_list)+".csv","w+")
            f.write('Gadget;Cluster;'+string_names+' \n')
            for pattern_list in table_list:
                string_values = ''
                for value in pattern_list[2:]:
                    string_values += str(value) + ';'
                f.write(self.to_latex(pattern_list[0])+';'+ 'C'+str(pattern_list[1])+';'+string_values+ '\n')
            f.close()
         
    def list_to_name(self, operation):
        
        name = ''
        for index in operation:
            name += str(index) + ','
            
        return [name]
        
    def translate(self, pattern_name):
        """
        Translation of the featured notation used in the algorithm to a readable notation.
        (If there is no feature_dict provided by the user, it leaves the pattern name unchanged.)

        Parameters
        ----------
        pattern_name : list
            Translated pattern name.

        Returns
        -------
        List with the translated pattern name.
            
        """
        
        if self.feature_dict:
            translated_pattern = []
            for pattern_element in pattern_name:
                translated_element = '$'
                features = pattern_element.split(',')
                for index, feature in enumerate(features):
                    if index in self.feature_dict and feature != '0':
                        if index == 2 and features[1] == '2': #remove this line if env is not qu computing
                            translated_element += self.feature_dict[index][feature] #remove this line if env is not qu computing
                        elif index == 2 and features[1] == '1':#remove this line if env is not qu computing
                            translated_element += feature #remove this line if env is not qu computing
                        else:#remove this line if env is not qu computing
                            translated_element += self.feature_dict[index][feature] 
                    if index == 1:
                        translated_element += '_{'
                    if index == 2:
                        translated_element += '}$'
                    
                translated_pattern.append(translated_element)
            
            return translated_pattern
        
        else:
            return pattern_name
    
    def to_latex(self, translated_pattern):
        """
        It removes the brackets from the given list and creates a str.
        
        Parameters
        ----------
        translated_pattern : list with the pattern elements as strings
            
        Returns
        -------
        string : str
            Pattern as a string to be written in the csv file with a latex format.

        """
        string = ''
        for i, pattern in enumerate(translated_pattern):
            string += pattern
            
            if i != (len(translated_pattern) - 1):
                string += ','
        
        return string
        

    def dataset_analysis(self, agent_id=None):
        # create a list with the identifiers of the analysed agents.
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None

        if agent_id:
            agent_list = [agent_id] if type(agent_id) == int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]


        print(agent_list)
        for ag in agent_list:
            print(ag)
            pattern_dict = np.load(self.results_folder + 'pattern_dict_run_' + str(ag) + '.npy',allow_pickle=True).item()
            for pattern in pattern_dict.keys():
                if pattern == str(['2,1,1,0', '2,1,1,0']):
                    cycle_list = pattern_dict[pattern].cycle
                    print(cycle_list)



           
        
    
if __name__ == "__main__":
    
    print(sys.path)
    MAIN_PATH = 'results/'
    EXPERIMENT = 'qi'
    NUM_CONFIG = 0
    NUM_AGENTS = 3
    AG_LIST = [0,1,2]
    config = get_config('exp_'+str(NUM_CONFIG) +'.cfg', EXPERIMENT)

    results_folder = MAIN_PATH + EXPERIMENT + '/exp_'+str(NUM_CONFIG) + '/'
    
    FEATURE_DICT = {0:{'1':'G', '2':'M'}, 1:{'1':'1', '2':'2'}, 2:{'1':'12', '2':'13', '3':'14', '4':'23', '5':'24', '6':'34'}}
    
    #Initialize class
    ra = ResultsAnalysis(results_folder, config, feature_dict=FEATURE_DICT, num_agents=NUM_AGENTS, with_env=True, plot_format='pdf')
    
    ra.learning_performance(agent_id=AG_LIST)

    