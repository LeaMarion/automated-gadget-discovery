import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
from tabulate import tabulate

from pattern_evaluation import PatternEvaluation
from seq_processor import SeqProcessor

from operator import itemgetter
import random
import sys, os

from utils import get_config
from PIL import Image, ImageDraw, ImageFont

#sys.path.append(os.path.join(os.path.dirname(__file__), '/home/leamt/gym-optical-tables/gym_optical_tables/envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/Users/leamarion/Git/gym-optical-tables/gym_optical_tables/envs'))
from optical_tables_env import OpticalTablesEnv



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
    
        
        self.results_folder = results_folder
        self.plots_folder = self.results_folder + 'analysis-and-plots/'
        pathlib.Path(self.plots_folder).mkdir(parents=True, exist_ok=True) 
        
        #get parameters from config 
        self.num_cycles = config['general']['CYCLES']
        self.episodes = config['general']['EPISODES'][0]
        self.data_collection_size = config['exploration']['agent_0']['DATASET_SIZE']
        self.int_rescale = config['jiminy']['INT_RESCALE']
        
        self.seqmin_hyperpar = config['mining']['HYPERPAR']
        # self.p_threshold = config['exchanger']['PROB_THRESHOLD']
        
        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_agents)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.tab20c)
        if not self.color_list:
            self.color_list = [cmap.to_rgba(i+1) for i in range(self.num_agents)]
        self.latex = True

    def gadgets(self, agent_id = None, with_plot = True, with_table = True, with_info=True, draw = False, pattern_eval = None, toolbox = None):
        """
        Analysis of the gadgets obtained by the agents.

        Parameters
        ----------
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.
    
        with_plot : bool, optional
            Whether to plot the gadgets in a bar plot or not. The default is True.

        """
        list_of_circuits = []
        #create a list with the identifiers of the analysed agents. 
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None
            
        if agent_id:
            agent_list = [agent_id] if type(agent_id)==int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]
        table_list = []
        info_list = []
        for ag in agent_list:
            print('\nAgent #', ag, '\n')
            #Load files.

            for cycle in range(self.num_cycles):
                pattern_dict = np.load(self.results_folder +'/sequence_mining/'+'final_rules_cycle_'+str(cycle)+'_run_'+str(ag)+'.npy', allow_pickle=True).item()

                circuit_counter = 0
                for p in pattern_dict:
                    translated_name = self.translate(pattern_dict[p][0])
                    circuit = self.generate_int_pattern(pattern_dict[p][0])

                    if draw:
                        info_list = self.render_circuit(circuit=circuit, num_circuit=circuit_counter, num_qudits=4, num_ops=len(circuit), num_cycle=cycle, pattern_eval = pattern_eval, toolbox=toolbox, info_list=info_list)
                    else:
                        list_of_circuits.append([circuit, ag])


                    circuit_counter += 1
                    entry = [translated_name, pattern_dict[p][3], pattern_dict[p][1], pattern_dict[p][2], [ag]]
                    if with_info:
                        reward, info = pattern_eval.check_reward(circuit)
                        info = [list(np.sort(elem)) for elem in info]
                        entry.append(info)
                    table_list.append(entry)

        if with_table:
            print(info_list)
            headers = ['Pattern', 'I', 'Support', 'Cohesion', 'Agent']
            if with_info:
                headers.append('Info')

            print(tabulate(table_list, headers=headers))
            
            #Create csv file with ranked patterns
            table_list.sort(reverse=True, key=lambda x: (x[5][0],x[1]))
            f = open(self.plots_folder + "gadgets.csv","w+")
            print(self.plots_folder)
            if with_info:
                f.write('Pattern;Interestingness;Support;Cohesion;Info \n')

            else:
                f.write('Pattern;Interestingness;Support;Cohesion;Agent \n')
            for pattern_list in table_list:

                if with_info:
                    if self.latex:
                        pattern_str, info_str = self.prep_csv_for_latex(pattern_list[0], pattern_list[5])
                    else:
                        pattern_str = str(pattern_list[0])
                        info_str = str(pattern_list[5])

                    f.write(pattern_str + ';' + str(pattern_list[1]) + ';' + str(pattern_list[2]) + ';' + str(
                        pattern_list[3]) + ';' + info_str + '\n')
                else:
                    f.write(str(pattern_list[0])+';'+ str(pattern_list[1])+';'+ str(pattern_list[2])+';'+ str(pattern_list[3])+';'+ str(pattern_list[4])+'\n')
            f.close()
            
        if with_plot:
            print('with plot', with_plot)

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
            plt.savefig(self.plots_folder + 'gadgets.'+str(self.plot_format), dpi = 200)

            return list_of_circuits
    
    def gadget_classes(self, agent_id=None):
        """
        Analysis of the results output by the exchanger, which classifies the gadgets into classes.

        Parameters
        ----------
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.
    
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
            exchang_dict = np.load(self.results_folder + 'exchangeability_matrix_run_'+str(ag)+'.npy', allow_pickle=True).item()
            
            probabilities = np.zeros([len(exchang_dict), len(exchang_dict)])
            for orig_id in exchang_dict.keys():
                probabilities[orig_id,orig_id] = 1
                for rep_id in exchang_dict[orig_id].keys():
                    probabilities[orig_id, rep_id] = exchang_dict[orig_id][rep_id]
                    
            #Plot
            labels = [self.translate(p.name) for p in pattern_dict.values()]

            array_toplot = np.round(probabilities,2)
            
            plt.figure()
            fig, ax = plt.subplots()
            
            im = ax.imshow(array_toplot,vmin=0.0, vmax=1,cmap='RdYlGn')
            ax.figure.colorbar(im, ax=ax)
            
            ax.set(yticks = np.arange(len(labels)), 
                   yticklabels=labels,
                   xticks = np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            plt.xlabel('Replacement gadget', fontsize=7)
            plt.ylabel('Gadget', fontsize=7)
            
            fig.tight_layout()
            plt.savefig(self.plots_folder + 'exchangeability_run_'+str(ag)+'.'+str(self.plot_format), dpi = 200)
            
            
            #Save classes into a csv file
            
            #Load classes
            classes = np.load(self.results_folder + 'exchanger/' + 'equivalence_classes_p_thres_'+str(self.p_threshold)+'_run_'+str(ag)+'.npy', allow_pickle=True)
            hierarchy_dict = np.load(self.results_folder + 'exchanger/' + 'class_hierarchy_p_thres_'+str(self.p_threshold)+'_run_'+str(ag)+'.npy', allow_pickle=True).item()
            
            table_list = []
            for p in pattern_dict.values():
                translated_name = self.translate(p.name)
                for ith_class, class_set in enumerate(classes):
                    if p.id in class_set:
                        table_list.append([translated_name, ith_class])
                        
            #Create csv file with ranked patterns according to their class
            table_list.sort(reverse=True, key=lambda x:x[1])
            f = open(self.plots_folder + "gadget_classes_p_thres_"+str(self.p_threshold)+"_run_"+str(ag)+".csv","w+")
            f.write('Pattern;Class \n')
            for pattern_list in table_list:
                f.write(str(pattern_list[0])+';'+ str(pattern_list[1])+ '\n')
            f.close()
            
            #Print hierarchy
            print('Hierarchy of classes: \n', hierarchy_dict, '\n')
            
    def gadget_clustering(self, min_samples=None, dist_measure=None, agent_id=None, sup_folder=None, ):
        """
        Analysis of the results output by the gadget clustering.

        Parameters
        ----------
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
        
        for ag in agent_list:
            
            #Load files.
            pattern_dict = np.load(self.results_folder + 'pattern_dict_run_'+str(ag)+'.npy', allow_pickle=True).item()
            cluster_labels = np.load(self.results_folder + sup_folder +'labels_run_'+str(ag)+'.npy')
            prob_cluster_labels = np.load(self.results_folder + sup_folder + 'prob_labels_run_'+str(ag)+'.npy')
            
            #Save classes into a csv file
            table_list = []
            for i, p in enumerate(pattern_dict.values()):
                translated_name = self.translate(p.name)
                table_list.append([translated_name, cluster_labels[i], prob_cluster_labels[i]])
                        
            #Create csv file with ranked patterns according to their class
            table_list.sort(reverse=True, key=lambda x:x[1])
            f = open(self.plots_folder +  "gadget_clusters_"+str(dist_measure)+"_distance_run_"+str(ag)+".csv","w+")
            f.write('Pattern;Cluster;Prob \n')
            for pattern_list in table_list:
                f.write(str(pattern_list[0])+';'+ str(pattern_list[1])+';'+str(pattern_list[2])+ '\n')
            f.close()
            
            
            #Plot
            distance_matrix = np.load(self.results_folder + sup_folder + 'distance_matrix_'+dist_measure+'_run_'+str(ag)+'.npy')
            
            labels = [self.translate(p.name) for p in pattern_dict.values()]

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
            plt.savefig(self.plots_folder + dist_measure+'_distance_matrix_run_'+str(ag)+'.'+str(self.plot_format), dpi = 200)

    def render_circuit(self, circuit=[], num_circuit=0, num_qudits=0, num_ops=6, num_cycle=0, pattern_eval = None, toolbox = None, info_list = []):
        """
        Draws a circuit given an input list of optical elements.

        Args:
            circuit (list) : list of optical elements
            num_circuit (int) : counter for saving the circuit image
            num_qudits (int): integer to determine image height
            num_qudits (int): integer to determine image width
        """

        reward, info = pattern_eval.check_reward(circuit)
        print('state',info[0])
        if list(sorted(info[0])) not in info_list:
            info_list.append(sorted(list(info[0])))

        width = num_ops * 50 + 100
        height = num_qudits * 50 + 100
        im = Image.new('RGB', (width, height), (194, 194, 198))
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), str(info), (0, 0, 0))
        draw.polygon(((0, 0), (width, height), (width, 0)), fill=(194, 194, 198))
        draw.polygon(((0, 0), (width, height), (0, height)), fill=(194, 194, 198))

        if reward == 1.0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        x_pos = 60 + 50 * num_ops
        y_pos = 10
        draw.rectangle((x_pos, y_pos, x_pos + 30, y_pos + 30), fill=color)
        y_start = 70
        for i in range(num_qudits):
            draw.line(((0, y_start + 10 + 50 * i), (width, y_start + 10 + 50 * i)), fill=(255, 255, 255), width=5)

        elements = [toolbox[element] for j, element in enumerate(circuit)]

        for j, ele in enumerate(elements):
            x_pos = 50 + 50 * j
            if ele[0] == 'H':
                color = (255, 122, 122)
                pos_1 = (int(ele[5]) - 1) * 50
                draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

            elif ele[0] == 'B':
                pos_1 = (int(ele[3]) - 1) * 50
                pos_2 = (int(ele[6]) - 1) * 50
                color = (95, 173, 233)
                draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)
                draw.ellipse((x_pos, y_start + pos_2, 20 + x_pos, y_start + 20 + pos_2), fill=color)
                draw.line(((x_pos + 10, y_start + 10 + pos_1), (x_pos + 10, y_start + 10 + pos_2)), fill=color, width=5)

            elif ele[0] == 'D':
                color = (242, 206, 121)
                pos_1 = (int(ele[3]) - 1) * 50
                draw.rectangle((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

            elif ele[0] == 'R':
                color = (133, 246, 159)
                pos_1 = (int(ele[5]) - 1) * 50
                draw.rectangle((x_pos, y_start + pos_1 - 20, 20 + x_pos, y_start + 40 + pos_1), fill=color)
        circuit_folder = self.plots_folder + '/circuits/'
        pathlib.Path(circuit_folder).mkdir(parents=True, exist_ok=True)
        im.save(circuit_folder + "cycle_" + str(num_cycle) + "_" + str(info) + "_circuit_num_" + str(num_circuit) + ".jpg")
        return info_list


    def render_multiple_circuits(self, circuits=[], num_circuit=0, num_qudits=0, num_ops=6, num_cycle=0, pattern_eval = None, toolbox = None, info_list = []):
        """
        Draws a circuit given an input list of optical elements.

        Args:
            circuit (list) : list of optical elements
            num_circuit (int) : counter for saving the circuit image
            num_qudits (int): integer to determine image height
            num_qudits (int): integer to determine image width
        """

        x_counter = 0
        y_count_list = []
        y_counter = 0
        max_x = 0
        max_y = 0

        circuit_list = []
        for circuit in circuits:
            reward, info = pattern_eval.check_reward(circuit[0])
            if info == []:
                pass
            elif list(info[0]) not in info_list:
                info_list.append(list(info[0]))
                y_count_list.append(1)
                x_counter = len(info_list)-1
                y_counter = 0
                max_x = 1
                max_y = 1
            else:
                x_counter = info_list.index(list(info[0]))
                y_count_list[x_counter] += 1
                y_counter = y_count_list[x_counter]-1

            if x_counter > max_x:
                max_x = x_counter
            if y_counter > max_y:
                max_y = y_counter
            circuit_list.append([circuit[0],list(info[0]),reward,x_counter,y_counter, circuit[1]])



        print(max_x, max_y, circuit_list)
        width = ((num_ops * 50) + 100)*max_x
        height = ((num_qudits * 50)+ 100)*(max_y+1)
        #print(width, height)
        im = Image.new('RGB', (width, height), (194, 194, 198))
        draw = ImageDraw.Draw(im)
        draw.polygon(((0, 0), (width, height), (width, 0)), fill=(255, 255, 255))
        draw.polygon(((0, 0), (width, height), (0, height)), fill=(255, 255, 255))
        font = ImageFont.truetype("Roboto-Light.ttf", 50)

        for enum, info in enumerate(info_list):
            draw.text(((enum*((num_ops * 50)+ 100)), 0), str(info), (0, 0, 0), font=font)

        #print(sorted(list(info[0])), x_counter, y_counter, y_count_list)
        #
        for circuit in circuit_list:
            if circuit[2] == 1.0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            x_pos = 10 + 50 * num_ops+circuit[3]*((num_ops * 50)+ 100)
            y_pos = 20*(num_circuit+1)+circuit[4]*250+100
            draw.rectangle((x_pos, y_pos, x_pos + 30, y_pos + 30), fill=color)
            y_start = 70*(num_circuit+1)+circuit[4]*250+100
            draw.text(((circuit[3] * ((num_ops * 50) + 100)), y_start-50), str(circuit[5]), (0, 0, 0), font=font)
            for i in range(num_qudits):
                draw.line((((num_ops * 50)+ 100)*(circuit[3]), y_start + 10 + 50 * i, (100*circuit[3])+((num_ops * 50))*(1+circuit[3]), y_start + 10 + 50 * i), fill=(0, 0, 0), width=5)

            elements = [toolbox[element] for j, element in enumerate(circuit[0])]
            print(elements)

            for j, ele in enumerate(elements):
                x_pos = 50 + 50 * j +circuit[3]*((num_ops * 50)+100)
                if ele[0] == 'H':
                    color = (255, 122, 122)
                    pos_1 = (int(ele[5]) - 1) * 50
                    draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

                elif ele[0] == 'B':
                    pos_1 = (int(ele[3]) - 1) * 50
                    pos_2 = (int(ele[6]) - 1) * 50
                    color = (95, 173, 233)
                    draw.ellipse((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)
                    draw.ellipse((x_pos, y_start + pos_2, 20 + x_pos, y_start + 20 + pos_2), fill=color)
                    draw.line(((x_pos + 10, y_start + 10 + pos_1), (x_pos + 10, y_start + 10 + pos_2)), fill=color, width=5)

                elif ele[0] == 'D':
                    color = (242, 206, 121)
                    pos_1 = (int(ele[3]) - 1) * 50
                    draw.rectangle((x_pos, y_start + pos_1, 20 + x_pos, y_start + 20 + pos_1), fill=color)

                elif ele[0] == 'R':
                    color = (133, 246, 159)
                    pos_1 = (int(ele[5]) - 1) * 50
                    draw.rectangle((x_pos, y_start + pos_1 - 20, 20 + x_pos, y_start + 40 + pos_1), fill=color)

        circuit_folder = self.plots_folder + '/circuits/'
        pathlib.Path(circuit_folder).mkdir(parents=True, exist_ok=True)
        im.save(circuit_folder + "ALL_circuits.jpg")
        return info_list

            
        
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
                translated_element = ''
                features = pattern_element.split(',')
                for index, feature in enumerate(features):
                    if index in self.feature_dict and feature != '0':
                            translated_element += self.feature_dict[index][int(feature)-1] + ','
                    elif feature != '0':
                        translated_element += feature + ','

                translated_pattern.append(translated_element)

            return translated_pattern

        else:
            return pattern_name


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


    def list_translate(self, p):
        name = ''
        for ele in p:
            ele = int(ele)-1
            name += self.feature_dict[0][ele]+'_'
        return name

    def generate_int_pattern(self, pattern):
        if type(pattern[0]) == str:
            pattern = [int(elem)-1 for elem in pattern]
        return pattern


    def combine_data(self, agent_list, cycle, path):
        all_agents_data = np.load(
            path + '/run_' + str(agent_list[0]) + '/pos_reward_data_cycle_' + str(cycle) + '_run_' + str(agent_list[0]) + '.npy',
            allow_pickle=True)
        for ag in agent_list[1:]:
            print('\nAgent #', ag, '\n')

            pos_data = np.load(
                 path +'/run_'+str(ag)+'/pos_reward_data_cycle_' + str(cycle) + '_run_' + str(ag) + '.npy',
                allow_pickle=True)
            all_agents_data = np.concatenate((all_agents_data, pos_data))
            print(len(all_agents_data), all_agents_data.shape)
            #print(pos_data[0:2])

        #all_agents_data = np.unique(all_agents_data,axis=1)
        #print('num elements',all_agents_data[:8],all_agents_data.shape)
        save_path = path+'/pos_reward_data_cycle' + str(cycle) + '_run_' + str(0) + '.npy'
        np.save(save_path, all_agents_data)

    def utility_clusters(self, agent_id=None, sup_folder=None, with_info=True):
        """
        Analysis of the results output by the utility clustering.

        Parameters
        ----------
        agent_id : int or list, optional
            Agents to be analysed. The default is None, which automatically takes into account all agents.

        sup_folder: (str) supplementary folder where the cluster labels are saved.

        """

        # get parameters
        if sup_folder == None:
            sup_folder = 'utility_clustering/'

        # create a list with the identifiers of the analysed agents.
        if type(agent_id) != int and type(agent_id) != list:
            agent_id = None

        if agent_id:
            agent_list = [agent_id] if type(agent_id) == int else agent_id
        else:
            agent_list = [i for i in range(self.num_agents)]

        patterns = []

        pattern_dict = np.load(results_folder + 'pattern_dict.npy', allow_pickle=True).item()
        patterns += [p for p in pattern_dict.values()]

        cluster_labels = np.load(self.results_folder + sup_folder + 'labels_utility_all.npy')
        prob_cluster_labels = np.load(self.results_folder + sup_folder + 'prob_labels_utility_all.npy')

        # Save classes into a csv file
        table_list = []
        for i, p in enumerate(patterns):
            # Pattern name
            print(p.name)
            translated_name = self.translate(p.name)

            if with_info:
                circuit = self.generate_int_pattern(p.name)
                reward, info = pattern_eval.check_reward(circuit)
                info = [list(np.sort(elem)) for elem in info]
                table_list.append([translated_name, p.I, p.F, p.C, cluster_labels[i], prob_cluster_labels[i], info])
            else:
                table_list.append([translated_name, p.I, p.F, p.C, p.cluster_labels[i], prob_cluster_labels[i]])


        # Create csv file with ranked patterns according to their class
        table_list.sort(reverse=True, key=lambda x: x[1])
        f = open(self.plots_folder + "utility_clusters_all.csv", "w+")
        if with_info:
            f.write('Pattern;Cluster;Prob;Info \n')
        else:
            f.write('Pattern;Cluster;Prob \n')
        for pattern_list in table_list:
            if with_info:
                f.write(str(pattern_list[0]) + ';' + str(pattern_list[4]) + ';' + str(pattern_list[5]) + ';' + str(pattern_list[6]) +'\n')
            else:
                f.write(str(pattern_list[0]) + ';' + str(pattern_list[4]) + ';' + str(pattern_list[5]) + '\n')
        f.close()

        # Create csv file with ranked patterns according to their class
        table_list.sort(reverse=True, key=lambda x: x[1])
        f = open(self.plots_folder + "utility_clusters_all_info.csv", "w+")
        if with_info:
            f.write('Pattern;Interestingness;Support;Cohesion;Cluster;Prob;Info \n')
        else:
            f.write('Pattern;Interestingness;Support;Cohesion;Cluster;Prob \n')
        for pattern_list in table_list:
            if with_info:
                if self.latex:
                    pattern_str, info_str = self.prep_csv_for_latex(pattern_list[0], pattern_list[6])
                else:
                    pattern_str = str(pattern_list[0])
                    info_str = str(pattern_list[5])
                f.write(pattern_str + ';' + str(pattern_list[1]) + ';' + str(pattern_list[2]) + ';' + str(pattern_list[3]) + ';' + str(pattern_list[4]) + ';' + str(pattern_list[5]) + ';' + info_str + '\n')
            else:
                f.write(str(pattern_list[0]) + ';' + str(pattern_list[1]) + ';' + str(pattern_list[2]) + ';' + str(pattern_list[3]) + ';' + str(pattern_list[4]) + ';' + str(pattern_list[5]) + '\n')
        f.close()

    def prep_csv_for_latex(self,pattern,info):
        #--latex encoding pattern
        pattern_str = ''
        for elem in pattern:
            if elem[0:2] == 'DP':
                index = 4
            elif elem[0:2] == 'Ho':
                index = 6
            else:
                index = len(elem)
            prev = elem[0:index]
            for num, letter in enumerate(list(map(chr, range(97, 101)))):
                new = elem[0:index].replace(str(num + 1), letter)
                elem = elem.replace(prev, new)
            elem = elem.replace(' ', "")
            elem = elem.replace('[', '$_{')
            elem = elem.replace(']', '}$')

            pattern_str += elem
        pattern_str = pattern_str[:-1]

        #--latex encoding SRV
        info = info[0]
        info.sort(reverse=True)
        info_str = '$('
        for elem in info:
            info_str += str(elem) +','
        info_str = info_str[:-1]
        info_str += ')$'
        return pattern_str, info_str


if __name__ == "__main__":
    MAIN_PATH = 'results/'
    EXPERIMENT = 'qo'
    NUM_CONFIG = 0

    AG_LIST = range(10)
    NUM_AGENTS = len(AG_LIST)


    config = get_config('exp_'+str(NUM_CONFIG) +'.cfg', EXPERIMENT)

    results_folder = MAIN_PATH + EXPERIMENT + '/exp_'+str(NUM_CONFIG)+'/'

    ENV_PARAMS = config['environment']['ENV_PARAMS']
    env = OpticalTablesEnv(**ENV_PARAMS)
    #set important env. attributes
    setattr(env, 'rand_start', False)
    setattr(env, 'max_steps', env.episode_length)
    if not hasattr(env, 'action_labels'):
        label_list = [np.array([x + 1]) for x in range(env.num_actions)]
        setattr(env, 'action_labels', label_list)
        action_list = list(range(env.num_actions))
        setattr(env, 'action_list', action_list)
        num_features = len(label_list[0])
        setattr(env, 'num_features', num_features)
    else:
        action_list = list(range(env.num_actions))
        setattr(env, 'action_list', action_list)
        num_features = len(env.action_labels[0])
        setattr(env, 'num_features', num_features)

    # initialize sequence processor
    ROOT_LENGTH = config['processor']['ROOT_LENGTH']
    FEATURES_REMOVED = config['processor']['FEATURES_REMOVED']
    NUM_FEATURES = len(env.action_labels[0])
    SEQ_LENGTH = env.max_steps

    max_elements = [0] * NUM_FEATURES
    for feature in range(NUM_FEATURES):
        max_elements[feature] = max([label[feature] for label in env.action_labels])

    if env.rand_start:
        INDICES_TO_MINE = [index for index in range(env.max_steps) if
                           index not in list(range(env.max_steps))[env.fixed_op - 1::env.fixed_op]]
    else:
        INDICES_TO_MINE = None

    seq_processor = SeqProcessor(root_length=ROOT_LENGTH, action_labels=env.action_labels, \
                                 features_removed=FEATURES_REMOVED, indices_to_mine=INDICES_TO_MINE,
                                 num_features=NUM_FEATURES, \
                                 sequence_length=SEQ_LENGTH, max_value_features=max_elements)

    pattern_eval = PatternEvaluation(seq_proc=seq_processor, env=env)
    toolbox = env.read_toolbox()
    dict_tools = {x:tool for x, tool in enumerate(toolbox)}
    print('TOOLS',dict_tools)

    
    FEATURE_DICT = {0:dict_tools}
    
    #Initialize class
    ra = ResultsAnalysis(results_folder, config, feature_dict=FEATURE_DICT, num_agents=NUM_AGENTS)

    circuit_counter = 0
    cycle = 0
    info_list = []


    list_of_circuits = ra.gadgets(agent_id=AG_LIST, pattern_eval = pattern_eval, toolbox=toolbox)
    ra.utility_clusters(agent_id=AG_LIST)
    print('CIRCUIT LIST',list_of_circuits)





    
    
    