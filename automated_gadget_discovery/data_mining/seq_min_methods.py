"""
The code below implements and extends the sequence mining algorithm in:
[1] Cheng Zhou, Boris Cule and Bart Goethals, Pattern Based Sequence Classification,
IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, May 2016, pp. 1285-1298, vol. 28
"""

########################################


import numpy as np
from collections import Counter
import math
import itertools
import hdbscan

from data_mining.pattern_evaluation import PatternEvaluation

class SeqMinMethods(PatternEvaluation):
    def __init__(self, **kwargs):
        """
        Class that contains all methods needed to implement the Sequence Mining algorithm.
        Args:
            
            **kwargs:
                
        """
        
        super().__init__(**kwargs) 
       
        
    def clustering(self, data, u, min_size=None, min_samples=20, metric='euclidean', encoding="vec_resource", verbose=False):
        """
        Clusters samples according to a given choice of metric, algorithm and utility
    
        Parameters
        ----------
        data : list
            List of lists of circuits.
            Each circuit is a list of elements, which is a list with 4 integers, as for the tree representation.
        u : list
            List of values corresponding to the availability of each type of operation (G1, G2, M1, M2).
            The higher this value, the more a type of operation is easy to perform or, in general, relevant.
        min_size : int, optional (default=len(`data`)/6).
            Minimum allowed number of points in each clusters.
            A value too high can make HDBSCAN label everything as noise, while a value too low can make clustering useless.
        min_samples : int, optional (default=20).
            See HDBSCAN for the documentation.
        
        Other Parameters
        ----------
        metric : str
            See HDBSCAN for the documentation.
        verbose : bool, optional (default=True).
            If True, it prints a short summary.
        
        
        
        See Also:
        ----------
        Sample_Tally(.)
    
        Returns
        ----------
        clusters : list
            List of lists of circuits, grouped by label according to the clustering algorithm.
        labels : list
            List of labels for all circuits.
        """
    
        data = np.array(data)
        
        min_cl_size = int(len(data) / 6) if min_size == None else min_size
    
        # m = '\nFor clustering, circuits are described by resource '
    
        if encoding == 'vec_resource':
            # print(m + 'vectors as (' + str(u[0]) + ' G1, ' + str(u[1]) + ' G2, ' + str(u[2]) + ' M1, ' + str(
                # u[3]) + ' M2).\n')
            data_toCluster = np.array([u * self.seq_proc.sample_tally(sample) for sample in data])
        elif encoding == 'sum_resource':
            # print(m + 'costs as (' + str(u[0]) + ' G1 + ' + str(u[1]) + ' G2 + ' + str(u[2]) + ' M1 + ' + str(
                # u[3]) + ' M2).\n')
            data_toCluster = np.array([[sum(u * self.seq_proc.sample_tally(sample))] for sample in data])
    
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cl_size, min_samples=min_samples, metric=metric, core_dist_n_jobs=-1)
    
        labels = clusterer.fit_predict(data_toCluster)
    
        clusters = [data[labels == cl] for cl in range(max(labels) + 1)]
        
        
        return clusters, labels
            
    #TODO: maybe we want to generalize normal_ordering. Right now, it depends on env.num_qubits, which is from the quantum-computing env.
    # def normal_ordering(self, circuit, formatting="digits"):
    #     """
    #     Relabels the qubits each element acts on, so that they appear in ascending order.
    
    #     Parameters
    #     ----------
    #     circuit   : list
    #         List of elements describing the quantum circuit.
    #     formatting : {'digits', 'letters'}, (default: 'digits', which assumes that `circuit` uses digits as tree labels, i.e. [_, _, _, _]).
    #         Decides how to handle the information in `circuit`.
    
    #     Notes
    #     ----------
    #         + The actual circuit is not modified, only the labeling.
    #         + This function does not use commutation rules.
    #         + This function substantially facilitates sequence-mining with large circuits.
    
    #     Returns
    #     ----------
    #     list
    #         The same circuit with relabeled qubits.
    
    #     """
    
    #     if formatting == "digits":
    #         pairs = self.pairs(formatting)
    #         combs_involved = [(pairs[int(el[self.seq_proc.root_length:-1]) - 1] if el[1] == '2' else [el[self.seq_proc.root_length:-1]]) for el in circuit]
    #         available_modes = iter(range(2, self.env.num_qudits + 1))
    
    #     elif formatting == "letters":
    #         qubits = ['a', 'b', 'c', 'd', 'e', 'f', '...'][:self.env.num_qudits]
    #         pairs = self.pairs(formatting)
    #         combs_involved = [([el[self.seq_proc.root_length], el[self.seq_proc.root_length + 1]] if len(el) == 5 else [el[self.seq_proc.root_length]]) for el in
    #                           circuit]
    #         available_modes = iter(qubits[1:])
    
    #     combs_involved_flat = [m for comb in combs_involved for m in comb]
    
    #     if formatting == "digits":
    #         dict_modes = {'1':1}
    #         modes_taken = ['1']
    #     elif formatting == "letters":
    #         dict_modes = {'a':'a'}
    #         modes_taken = ['a']
    #     ordered_combs = [[] for el in circuit]
    
    #     for c in combs_involved_flat:
    #         if c not in modes_taken:
    #             modes_taken.append(c)
    #             dict_modes[c] = next(available_modes)
    
    #     for i, c in enumerate(combs_involved):
    #         for m in c:
    #             ordered_combs[i].append(str(dict_modes[m]))
    #         ordered_combs[i].sort()
    
    #     for i, el in enumerate(circuit):
    #         if formatting == "digits":
    #             circuit[i] = el[0:self.seq_proc.root_length] + (
    #                 str(1 + int(pairs.index(ordered_combs[i]))) if el[1] == '2' else ordered_combs[i][0]) + el[-1]
    #         elif formatting == "letters":
    #             circuit[i] = el[0:self.seq_proc.root_length] + ordered_combs[i][0] + (ordered_combs[i][1] if len(el) == 5 else '')
    
    #     return circuit
    
    def preprocessing(self, clusters, max_cluster_size=40000, ordering=False):
        """
        Preprocesses the data for sequence mining, after clustering.
    
        Parameters
        ----------
        clusters   : list
            List of lists of circuits. Each cluster has a different label, starting from 0.
        max_cluster_size : int, optional (default=40000)
            Maximum allowed size for the clusters. Larger clusters will be capped to this value.
        ordering : bool, optional (default: True). #TODO: what happend here?
            Whether we apply Normal_Ordering(.) to all circuits.
        cluster_on: bool, False if the clustering is deactivated.
        
        Notes
        ----------
        Necessary for compatibility, because clustering and sequence mining were not conceived from the beginning.
    
        Returns
        ----------
        list
            Preprocessed clusters (renamed in classes, from now on), ready to be mined.

        """

        if len(clusters) > 0:  # If the clustering algorithm has found at least one cluster.
            # if ordering == True:
            #     clusters = [[self.normal_ordering(sample) for sample in cl] for cl in clusters]
    
            min_cl_Size = min([len(cl) for cl in clusters])
            cluster_size = min(min_cl_Size, int(max_cluster_size)) 
            if self.seq_proc.features_removed == 0:
                classes = [[[el for el in seq] for seq in cl[:cluster_size]] for cl in clusters]
            else:
                classes = [[[','.join([str(n) for n in el.split(',')[:len(el.split(','))-self.seq_proc.features_removed]]) for el in seq] for seq in cl[:cluster_size]] for cl in clusters]

            S = [seq for cl in classes for seq in cl]  # Same data, without the division in clusters. Useful, not necessary.
    
            return classes, S
    
        else:  # If all circuits were labeled as noise.
            print("No cluster was found. Try reducing the minimum cluster size.\n")
            return None, None

    def rescale_threshold(self, m, length_P):
        """
        Lowers the thresholds in SequenceMining(.), to allow longer patterns to be selected.
    
        Parameters
        ----------
        m   : float in (0,1]
            Rescaling factor, from length (l) to (l+1)
        length_P : int
            Length of the pattern.
    
        Notes
        ----------
        This modification is not used in the literature, and the expression is arbitrary: better expressions are likely to exist.
    
        See Also:
        ----------
        Enumerate_Sequence(.)
    
        Returns
        ----------
        float
            Rescaling factor
    
        """

        return m ** length_P
    
    def sequence_mining(self, S, classes, min_sup, min_coh, factor_min_int, min_conf, max_size=None, m=0.66, verbose=1):
        """
        Sequence mining algorithm. Relevant patterns identified by this function still need to be filtered.
        
        This function creates the dictionary:
        dict_P2Nk : dictionary
            Dictionary matching every subsequence `P` (key) with the list (value) of sequences that contain it with label `k`.
    
        Parameters
        ----------
        S : list
            List of all sequences, ignoring all labels.
        classes : list
            List of lists of circuits. Every class has label `k`.
        min_sup : float
            Minimum allowed support to consider a pattern as relevant. Value in [0,1].
        min_coh : float
            Minimum allowed cohesion to consider a pattern as relevant. Value in [0,1].
        factor_min_int : float. Default: 1 (if 1, the minimum allowed interestingness is just min_sup*min_coh)
            factor_min_int*min_sup*min_coh is the minimum allowed interestingness to consider a pattern as relevant. 
        min_conf : float
            Minimum allowed confidence to consider a pattern as relevant. Value in [0,1].
        m : float, optional (default=0.25).
            Factor used by Rescale_Threshold(.) to rescale the thresholds in Enumerate_Sequence(.). Value in [0,1].
    
        Other Parameters
        ----------
        max_size : int, optional (default=sequence_length, maximum)
            Maximum allowed length for relevant patterns. Limits the level of recursion.
        verbose : {0,1}, optional (default=1, which prints intermediate information on the sequence mining)
            Prints how many patterns are found for length=3 and length=4, as the algorithm continues to mine.
    
        Notes
        ----------
            + Pays more attention to patterns with length>2 than it is done in the literature.
            + Later, this function could integrate the While loop in the main source file.
    
        References
        ----------
        Algorithm described in "Algorithm 4: Generating interesting subsequences" and "Algorithm 5: Enumerate-Sequence(Q)" in
        [1] Cheng Zhou, Boris Cule and Bart Goethals, Pattern Based Sequence Classification,
        IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, vol. 28, pp. 1285-1298 (2016).
    
        See Also:
        ----------
        Enumerate_Sequence(.), Rescale_Threshold(.)
    
        Returns
        ----------
        list
            List of candidate relevant rules (rule = [pattern, label]). These rules still need to be filtered.
    
        """
        
        if max_size is None:
            max_size = self.seq_proc.sequence_length
            
        raw_rules = []
        self.dict_P2Nk = [[]] * len(classes)
    
        if verbose > 0:
            print("\n______________  Support:", round(min_sup, 4), "    Cohesion:", round(min_coh, 4), "    Interest:", round(factor_min_int*min_sup*min_coh, 4),
                  "    Confidence:", min_conf, "    Rescale:", round(m, 2), "\n")
    
        for k, S_k in enumerate(classes):
            print("\nCluster " + str(k)+" with size "+str(len(S_k)))
            self.cont_L2, self.cont_L3, self.cont_L4, self.cont_L5 = 0, 0, 0, 0

            self.enumerate_sequence(k, "Start", S, S_k, 0, min_sup, min_coh, factor_min_int, min_conf, max_size, m=m, verbose=verbose)
            print("L2:", self.cont_L2, "   L3:", self.cont_L3, "   L4:", self.cont_L4, "   L5:", self.cont_L5)
            raw_rules.extend([[P, k] for P in self.Y_k])

    
        return raw_rules


    def enumerate_sequence(self, k, Q, S, S_k, l, min_sup, min_coh, factor_min_int, min_conf, max_size, m, verbose=1):
        """
        Building block to create patterns of increasing length in each class `k`.
    
        Parameters
        ----------
        k   : int
            Class label.
        Q : list
            List of current relevant patterns, to be combined to create new candidate patterns.
        S : list
            List of all sequences, ignoring all labels.
        S_k   : list
            List of circuits in class `k`.
        l : int
            Level of the recursion. The length of the current patterns is equal to `l`+1.
        min_sup : float
            Minimum allowed support to consider a pattern as relevant. Value in [0,1].
        min_coh : float
            Minimum allowed cohesion to consider a pattern as relevant. Value in [0,1].
        factor_min_int : float. Default: 1 (if 1, the minimum allowed interestingness is just min_sup*min_coh)
            factor_min_int*min_sup*min_coh is the minimum allowed interestingness to consider a pattern as relevant. 
        max_size : int
            Maximum allowed length for relevant patterns. Limits the level of recursion.
        verbose : {0,1}, optional (default=1, which prints intermediate information on the sequence mining)
            Prints how many patterns are found for length=3 and length=4, as the algorithm continues to mine.
    
        Notes
        ----------
            + Recursive function
            + Applies preliminary filters for the first time, to create candidate patterns.
            + Allows for more filters and modifications to be integrated.
    
        References
        ----------
        Recursive function used in "Algorithm 4: Generating interesting subsequences" and "Algorithm 5: Enumerate-Sequence(Q)" in
        [1] Cheng Zhou, Boris Cule and Bart Goethals, Pattern Based Sequence Classification,
        IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, vol. 28, pp. 1285-1298 (2016).
    
        See Also:
        ----------
        SequenceMining(.)
    
        Returns
        ----------
        list
            List of candidate relevant patterns, still to be filtered.
    
        """
        rescale = self.rescale_threshold(m, l)

        if l == 0:
            self.dict_P2Nk[k] = {}
    
            frequencies = Counter([el for seq in S_k for el in seq])
            
            #preliminary filter, count can be higher than the real support because it counts how many times an element appears inside a sequence too (not only if it appears in the sequence).
            Q = list([[el] for (el, count) in frequencies.items() if count > len(S_k) * min_sup]) 
            
            self.Y_k = list(filter(lambda p: self.interestingness(p, S_k, self.N(p, S_k)) >= min_sup * min_coh, Q))  # stores the final sequences
            [self.dict_P2Nk[k].update({str(s): self.N(s, S_k)}) for s in self.Y_k]  # We start populating the dictionary
            
            Q = self.Y_k.copy() #needed, so that the dictionary contains the same items as Q (there can be elements of initial Q that do not pass the filter of Y_k).
           
        for alpha_I in Q:
            
            a = []  # The paper uses an extra variable t, which seems not necessary (Y_k already does the job)
    
            for alpha_J in Q:

                s = alpha_I.copy()
                s.append(alpha_J[l])  # Connects the two string elements, without joining them

                Nk_P = self.N(s, self.dict_P2Nk[k][str(alpha_I)])  # Nk_P = N(s, S_k) # We precompute it and save it in dict_P_to_Nk to make things faster
                support = self.support(S_k, Nk_P)
    
                if support >= min_sup * rescale:  # 1st filter
                    a.append(s)  # This continues to be relevant, even if it does not pass the next filter (see the last line)
                    self.dict_P2Nk[k][str(s)] = Nk_P.copy()  # We store Nk_P in a dictionary, not to compute it again
                    
                    if self.cohesion(s, Nk_P) >= min_coh * rescale:  # 2nd filter
                        if support * self.cohesion(s, Nk_P) >= factor_min_int * min_sup * min_coh * rescale:  # 3rd filter
                            if self.confidence(s, S, Nk_P) >= min_conf * rescale:  # 4th filter (we can add more)
        
                                self.Y_k.append(s)  # This is the collection of good rules (which will be filtered later)
        
                                if verbose >= 1:
                                    if l == 0:
                                        self.cont_L2 += 1
                                    elif l == 1:
                                        self.cont_L3 += 1
                                    elif l == 2:
                                        self.cont_L4 += 1
                                    elif l == 3:
                                        self.cont_L5 += 1
                                    if verbose == 1 and l >= 1 and (self.cont_L2 + self.cont_L3 + self.cont_L4 + self.cont_L5) % 100 == 0:
                                        print("L2:", self.cont_L2, "   L3:", self.cont_L3, "   L4:", self.cont_L4, "   L5:", self.cont_L5)
            if l < max_size - 2:
                self.enumerate_sequence(k, a, S, S_k, l + 1, min_sup, min_coh, factor_min_int, min_conf, max_size, m, verbose)


    def is_emergent(self, rule, classes, tol):
        """
        Checks whether a rule is emergent, according to the custom criteria below (see also [1]).
    
        Parameters
        ----------
    
        rule   : list
            Pair [`P`, label]
        classes : list
            List of all sequences, separated by label.
        tol   : list
            [emergence_tolerance (float in [0,1]), support_tolerance (float in [0,1])].
    
        Notes
        ----------
            + A pattern `P` can be matched with multiple labels, but every rule always pairs one pattern with one label.
            + A rule is emergent in one class if its interest and support in all other classes are lesser than two tolerance thresholds.
    
        References
        ----------
        [1]: ...
    
        Returns
        ----------
        bool
            Whether a rule is emergent.
    
        """
    
        emergence_tol, support_tol = tol[0], tol[1]
        P, label = rule[0], rule[1]
        N_rule = self.dict_P2Nk[label][str(P)]
    
        # I_rule and S_rule refer to the rule=[pattern, label] we are checking
        I_rule = self.interestingness(P, classes[label], N_rule)
        S_rule = self.support(classes[label], N_rule)
    
        flag = 0  # The rule will have to satisfy the criteria below len(classes)-1 times to be considered emergent
    
        for k, S_k in enumerate(classes):
    
            if k != label:  # We quantify how the rule performs in the other classes
                Nk_P = self.N(P, S_k)
                if self.interestingness(P, S_k, Nk_P) / I_rule <= emergence_tol and self.support(S_k, Nk_P) / S_rule <= support_tol:
                    flag += 1
    
        return True if flag == len(classes) - 1 else False

    def prune_bycoverage(self, S, labels, sorted_Rules, max_coverage=1):
        """
        Prunes pre-sorted rules by coverage.
    
        Parameters
        ----------
    
        S : list
            List of all sequences, ignoring all labels.
        labels : list
            List of labels for all circuits.
        sorted_Rules : list
            List of rules found by SequenceMining(.) and sorted according to custom criteria.
        max_coverage : int, optional (default=1)
            Maximum number of times a sequence can be covered by a rule. The lower the value, the more it cuts.
    
        Notes
        ----------
            + This filter is relevant if Sort_Rules(.) gives priority to the length, otherwise it does not cut much.
    
        References
        ----------
        Algorithm described in "Algorithm 6: Finding the most important rules" in
        [1] Cheng Zhou, Boris Cule and Bart Goethals, Pattern Based Sequence Classification,
        IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, vol. 28, pp. 1285-1298 (2016).
    
        See Also:
        ----------
        Sort_Rules(.), Filter_Rules(.), filter_byCOVERAGE(.)
    
        Returns
        ----------
        list
            List of relevant rules (rule = [pattern, label]) filtered by coverage.
        """

        D_counts = np.zeros(shape=len(S))
        D = S.copy()
        next_labels = labels.copy()
    
        pruned_Rules = [[]] * len(sorted_Rules)
    
        for r, rule in enumerate(sorted_Rules):
            D = [data for data in D if data[0] != 'Covered']  # Eliminates data predicted by the previous rule
            next_labels = [L for L in next_labels if L != 'Covered']  # Same as above, for labels
            D_counts = D_counts[D_counts >= 0]  # Same as above, for counts (see max_coverage)
    
            if next_labels != [] and len(rule[0])>1: #only the rules with more than one element are considered for coverage.
    
                for s, seq in enumerate(D):
                    if self.is_subsequence(rule[0], seq) and rule[1] == next_labels[s]:# Does it "match" and "predict" correctly?
                        pruned_Rules[r] = rule  # We save it
                        D_counts[s] += 1  # This data was predicted by one more rule (see max_coverage)
                        if D_counts[s] >= max_coverage:
                            D[s] = ['Covered']
                            D_counts[s] = -1  # Arbitrary negative value, it's just a flag
                            next_labels[s] = 'Covered'
        return [r for r in pruned_Rules if (r != [])]  # Returns only the rules that were saved


    def sort_rules(self, rules, S, classes, resource=[1, 1, 1, 1]):
        """
        Sorts all rules according to pre-defined criteria.
    
        Parameters
        ----------
        rules : list
            List of all rules (rule = [pattern, label]).
        S : list
            List of all sequences, ignoring all labels.
        classes : list
            List of lists of circuits. Every class has label `k`.

        
        Notes
        ----------
            + IMPORTANT: The order of the 3+ criteria is relevant (due to Prune_my_Rules() ) and should be chosen carefully.
    
        References
        ----------
        Algorithm described in "Algorithm 6: Finding the most important rules" in
        [1] Cheng Zhou, Boris Cule and Bart Goethals, Pattern Based Sequence Classification,
        IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, vol. 28, pp. 1285-1298 (2016).
    
        See Also:
        ----------
        Filter_Rules(.), Prune_byCoverage(.), filter_byCOVERAGE(.)
    
        Returns
        ----------
        list
            List of rules (rule = [pattern, label]) sorted according to pre-defined criteria.
        """
        scalar_product = lambda b: sum([resource[i] * b[i] for i in range(len(b))])

        #TODO: removed - infront of last sorting criterion such that it does choose longest not the shortest patterns
        return sorted(rules, key=lambda x: (
# =======
            self.confidence(x[0], S, self.dict_P2Nk[x[1]][str(x[0])]),  # 1 Confidence (this is always 1 if there are no classes)
            self.interestingness(x[0], classes[x[1]], self.dict_P2Nk[x[1]][str(x[0])]),  # 2 Interestingness
            len(x[0]),  # 3: Length of the gadget
            scalar_product(self.seq_proc.sample_tally(x[0])) / len(x[0]) #4 : Utilities (if utilities are all 1, this does not play a role)
            ), reverse=True)



    def filter_bylength(self, rules, min_L, criterion='all'):
        """
        Filters rules according to their length.
    
        Parameters
        ----------
        rules : list
            List of all rules (rule = [pattern, label]).
        min_L : length
            Minimum length allowed by this filter. Shorter patterns will be discarded.
        criterion : {'at_least_L', 'exactly_L', 'discard_long', both_bounds', 'all'}
            Whether to select rules long at least, or exactly, `min_L`.
            'discard_long': discards patterns longer than max_op/2, since they have a bias towards high cohesion (specially if max_op is low).
            'both_bounds': discards all patterns that have a length below min_L and above max_op/2.
            If 'all' is selected, analyses are still performed but no rule is discarded.
    
        Notes
        ----------
            + This is not standard in the literature.
    
        See Also:
        ----------
        Filter_Rules(.)
    
        Returns
        ----------
        list
            List of rules (rule = [pattern, label]) filtered by length.
        """

        initialLength = len(rules)
    
        if criterion == "at_least_L":
            filtered_Rules = list(filter(lambda x: len(x[0]) >= min_L, rules))
        elif criterion == "exactly_L":
            filtered_Rules = list(filter(lambda x: len(x[0]) == min_L, rules))
        elif criterion == "discard_long":
            filtered_Rules = list(filter(lambda x: len(x[0]) <= int(self.seq_proc.sequence_length/2), rules))
        elif criterion == "both_bounds":
            filtered_Rules = list(filter(lambda x: len(x[0]) <= int(self.seq_proc.sequence_length/2) and len(x[0]) >= min_L, rules))
        else:
            filtered_Rules = rules
    
        finalLength = len(filtered_Rules)
    
        if finalLength == 0:
            print("No rules were found with minimum length " + str(min_L) + ".")
            quit()  # Something more interesting can be done here, instead of quit().
    
        else:
    
            for l in [1, 2, 3, 4, 5]:
                print("Length " + str(l) + ":   " + str(len(list(filter(lambda x: len(x[0]) == l, filtered_Rules)))))
            print("\n", str(finalLength) + " rules have been selected by LENGTH (" + criterion + "), out of " + str(
                initialLength) + "\n")
            min_rule_Cluster = min(list(Counter([r[1] for r in filtered_Rules]).values()))
    
            return filtered_Rules, finalLength, min_rule_Cluster


    def filter_bycoverage(self, rules, S, classes, resource=[1, 1, 1, 1], max_coverage=2):
        """
        Filters rules according to the coverage mechanism.
    
        Parameters
        ----------
        rules : list
            List of all rules (rule = [pattern, label]).
        S : list
            List of all sequences, ignoring all labels.
        classes : list
            List of lists of circuits. Every class has label `k`.
        max_coverage : int, optional (default=1)
            Maximum number of times a sequence can be covered by a rule. The lower the value, the more it cuts.
    
        Notes
        ----------
            + IMPORTANT: The order of the criteria in Sort_Rules(.) is relevant and should be chosen carefully.
    
        References
        ----------
        Algorithm described in "Algorithm 6: Finding the most important rules" in
        [1] Cheng Zhou, Boris Cule and Bart Goethals, Pattern Based Sequence Classification,
        IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, vol. 28, pp. 1285-1298 (2016).
    
        See Also:
        ----------
        Sort_Rules(.), Prune_byCoverage(.), Filter_Rules(.)
    
        Returns
        ----------
        list
            List of rules (rule = [pattern, label]) filtered by coverage.
        """
    
        initialLength = len(rules)
        sorted_rules = self.sort_rules(rules, S, classes, resource)
        print("Rules have been sorted\n")
        print(sorted_rules)
    
        labels = [k for k in range(len(classes)) for i in range(len(classes[k]))]
        filtered_Rules = self.prune_bycoverage(S, labels, sorted_rules, max_coverage)
        print(filtered_Rules)
        finalLength = len(filtered_Rules)
        print(str(finalLength) + " rules have been selected by COVERAGE, out of " + str(initialLength) + ".\n")
        
        if finalLength != 0:
            min_rule_Cluster = min(list(Counter([r[1] for r in filtered_Rules]).values()))
        else:
            min_rule_Cluster = 0
    
        for l in [1, 2, 3, 4]:
            print("Length " + str(l) + ":   " + str(len(list(filter(lambda x: len(x[0]) == l, filtered_Rules)))))
    
        return filtered_Rules, finalLength, min_rule_Cluster


    def filter_byemergence(self, rules, classes, filter_ratio, min_R_cl, tol=[0.05, 0.05]):
        """
        Filters rules according to the emergence mechanism.
    
        Parameters
        ----------
        rules : list
            List of all rules (rule = [pattern, label]).
        classes : list
            List of lists of circuits. Every class has label `k`.
        filter_ratio : list
            List ([min_ratio, max_ratio]) with the minimum and maximum fraction of rules that we allow to cut.
            This filter tries to adapt its tolerance to stay in this region.
        min_R_cl : int
            Minimum number of rules that we want to have in every class after the filter is applied.
            If the filter would not satisfy this criterion, the closest best possible cut is returned.
        tol : list, optional (default=[0.1,0.1]).
            Controls the strength of the filter. The lower the value, the more the filter will cut.
    
        Notes
        ----------
            + The existence of two hyper-parameters for the tolerance is not necessary.
    
        References
        ----------
        Inspired by...
    
        See Also:
        ----------
        is_Emergent(.), Filter_Rules(.)
    
        Returns
        ----------
        list
            List of rules (rule = [pattern, label]) filtered by emergence.
        """

        length_0 = len(rules)
        rules_next = rules
        em_tol, sup_tol = tol[0], tol[1]
    
        max_toPass = int(length_0 * filter_ratio[1])
        min_toPass = int(length_0 * filter_ratio[0])
        print("  Bounds = (" + str(min_toPass) + ", " + str(max_toPass) + ")\n")
    
        while True:
    
            rules_tmp = rules_next
            tol = [em_tol, sup_tol]
    
            rules_next = list(filter(lambda x: self.is_emergent(x, classes, tol), rules))
            currentLength = len(rules_next)
    
            if currentLength > 0:
                min_rule_Cluster = min(list(Counter([r[1] for r in rules_next]).values()))
                str_tol = str(round(em_tol, 2))
                str_sup = str(round(sup_tol, 2))
                str_len = str(currentLength)
    
                if min_rule_Cluster < int(min_R_cl):
    
                    print("Using tolerance (" + str_tol + " " + str_sup + "), one cluster would have only " + str(
                        min_rule_Cluster) + " rules.")
                    print("We will consider the previous selection (" + str(len(rules_tmp)) + ").\n")
                    filtered_Rules = rules_tmp
                    min_rule_Cluster = min(list(Counter([r[1] for r in filtered_Rules]).values()))
                    return filtered_Rules, len(filtered_Rules), min_rule_Cluster
    
                elif currentLength >= length_0 * filter_ratio[1]:
                    print(str_len + " rules would be selected with tolerance (" + str_tol + " " + str_sup + ").")
                    em_tol -= 0.005
                    sup_tol -= 0.005
                elif currentLength <= length_0 * filter_ratio[0]:
                    print(str_len + " rules would be selected with tolerance (" + str_tol + " " + str_sup + ").")
                    em_tol += 0.02
                    sup_tol += 0.02
                else:
                    filtered_Rules = rules_next
                    print(str_len + " rules have been selected by EMERGENCE, out of " + str(length_0) + ".\n")
    
                    for l in [1, 2, 3, 4]:
                        nRules_byLength = len(list(filter(lambda x: len(x[0]) == l, filtered_Rules)))
                        print("Length " + str(l) + ":   " + str(nRules_byLength))
    
                    return filtered_Rules, currentLength, min_rule_Cluster
            else:
                filtered_Rules = rules_tmp
                print(str(len(filtered_Rules)) + " rules have been selected by EMERGENCE, out of " + str(length_0) + ".\n")
                min_rule_Cluster = min(list(Counter([r[1] for r in filtered_Rules]).values()))
                return filtered_Rules, len(filtered_Rules), min_rule_Cluster



    
    
    def filter_rules(self, S, classes, raw_rules, filter_ratio=[0.33, 0.66], resource=[1,1,1,1], min_R_cl=3,
                 min_L=3, criterion="all", coverage_filter=True, max_coverage=1,
                 emergence_filter=True,
                 reward_filter=False, min_ratio_Rewards=0.4, L_view=3):
        """
        Filters rules according to various mechanisms.
    
        Parameters
        ----------
        S : list
            List of all sequences, ignoring all labels.
        classes : list
            List of lists of circuits. Every class has label `k`.
        raw_rules : list
            List of all rules (rule = [pattern, label]) found by SequenceMining(.).
        dict_P2Nk : dictionary
            Dictionary matching every subsequence `P` (key) with the list (value) of sequences that contain it with label `k`.
        filter_ratio : list
            List ([min_ratio, max_ratio]) with the minimum and maximum fraction of rules that we allow to cut.
            This filter tries to adapt its tolerance to stay in this region.
        resources : list
            List of values corresponding to the availability of each type of operation (G1, G2, M1, M2).
            The higher this value, the more a type of operation is easy to perform or, in general, relavant.
        min_R_cl : int
            Minimum number of rules that we want to have in every class after the filter is applied.
            If the filter would not satisfy this criterion, the closest best possible cut is returned.
    
        coverage_filter : bool, optional (default=True)
            Whether to apply filter_byCOVERAGE(.)
        emergence_filter : bool, optional (default=True)
            Whether to apply filter_byEMERGENCE(.)
        reward_filter : bool, optional (default=False)
            Whether to apply filter_byREWARD(.)
    
        Other Parameters
        ----------
    
        tol : list, optional (default=[0.1,0.1]).
            Controls the strength of the filter. The lower the value, the more the filter will cut.
            raw_rules : list
        min_L : length
            Minimum length allowed by filter_byLENGTH(.). Shorter patterns will be discarded.
        criterion : {'at_least_L', 'exactly_L', 'all'}
            Whether filter_byLENGTH(.) selects rules long at least, or exactly, `min_L`.
            If 'all' is selected, analyses in filter_byLENGTH(.) are still performed but no rule is discarded.
        max_coverage : int, optional (default=1)
            Maximum number of times a sequence can be covered by a rule in filter_byCOVERAGE(.). The lower the value, the more it cuts.
        
            ...
        min_ratio_Rewards : float
            Controls the strength of the filter. The higher the value, the more the filter will cut.
        L_view : int, optional (default=3)
            Length of the rules we want to display.
    
        Notes
        ----------
            + This function combines filters from the literature and ad-hoc filters.
    
        See Also:
        ----------
        filter_byLENGTH(.), filter_byCOVERAGE(.), filter_byEMERGENCE(.), filter_byREWARD(.), Present_Rules(.)
    
        Returns
        ----------
        list
            List of rules (rule = [pattern, label]) that survived all filters.
        """
    
    
        # show_tmp = lambda x: self.present_rules(S, classes, x, resource, L_view=L_view, print_output=True,
        #                                     arrange="by_cluster", encoding='sum_resource', detailed=False, readable=True,
        #                                     permutations=False)
        print("\n\n__________________________\n")
    
        filtered_Rules, _, min_rule_Cluster = self.filter_bylength(raw_rules, min_L, criterion)
        # show_tmp(filtered_Rules)
    
        if emergence_filter == True and min_rule_Cluster >= min_R_cl:
            filtered_Rules, _, min_rule_Cluster = self.filter_byemergence(filtered_Rules, classes, filter_ratio,
                                                                     min_R_cl)
            # show_tmp(filtered_Rules)
    
        if coverage_filter == True and min_rule_Cluster >= min_R_cl:
            filtered_Rules, _, min_rule_Cluster = self.filter_bycoverage(filtered_Rules, S, classes, resource,
                                                                    max_coverage)
            # show_tmp(filtered_Rules)
    
        # if reward_filter == True and min_rule_Cluster >= min_R_cl:
            # filtered_Rules, _ = self.filter_byreward(filtered_Rules, filter_ratio, min_ratio_Rewards, env=env)
            # show_tmp(filtered_Rules)
    
        return filtered_Rules
    
            
    #-----------informative method--------------------#
    
    def describe_rule(self, P, S, S_k, decimals=3):
        """
        Utility function: returns a dictionary to summarize all the information of a pattern P in a list of sequences.
    
        Parameters
        ----------
        P   : list
            Subsequence
        S : list
            List of all sequences, ignoring all labels.
        S_k : list
            List of sequences for the class with label `k`.
        decimals : int
            Number of digits to use in round(.) when displaying values.
    
        See Also:
        ----------
        Present_Rules(.)
    
        Returns
        ----------
        list
            List of all elements in a pattern.
        dictionary
            Dictionary with Support, Cohesion, Interest, Confidence (keys) and their corresponding values.
        """
    
        Nk_P = self.N(P, S_k)
        
        return P, {"Support": round(self.support(S_k, Nk_P), decimals), "Cohesion": round(self.cohesion(P, Nk_P), decimals),
                   "Interest": round(self.interestingness(P, S_k, Nk_P), decimals),
                   "Confidence": round(self.confidence(P, S, Nk_P), decimals)}

    
    #-----------helper methods--------------------#
        
    #pairs only used in normal_ordering (see above).
    
    # def pairs(self, formatting):
    #     if formatting == "letters":
    #         modes = ['a', 'b', 'c', 'd', 'e', 'f', 'add_more'][:self.env.num_qudits]
    #         return [p for p in itertools.combinations(modes, 2)]  # Where two-qubit elements act on
    #     elif formatting == "digits":
    #         return [[str(p[0] + 1), str(p[1] + 1)] for p in
    #                 itertools.combinations(range(self.env.num_qudits), 2)]  # Where two-qubit elements act on

    
