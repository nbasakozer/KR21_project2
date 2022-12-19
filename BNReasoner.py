from typing import Union, List, Dict
from BayesNet import BayesNet
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from itertools import product
from copy import deepcopy

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: This is where your methods should go


    def min_degree_order(self, network: BayesNet) -> List[str]:
        
        '''
        Minimum degree ordering: 
        
        * Queue variable X' ∈ X ⊆ V with the minimum degree in the interaction graph to the ordering.
        * Sum-out X' from the interaction graph.
        * Repeat until all variables X are summed-out.
       
        '''

        elim_order = []

        # get interaction graph from BayesNet class
        int_graph = network.get_interaction_graph()

        for i in range(len(int_graph.nodes)):
            '''
            # show elimination step by step
            plt.figure(i)
            nx.draw_networkx(int_graph, with_labels=True)
            plt.show()
            '''
            # queue variable with the minimum degree in the interaction graph to the ordering
            degrees = {}
            for node in int_graph.nodes:
                degrees[node] = int_graph.degree(node) # degree of node
            
            var_min_degree = min(degrees, key=degrees.get)
            elim_order.append(var_min_degree)
            
            # sum-out from the interaction graph
            neighbor_of_min = []
            for neighbor in int_graph.neighbors(var_min_degree):
                neighbor_of_min.append(neighbor)
            
            if len(neighbor_of_min) >= 2:
                for c in itertools.combinations(neighbor_of_min, 2):
                    int_graph.add_edge(c[0], c[1])

            int_graph.remove_node(var_min_degree)

        return elim_order
           

    def min_fill_order(self, network: BayesNet) -> List[str]:
        
        '''
        Minimum fill ordering: 
        
        * Queue variable X' ∈ X whose deletion would add the fewest new interactions to the ordering.
        * Sum-out X from the interaction graph.
        * Repeat until all variables X are summed-out.
       
        '''

        elim_order = []

        # get interaction graph from BayesNet class
        int_graph = network.get_interaction_graph()
        
        for i in range(len(int_graph.nodes)):
            '''
            # show elimination step by step
            plt.figure(i)
            nx.draw_networkx(int_graph, with_labels=True)
            plt.show()
            '''
            # queue variable whose deletion would add the fewest new interactions to the ordering
            new_inters = {}

            for node in int_graph.nodes:
                
                # create the "next graph" (i.e. the graph with one less node after elimination) 
                # see how many new edges would be added by subtracting total number of edges of each graph from each other
                next_graph = deepcopy(int_graph)
                next_graph.remove_node(node)
                new_inters[node] = int_graph.number_of_edges() - next_graph.number_of_edges()             
            
            var_min_degree = min(new_inters, key=new_inters.get)
            elim_order.append(var_min_degree)

            # sum-out from the interaction graph
            neighbor_of_min = []
            for neighbor in int_graph.neighbors(var_min_degree):
                neighbor_of_min.append(neighbor)
            
            if len(neighbor_of_min) >= 2:
                for c in itertools.combinations(neighbor_of_min, 2):
                    int_graph.add_edge(c[0], c[1])

            int_graph.remove_node(var_min_degree)

        return elim_order


    def pruning(self, query_vars: List[str], evidence: Dict[str,bool]) -> BayesNet: # for MAP-MPE
        
        graph = deepcopy(self.bn)
        qUe = query_vars + list(evidence.keys()) # union
        
        for e in list(evidence.keys()): # edge pruning
            for child in graph.get_children(e):
                graph.del_edge([e, child])
                updated_cpt =  graph.get_compatible_instantiations_table(pd.Series(evidence), graph.get_cpt(child)).reset_index(drop=True)
                updated_cpt = updated_cpt.drop(e, axis=1) # remove evidence
                graph.update_cpt(child, updated_cpt)
            u_cpt =  graph.get_compatible_instantiations_table(pd.Series(evidence), graph.get_cpt(e)).reset_index(drop=True)
            graph.update_cpt(e, u_cpt)
        
        while True: # leaf node pruning
            l_nodes = []
            for v in graph.get_all_variables():
                if graph.get_children(v) == []:
                    l_nodes.append(v)
            for node in l_nodes: 
                if node not in qUe:
                    graph.del_var(node)
            if set(l_nodes) - set(qUe) == set():
                break

        return graph


    def maxing_out(self, factor: pd.DataFrame, variable: str) -> pd.DataFrame:
        """
        :param factor: table
        :param variable: variable to max out 
        """

        vars = []
        for var in factor:
            if var != variable and var != 'p':
                vars.append(var)

        var_values = []
        var_len = len(vars)

        # assign all possible True False combinations to variables
        for bool_combos in product([True, False], repeat=var_len):

            var_dict = {}
            for v, var in enumerate(vars):
                var_dict[var] = bool_combos[v]

            var_values.append(var_dict)

        # match bool combinations to the values in original cpt then sum out
        for var_value in var_values:
            
            com_cpt = self.bn.get_compatible_instantiations_table(pd.Series(var_value), factor)
            max_p = com_cpt['p'].max()
            var_bool = bool
            for var in com_cpt.iterrows():
                if var[1]['p'] == max_p:
                    var_bool = var[1][variable]

            if 'p' not in var_value:
                var_value['max p'] = str(f'{max_p}, {variable}={var_bool}')

        return pd.DataFrame(var_values)