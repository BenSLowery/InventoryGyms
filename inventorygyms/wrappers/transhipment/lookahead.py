# 2-period lookahead policy determining the optimal order as part of the cost function

import numpy as np
from gymnasium import Wrapper
import scipy.stats as sp
import functools
import wrapper_alternative as rust_lookahead



class ts_la(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.exp_calc = 0
        self.cached_configs = {}
    
    def _calc_max_qs(self,d2_means):
        """
            Calculate the maximium orders for each store based on proportion of warehouse availability
        """
        means = []
        for idx, store_dist in enumerate(self.unwrapped.demand_distribution[1:]):
            if store_dist == 'Poisson':
                store_mean = d2_means[idx]
            elif store_dist == 'NegBin':
                store_mean = (d2_means[idx][0]*(1-d2_means[idx][1]))/(d2_means[idx][1]) if d2_means[idx][1] != 0 else 0
            elif store_dist == 'Binomial':
                store_mean = d2_means[idx][0]*d2_means[idx][1]
            means.append(store_mean)
        return [np.floor((store_mean/sum(means))*self.unwrapped.state['warehouse'][0]) for store_mean in means]
        
    def _lookahead(self, on_hand, k, d1_params, d2_params, max_order, distribution='Poisson'):
        
        """
            Lookahead policy, optimises transhipments in the first period and orders in the second
        """
        config = (on_hand+k, d1_params, d2_params, max_order, distribution)
        if config in self.cached_configs:
            return self.cached_configs[config]
        else:
            if distribution =='Poisson':
                exp, exp_first_stage, q = rust_lookahead.lookahead(int(on_hand+k), d1_params, d2_params, int(max_order), cu=self.unwrapped.cu, co=self.unwrapped.co_s, cdfw=self.unwrapped.c_dfw,p=self.unwrapped.p)
            elif distribution == 'NegBin':
                exp, exp_first_stage, q = rust_lookahead.lookahead(int(on_hand+k), d1_params[0], d2_params[0], int(max_order), d1_params[1], d2_params[1],  distribution='N',cu=self.unwrapped.cu, co=self.unwrapped.co_s, cdfw=self.unwrapped.c_dfw,p=self.unwrapped.p)
            elif distribution == 'Binomial':
                exp, exp_first_stage, q = rust_lookahead.lookahead(int(on_hand+k), d1_params[0], d2_params[0], int(max_order), d1_params[1], d2_params[1],  distribution='B',cu=self.unwrapped.cu, co=self.unwrapped.co_s, cdfw=self.unwrapped.c_dfw,p=self.unwrapped.p)
            # Add to cache
            self.cached_configs[config] = [exp, exp_first_stage, q]

            return (exp,exp_first_stage,q)
        


    def _net_transhipments_sum(self):
        """
            The s(i,k) function which tells you the net transhipments for store i given a change in k.
        """
        return [sum([self.action_array[j+1,i+1]-self.action_array[i+1,j+1] for j in range(self.unwrapped.N)]) for i in range(self.unwrapped.N)]

    def generate_action(self, warehouse_order, transhipment=True):
        # Assumes the warehouse operates an order up to policy
        self.action_array = np.zeros((self.unwrapped.N+1,self.unwrapped.N+1))

        t = self.unwrapped.t
        
        # Check we can actually lookahead
        if t == self.unwrapped.periods:
            return self.action_array
  
        IL = self.unwrapped.state['store'][:,0] # Inventory level for each store

        # Calculate means
        d1_means = [self.unwrapped.store_demand_params[store][t] for store in range(self.unwrapped.N)]

        # Final stage means for each distribution (since this will be the terminal period so only a one period lookahead)
        final_stage_means = {'Poisson': 0, 'Binomial': (0,0), 'NegBin': (0,0)}

        if t >= self.unwrapped.periods - 1:
            d2_means = [final_stage_means[self.unwrapped.demand_distribution[store+1]] for store in range(self.unwrapped.N)]
        else:
            d2_means = [self.unwrapped.store_demand_params[store][t+1] for store in range(self.unwrapped.N)]

        if transhipment == True:

            # Transhipment code
            ######################

            # Separate source and destination nodes
            source = [i for i in range(self.unwrapped.N) if IL[i] > 0]
            destination = [i for i in range(self.unwrapped.N)]

            # Max q for each store
            max_q = self._calc_max_qs(d2_means) if self.unwrapped.t < self.unwrapped.periods - 1 else [0 for i in range(self.unwrapped.N)]

            # LOGIC: its very hacky 
            while (len(source) > 0) and (len(destination) > 0):
                # Get transhipments
                ts_sums = self._net_transhipments_sum()

                alpha_val = [] # To find best source 
                alpha_val_immediate = []
                delta_val = [] # To find best destination
                delta_val_immediate = []
                
                for s_idx in source:
                    # Calculate expected shortages
                    alpha_minus_1, alpha_minus_1_immediate, q = self._lookahead(IL[s_idx]+ts_sums[s_idx], -1, d1_means[s_idx], d2_means[s_idx], max_q[s_idx], self.unwrapped.demand_distribution[s_idx+1])
                    alpha_0, alpha_0_immediate, _ = self._lookahead(IL[s_idx]+ts_sums[s_idx], 0, d1_means[s_idx], d2_means[s_idx], max_q[s_idx], self.unwrapped.demand_distribution[s_idx+1])
                    alpha_val.append(alpha_minus_1-alpha_0)
                    alpha_val_immediate.append(alpha_minus_1_immediate-alpha_0_immediate)
                
                # Get arg min
                min_alpha = min(alpha_val)
                alpha = alpha_val.index(min_alpha)

                for d_idx in destination:
                    # Calculate expected shortages
                    delta_0, delta_0_immediate, _ = self._lookahead(IL[d_idx]+ts_sums[d_idx], 0,  d1_means[d_idx], d2_means[d_idx], max_q[d_idx], self.unwrapped.demand_distribution[d_idx+1])
                    delta_plus_1, delta_plus_1_immediate, _ = self._lookahead(IL[d_idx]+ts_sums[d_idx], 1, d1_means[d_idx], d2_means[d_idx], max_q[d_idx], self.unwrapped.demand_distribution[d_idx+1])
                    delta_val.append(delta_0-delta_plus_1)
                    delta_val_immediate.append(delta_0_immediate-delta_plus_1_immediate)
                
                # Get arg max
                max_delta = max(delta_val)
                delta = delta_val.index(max_delta) 

                if ((max_delta-min_alpha) > self.unwrapped.c_ts/self.unwrapped.cu) and ((delta_val_immediate[delta]) >= (alpha_val_immediate[alpha])):
                    # Make the transhipment
                    self.action_array[source[alpha]+1, destination[delta]+1] += 1
                    
                    # Check we haven't done a transhipment to ourselves. If so we raise an error
                    if np.diag(self.action_array).sum() > 0:
                        raise Exception('Stuck in an infinite transhipment loop :(')
                    
                    if (IL[source[alpha]] + ts_sums[source[alpha]]<= 0):
                        source.remove(source[alpha])

                else: 
                    destination = []
                    source = []
        # Calculate the transhipment matrix and each individual q to return
        final_ts_sums = self._net_transhipments_sum()
        # The warehouse orders an echelon base-stock policy
        orders = [max(warehouse_order-self.unwrapped.state['warehouse'][0],0)] + [self._lookahead(IL[s]+final_ts_sums[s], 0, d1_means[s], d2_means[s], max_q[s], self.unwrapped.demand_distribution[s+1])[-1] for s in range(self.unwrapped.N)]
        self.action_array[0] = orders

        return self.action_array
