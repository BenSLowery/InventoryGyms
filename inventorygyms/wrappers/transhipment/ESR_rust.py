# Expected shortage reduction heuristic policy (2 period lookahead)
# Will assume a lead-time of 1 for calculating expected shortages.
# Assumes poisson demand

# Written with a rust binding for speed

import numpy as np
from gymnasium import Wrapper
import scipy.stats as sp
import functools
import wrapper_alternative as rust_expectation


class ts_ESR(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.exp_calc = 0
        self.cached_configs = {}
        
    def _two_period_expectation(self, idx, on_hand, k, d1_params, d2_params, distribution='Poisson'):
        """
            Calculate the discritised expectation for two periods

            Note p is set internally by TM but in general we use 80%
        """
        # Precompute net transhipments to reduce calling function
        net_ts = self._net_transhipments_sum(idx, k)

        # See if combo is already cached
        config = (idx,on_hand,k,net_ts,d1_params, d2_params, distribution)
        
        if config in self.cached_configs:
            return self.cached_configs[config]
        else:
            if self.r is None:
                if distribution == 'Poisson':
                    # Go through the distributions which decides what we pass to the rust function
                    exp, exp_first_stage = rust_expectation.expectation(int(on_hand+net_ts), d1_params, d2_params, self.st_out[idx],p=self.unwrapped.p)
                elif distribution == 'Binomial':
                    exp, exp_first_stage = rust_expectation.expectation(int(on_hand+net_ts), d1_params[0], d2_params[0], self.st_out[idx], d1_params[1], d2_params[1], distribution='B',p=self.unwrapped.p)
                elif distribution == 'NegBin':
                    exp, exp_first_stage = rust_expectation.expectation(int(on_hand+net_ts), d1_params[0], d2_params[0], self.st_out[idx], d1_params[1], d2_params[1], distribution='N',p=self.unwrapped.p)
            else:
                if distribution == 'Poisson':
                    # Go through the distributions which decides what we pass to the rust function
                    exp, exp_first_stage = rust_expectation.expectation(int(on_hand+net_ts), d1_params, d2_params, self.st_out[idx], self.r[idx], distribution='P',p=self.unwrapped.p)
                elif distribution == 'Binomial':
                    # Go through the distributions which decides what we pass to the rust function
                    exp, exp_first_stage = rust_expectation.expectation(int(on_hand+net_ts), d1_params[0], d2_params[0], self.st_out[idx], d1_params[1], d2_params[1], r=self.r[idx], distribution='B',p=self.unwrapped.p)
                elif distribution == 'NegBin':
                    # Go through the distributions which decides what we pass to the rust function
                    exp, exp_first_stage = rust_expectation.expectation(int(on_hand+net_ts), d1_params[0], d2_params[0], self.st_out[idx], d1_params[1], d2_params[1], r=self.r[idx], distribution='N',p=self.unwrapped.p)
            # Add to cache
            self.cached_configs[config] = [exp, exp_first_stage]
            return exp, exp_first_stage


    def _net_transhipments_sum(self, i,k):
        """
            The s(i,k) function which tells you the net transhipments for store i given a change in k.
        """
        return sum([self.action_array[j+1,i+1]-self.action_array[i+1,j+1] for j in range(self.unwrapped.N)])+k

    def _allocate_stock(self, desired, avaialable):
        '''
            Allocate stock to the stores
        '''
        # Add online order to the desired quantity (since we need to keep some in for online demand)
        desired = [sp.poisson(self.unwrapped.online_demand_means[self.unwrapped.t]).ppf(self.unwrapped.cu/(self.unwrapped.cu+self.unwrapped.co_w))] + desired
        actual_allocation = [0 for i in range(len(desired))]

        for i in range(int(avaialable)):
            idx = np.argmax(desired)
            actual_allocation[idx] += 1
            desired[idx] -= 1
        # We dont' need to return the warehouse quantity since this will be implied from the other orders
        return actual_allocation[1:]

    def capped_base_stock_ordering_action(self,O_u_T, r=None):
        """
            Implementing an ordering policy based on the capped base-stock policy. And combining this with the transhipment to give a complete action to take
        """

        wh_out = O_u_T['warehouse']
        store_out = O_u_T['store']

        t = self.unwrapped.t
        IP_wh = self.unwrapped.state['warehouse'].sum()
        # Inventory position post transhipment
        IP_st = self.unwrapped.state['store'].sum(axis=1)-np.array([self._net_transhipments_sum(i,0) for i in range(self.unwrapped.N)])
        
        # Capacity constraints
        prod_cap = self.unwrapped.cap_prod
        warehouse_cap = self.unwrapped.cap_w
        store_cap = self.unwrapped.cap_s

        # Set order cap to the store OuT if one is not provided
        if r is None:
            r = store_out
        
        # Calculate warehouse inventory order (easy)
        total_inv = IP_wh + np.sum(IP_st)
        wh_order = max(wh_out - total_inv, 0)
        wh_order = min(wh_order, prod_cap) # Production contraint
        wh_order = min(wh_order, warehouse_cap - total_inv)

        # Calculate store inventory order (not as easy)
        desired_store_orders = []
        for store in range(self.unwrapped.N):
            inv_pipeline_store = IP_st[store]
            desired_store_q = max(store_out[store] - inv_pipeline_store, 0)
            desired_store_q = min(desired_store_q, store_cap - inv_pipeline_store)
            desired_store_q = min(desired_store_q, r[store]) # Order cap
            desired_store_orders.append(desired_store_q)

        total_store_orders = np.sum(desired_store_orders)
        if total_store_orders > self.unwrapped.I_o_H_wh[t]:
            desired_store_orders = self._allocate_stock(desired_store_orders, self.unwrapped.I_o_H_wh[t])

        return  [wh_order] + desired_store_orders

    def generate_action(self, ordering_type, ordering_action, transhipment=True):
        """
            Generate an expected shortage reduction action. For now assume ordering is just a vector-base-stock policy
            * ordering_type: the type of ordering policy. Available: "EchBS", "Capped"
            * ordering_action: a dictionary of the ordering policy corresponding to the type
            * transhipment: if a  transhipment is taking place this period
        """

        # Check the ordering policy
        self.st_out = ordering_action['store']

        if ordering_type == 'Capped':
            self.r = ordering_action['r']
        else:
            self.r = None

        self.action_array = np.zeros((self.unwrapped.N+1,self.unwrapped.N+1))

        t = self.unwrapped.t
        
        # Check we can actually lookahead
        if t == self.unwrapped.periods:
            return self.action_array
  
        IL = self.unwrapped.state['store'][:,0] # Inventory level for each store

        if transhipment == True:
            
            # Transhipment code
            ######################


            # Separate source and destination nodes
            source = [i for i in range(self.unwrapped.N) if IL[i] > 0]
            destination = [i for i in range(self.unwrapped.N)]

            # Calculate means
            d1_means = [self.unwrapped.store_demand_means[store][t] for store in range(self.unwrapped.N)]

            # Final stage means for each distribution (since this will be the terminal period so only a one period lookahead)
            final_stage_means = {'Poisson': 0, 'Binomial': (0,0), 'NegBin': (0,0)}

            if t >= self.unwrapped.periods - 1:
                d2_means = [final_stage_means[self.unwrapped.demand_distribution[store+1]] for store in range(self.unwrapped.N)]
            else:
                d2_means = [self.unwrapped.store_demand_means[store][t+1] for store in range(self.unwrapped.N)]

            # LOGIC: its very hacky 
            while (len(source) > 0) and (len(destination) > 0):
                alpha_val = [] # To find best source 
                alpha_val_immediate = []
                delta_val = [] # To find best destination
                delta_val_immediate = []
                
                for s_idx in source:
                    # Calculate expected shortages
                    alpha_minus_1, alpha_minus_1_immediate = self._two_period_expectation(s_idx,IL[s_idx], -1, d1_means[s_idx], d2_means[s_idx], self.unwrapped.demand_distribution[s_idx+1])
                    alpha_0, alpha_0_immediate = self._two_period_expectation(s_idx,IL[s_idx], 0, d1_means[s_idx], d2_means[s_idx], self.unwrapped.demand_distribution[s_idx+1])
                    alpha_val.append(alpha_minus_1-alpha_0)
                    alpha_val_immediate.append(alpha_minus_1_immediate-alpha_0_immediate)
                
                # Get arg min
                min_alpha = min(alpha_val)
                alpha = alpha_val.index(min_alpha)

                for d_idx in destination:
                    # Calculate expected shortages
                    delta_0, delta_0_immediate = self._two_period_expectation(d_idx,IL[d_idx], 0,  d1_means[d_idx], d2_means[d_idx], self.unwrapped.demand_distribution[d_idx+1])
                    delta_plus_1, delta_plus_1_immediate = self._two_period_expectation(d_idx,IL[d_idx], 1, d1_means[d_idx], d2_means[d_idx], self.unwrapped.demand_distribution[d_idx+1])
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

                    if (IL[source[alpha]] + self._net_transhipments_sum(source[alpha],0) <= 0):
                        source.remove(source[alpha])

                else: 
                    destination = []
                    source = []
                
        # Given the transhipment decision has been made, generate the acceptable order-up-to levels
        if ordering_type == 'EchBS':
            orders = self.capped_base_stock_ordering_action(ordering_action)
        elif ordering_type == 'Capped':
            # Terribly inefficiant
            orders = self.capped_base_stock_ordering_action({'warehouse': ordering_action['warehouse'], 'store': ordering_action['store']},ordering_action['r'])
        else:
            raise Exception('Unspecified ordering action given.')


        self.action_array[0] = orders

        return self.action_array