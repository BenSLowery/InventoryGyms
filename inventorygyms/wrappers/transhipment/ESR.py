# Old don't use. Just a none rust version

import numpy as np
from gymnasium import Wrapper
import scipy.stats as sp
import functools


class ts_ESR(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.exp_calc = 0
        self.cached_configs = {}
        
    def _two_period_expectation(self, idx, on_hand, k, d1_vals, d1_pmf, d2_vals, d2_pmf):
        """
            Calculate the discritised expectation for two periods

            Note p is set internally by TM but in general we use 80%
        """
        # Precompute net transhipments to reduce calling function
        net_ts = self._net_transhipments_sum(idx, k)
        dfw_p = self.unwrapped.p

        # See if combo is already cached
        config = (idx,on_hand,k,net_ts,d1_pmf.index(max(d1_pmf)), d2_pmf.index(max(d2_pmf)))
        
        if config in self.cached_configs:
            return self.cached_configs[config]
        else:

            # Calculate q
            if self.r:
                q = min(max(self.st_out[idx]-on_hand-net_ts,0),self.r[idx])
            else:
                q = max(self.st_out[idx]-on_hand-net_ts,0)
                


            exp = 0
            exp_first_stage = 0 # Just immediate cost
            # Discretise
            for d_1,pmf_1 in zip(d1_vals, d1_pmf):
                # Calculate DFW random variable
                shortage_p_1 = max(d_1-on_hand-net_ts,0)
                beta_1_val = [i for i in range(int(shortage_p_1)+1)]
                beta_1_pmf = sp.binom._pmf(beta_1_val, shortage_p_1,dfw_p)
                
                # First stage shortage
                for b_1_v, b_1_p in zip(beta_1_val, beta_1_pmf):
                    fs_c = b_1_p*pmf_1*(max(d_1-on_hand-net_ts,0)-b_1_v)
                    exp += fs_c
                    exp_first_stage += fs_c


                # Second stage shortage
                for d_2, pmf_2 in zip(d2_vals, d2_pmf):
                    shortage_p_2 = max(d_2-max(net_ts+on_hand-d_1,0) - q,0)
                    beta_2_val = [i for i in range(int(shortage_p_2)+1)]
                    beta_2_pmf = sp.binom._pmf(beta_2_val, shortage_p_2,dfw_p)
                    for b_1_v, b_1_p in zip(beta_2_val, beta_2_pmf):
                        exp += b_1_p*pmf_1*pmf_2*(max(d_2-max(net_ts+on_hand-d_1,0) - q,0)-b_1_v)
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

            # Get store means to generate demand and pmf. We set a tolerance of 1e-7
            tol = 1e-7
            
            
            d_vals = {t: [], t+1: []}
            d_pmfs = {t: [], t+1: []}

            # Awful code plz forgive me
            if t >= self.unwrapped.periods - 1:
                t_la_all = [t]
            else:   
                t_la_all = [t,t+1]
            for t_la in t_la_all:
                for store in range(self.unwrapped.N):
                    gen_max_d = False
                    curr_max_val = 0
                    vals_i = []
                    pmf_i = []
                    while not gen_max_d:
                        vals_i.append(curr_max_val)
                        pmf_i_d = sp.poisson._pmf(curr_max_val,self.unwrapped.store_demand_means[t_la][store])
                        pmf_i.append(pmf_i_d)
                        if (pmf_i_d < tol):
                            gen_max_d = True
                        curr_max_val += 1
                    d_vals[t_la].append(vals_i)
                    d_pmfs[t_la].append(pmf_i)
            # If were in the final period we set the second stage demand to 0 with prob. 1
            if t >= self.unwrapped.periods - 1:
                for store in range(self.unwrapped.N):
                    d_vals[t+1].append([0])
                    d_pmfs[t+1].append([1])

            # Separate source and destination nodes
            source = [i for i in range(self.unwrapped.N) if IL[i] > 0]
            destination = [i for i in range(self.unwrapped.N)]


            # LOGIC: its very hacky 
            while (len(source) > 0) and (len(destination) > 0):
                alpha_val = [] # To find best source 
                alpha_val_immediate = []
                delta_val = [] # To find best destination
                delta_val_immediate = []
                
                for s_idx in source:
                    # Calculate expected shortages
                    alpha_minus_1, alpha_minus_1_immediate = self._two_period_expectation(s_idx,IL[s_idx], -1, d_vals[t][s_idx], d_pmfs[t][s_idx], d_vals[t+1][s_idx], d_pmfs[t+1][s_idx])
                    alpha_0, alpha_0_immediate = self._two_period_expectation(s_idx,IL[s_idx], 0, d_vals[t][s_idx], d_pmfs[t][s_idx], d_vals[t+1][s_idx], d_pmfs[t+1][s_idx])
                    alpha_val.append(alpha_minus_1-alpha_0)
                    alpha_val_immediate.append(alpha_minus_1_immediate-alpha_0_immediate)
                
                # Get arg min
                min_alpha = min(alpha_val)
                alpha = alpha_val.index(min_alpha)

                for d_idx in destination:
                    # Calculate expected shortages
                    delta_0, delta_0_immediate = self._two_period_expectation(d_idx,IL[d_idx], 0,  d_vals[t][d_idx], d_pmfs[t][d_idx], d_vals[t+1][d_idx], d_pmfs[t+1][d_idx])
                    delta_plus_1, delta_plus_1_immediate = self._two_period_expectation(d_idx,IL[d_idx], 1, d_vals[t][d_idx], d_pmfs[t][d_idx], d_vals[t+1][d_idx], d_pmfs[t+1][d_idx])
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

                    if (IL[source[alpha]] - self._net_transhipments_sum(source[alpha],0) <= 0):
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