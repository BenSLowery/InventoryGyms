import numpy as np
from gymnasium import Wrapper
import scipy.stats as sp

class EchelonBaseStock(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _get_inventory_position(self):
        '''
            Get inventory position
        '''
        return {'Warehouse': self.unwrapped.state['warehouse'].sum(), 'store': self.unwrapped.state['store'].sum(axis=1)}

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

    def base_stock_action(self,O_u_T, r=None):
        '''
            Sample a base-stock action
            O_u_T = [Integer, ndarray]: Base-stock level
        '''

        wh_out = O_u_T['warehouse']
        store_out = O_u_T['store']

        t = self.unwrapped.t
        IP = self._get_inventory_position() # Inventory Position
        # Capaicity constraints
        prod_cap = self.unwrapped.cap_prod
        warehouse_cap = self.unwrapped.cap_w
        store_cap = self.unwrapped.cap_s

        # Set order cap to the store OuT if one is not provided
        if r is None:
            r = store_out
        
        # Calculate warehouse inventory order (easy)
        total_inv = IP['Warehouse'] + np.sum(IP['store'])
        wh_order = max(wh_out - total_inv, 0)
        wh_order = min(wh_order, prod_cap) # Production contraint
        wh_order = min(wh_order, warehouse_cap - total_inv)

        # Calculate store inventory order (not as easy)
        desired_store_orders = []
        for store in range(self.unwrapped.N):
            inv_pipeline_store = IP['store'][store]
            desired_store_q = max(store_out[store] - inv_pipeline_store, 0)
            desired_store_q = min(desired_store_q, store_cap - inv_pipeline_store)
            desired_store_q = min(desired_store_q, r[store]) # Order cap
            desired_store_orders.append(desired_store_q)

        total_store_orders = np.sum(desired_store_orders)
        if total_store_orders > self.unwrapped.I_o_H_wh[t]:
            desired_store_orders = self._allocate_stock(desired_store_orders, self.unwrapped.I_o_H_wh[t])
        
        return [wh_order] + desired_store_orders
