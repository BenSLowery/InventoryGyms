# A tie policy based on mean demand. With various ordering policies to use alongside
import numpy as np
from gymnasium import Wrapper
import scipy.stats as sp

class ts_TIE(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def _get_inventory_level(self):
        '''
            Get inventory level of the store for the transhipment lead-time
        '''
        return self.unwrapped.state['store'][:,0]

    def _get_inventory_position(self):
        '''
            Get inventory position and add transhipment decisions (as these will be part of the inventory position pipeline)
        '''
        return {'Warehouse': self.unwrapped.state['warehouse'].sum(), 'store': self.unwrapped.state['store'].sum(axis=1)+self._net_transhipments()}

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

    def _net_transhipments(self):
        """
            Calculate the net transhipments for store i (index starting at 0)
        """
        stores = self.unwrapped.N
        s_sum = np.zeros(stores)
        for i in range(stores):
            s_sum[i] = sum([self.transhipment_arr[j,i]-self.transhipment_arr[i,j] for j in range(stores)])
        return s_sum


    def capped_base_stock_ordering_action(self,O_u_T, r=None):
        """
            Implementing an ordering policy based on the capped base-stock policy. And combining this with the transhipment to give a complete action to take
        """

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

        return  [wh_order] + desired_store_orders


    def transhipment_action(self):
        '''
            An equalised transhipment action.
        '''

        t = self.unwrapped.t
        IL = self._get_inventory_level() # Inventory Position of stores for the transhipment lead-time

        # Get the demand over this lead time
        demand_stores = np.array(self.unwrapped.store_demand_means[t])

        # Calculate equalised inventories for each store
        eq_func = lambda d: d/demand_stores.sum()*IL.sum() # Lambda function to calculate equalised inventory
        eq_inv = eq_func(demand_stores)


        # Check if everything divided nicely into an integer
        # if so we can just move onto the movements
        # Otherwise, we need to allocate such that everything is an integer
        if all([i.is_integer() for i in eq_inv]) == False:
            # TODO: HAVE BETTER ALLOCATION RULING
            # Currently just all the excess is allocated at random to places with fractional values
            # At most stores can get one additional unit
            
            # Round down and see how much we have to randomly allocate and where we can allocate
            eq_inv_new = np.floor(eq_inv)
            alloc_num = int(eq_inv.sum()-eq_inv_new.sum())
            alloc_idx = np.where(eq_inv != eq_inv_new)[0]

            # Choose locations to increase the allocation
            increasing = np.random.choice(alloc_idx, alloc_num, replace=False)
            eq_inv_new[increasing] += 1
            eq_inv = eq_inv_new
            
        # Work out movements
        inv_trans = IL-eq_inv

        # Execute location to where we tranship too 
        # Currently assume no distance based costs so we rank and allocate
        source = list(np.where(inv_trans > 0)[0]) # Pick up
        destination = list(np.where(inv_trans < 0)[0]) # Deliver
        
        # Create empty transhipment array, omit the first row (which are ordering decisions),
        # the first column will be zero as its transhipments from the store to the warehouse which we do not allow
        transhipment_arr = np.zeros((self.unwrapped.N, self.unwrapped.N))
        for idx, S in enumerate(inv_trans):
            avail_inv = S
            if idx in source:
                satisfied = 0
                for dest in destination:
                    fulfilment_need =  inv_trans[dest]
                    avail_to_move = min(avail_inv,abs(fulfilment_need))
                    avail_inv = max(avail_inv+fulfilment_need,0)

                    # Send this much to the destination from the source
                    transhipment_arr[idx,dest] = avail_to_move

                    # Some checks to keep computing
                    if avail_to_move == fulfilment_need:
                        satisfied +=1
                    else:
                        inv_trans[dest] += avail_to_move
                    if avail_inv == 0:
                        break
                # Remove the destination nodes for the next source location
                destination = destination[satisfied:]

        return transhipment_arr


    def generate_action(self,ordering_type, ordering_action, transhipment=True):
        """ 
            Combines a transhipment and ordering action together
            * ordering_type: the type of ordering policy. Available: "EchBS", "Capped"
            * ordering_action: a dictionary of the ordering policy corresponding to the type
            * transhipment: if a TIE transhipment is taking place this period
        """

        action_array = np.zeros((self.unwrapped.N+1,self.unwrapped.N+1))
        # See if a transhipment is occuring, if so we generate the matrix, otherwise we generate a matrix of zeros since no movements are occuring (but is required for the simulation)
        if (transhipment):
            transhipment_array = self.transhipment_action()
        else:
            transhipment_array = np.zeros((self.unwrapped.N,self.unwrapped.N))
        self.transhipment_arr = transhipment_array
        action_array[1:,1:] = transhipment_array

        # Given the transhipment decision has been made, generate the acceptable order-up-to levels
        if ordering_type == 'EchBS':
            orders = self.capped_base_stock_ordering_action(ordering_action)
        elif ordering_type == 'Capped':
            # Terribly inefficiant
            orders = self.capped_base_stock_ordering_action({'warehouse': ordering_action['warehouse'], 'store': ordering_action['store']},ordering_action['r'])
        else:
            raise Exception('Unspecified ordering action given.')

        action_array[0] = orders
        return action_array
