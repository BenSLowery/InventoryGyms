'''
   Model w/ Transshipments
'''

import gymnasium as gym
import numpy as np

class TwoEchelonPLSTS(gym.Env):
    '''
        lead_time = [warehouse, store, transhipment]
    '''
    def __init__(self, periods=100, stores=2, lead_time=[2,2,0],  production_capacity=1000, warehouse_capacity=1000, store_capacity=200, dfw_chance=0.8, dfw_cost=0, ts_cost=1, holding_warehouse=1, holding_store=1, penalty=18, initial_inventory=[[50,0,0],[50,0,0],[50,0,0]],  online_demand_means=[0 for i in range(100)], store_demand_means = [[5,5] for i in range(100)]):
        self.periods = periods
        self.N = stores
        self.wh_lt = lead_time[0]
        self.st_lt = lead_time[1]
        self.ts_lt = lead_time[2]
        self.cap_prod = production_capacity
        self.cap_w = warehouse_capacity
        self.cap_s = store_capacity
        self.p = dfw_chance
        self.c_dfw = dfw_cost
        self.c_ts = ts_cost
        self.co_w = holding_warehouse
        self.co_s = holding_store
        self.cu = penalty
        self.init_warehouse = initial_inventory[0]
        self.init_store = initial_inventory[1:]
        self.online_demand_means = online_demand_means
        self.store_demand_means = store_demand_means

        # Check everything entered was correct
        # TODO: Add assertions
        assert self.wh_lt > 0, "You need a positive warehouse lead time."
        assert self.st_lt > 0, "You need a positive store lead time."
        assert self.ts_lt <= self.st_lt, "The transhipment lead time cannot be more than the warehouse lead-time"

        

        # Initialise
        self.reset()

        # Set up action space
        # For the transhipment we create a matrix of size N+1
        # For each position pairing (i,j) is how much stock i sends j. 
        # Array index 0 is the warehouse
        action_array = [[self.cap_w] + [self.cap_s for s in range(self.N)]]
        for i in range(self.N):
            action_array.append([1] + [self.cap_s if s!= i else 1 for s in range(self.N)])

        self.action_space = gym.spaces.MultiDiscrete(np.array(action_array), dtype=np.int16)
        # Set up state space
        self.observation_space = gym.spaces.Dict({
            "warehouse": gym.spaces.Box(low=np.zeros(self.wh_lt+1), high=np.ones(self.wh_lt+1)*self.cap_w, dtype=np.int32), 
            "store": gym.spaces.Box(low=np.zeros((self.N, self.st_lt+1)), high=np.ones((self.N,self.st_lt+1))*self.cap_s, dtype=np.int32)
        })


    def net_transhipments(self, ts_action):
        stores = self.N
        s_sum = np.zeros(stores)
        for i in range(stores):
            s_sum[i] = sum([ts_action[j,i]-ts_action[i,j] for j in range(stores)])
        return s_sum

    def _RESET(self):
        T = self.periods
        init_inv_wh = self.init_warehouse[0]
        init_inv_st = np.array(self.init_store)[:,0]

        # Information on the simulation
        self.I_o_H_wh = np.zeros(T+1) # Warehouse inventory on hand
        self.I_o_H_st = np.zeros((self.N, T+1)) # Store inventory on hand [store][t]
        self.D = np.zeros((self.N+1, T)) # Demand
        self.dfw_fulfillment = np.zeros((self.N,T)) # Direct From Warehouse demand
        self.C_wh = np.zeros(T) # Warehouse Period cost
        self.C_st = np.zeros((self.N, T)) # Store Period cost

        # Initialise
        self.t = 0 # Initial period
        self.I_o_H_wh[0] = init_inv_wh
        self.I_o_H_st[:,0] = init_inv_st

        self.wh_actions_log = np.zeros(T, dtype=np.int32) # Logs the warehouse actions for the pipeline, i.e. how much got added
        self.st_actions_log = np.zeros((self.N, T), dtype=np.int32) # Logs the store actions for the pipeline
        self.st_actions_ts_log = np.zeros((self.N, T), dtype=np.int32) # Logs transhipment *incoming* actions (how many are added to the pipeline of store i)

        self._update_state()

        return self.state, {}
    
    # To update the state, you can either: 'order', 'move', or 'fetch'
    def _update_state(self,type='fetch'):
        if type == 'fetch':
            self.state =  self.fetch_state()
        elif type == 'move':
            self.state =  self.move_state()
    

    def move_state(self):
        # Moves the pipeline forward
        t = self.t
        state = self.state.copy()

        
        # Warehouse 
        state['warehouse'][-1] = self.wh_actions_log[t-1]
        state['warehouse'][0] = self.I_o_H_wh[t]+state['warehouse'][1]
        for pos in range(1,self.wh_lt):
            state['warehouse'][pos] = state['warehouse'][pos+1]
        state['warehouse'][-1]  = 0
        
        # Store
        for store in range(self.N):
            state['store'][store][-1] = self.st_actions_log[store,t-1]
            if self.ts_lt > 0:
                state['store'][store][self.ts_lt] = self.st_actions_ts_log[store][t-1]
            state['store'][store][0] = self.I_o_H_st[store,t]+state['store'][store][1]
            for pos in range(1,self.st_lt):
                state['store'][store][pos] = state['store'][store][pos+1]
            state['store'][store][-1] = 0

        self.I_o_H_wh[t] = state['warehouse'][0]
        self.I_o_H_st[:,t] = state['store'][:,0]
        return state
    
    def fetch_state(self):
        # Fetch the current state
        t = self.t
        wh_lt = self.wh_lt
        st_lt = self.st_lt

        # The state is a dictionary, split by warehouse and store
        # As the state is returned by the reset and step functions, they need to be in the same form as the obsevations space.
        state = {
            'warehouse': np.zeros(self.wh_lt+1,dtype=np.int32),
            'store': np.zeros((self.N, self.st_lt+1),dtype=np.int32)
        }

        # Get initial state
        if t == 0:
            state['warehouse'] = np.array(self.init_warehouse,dtype=np.int32)
            state['store'] = np.array(self.init_store, dtype=np.int32)

            self.wh_actions_log[:len(self.init_warehouse)-1] = self.init_warehouse[1:]
            self.st_actions_log[:,:len(self.init_store[0])-1] = np.array(self.init_store, dtype=np.int32)[:,1:]
        else:
            state['warehouse'][0] = self.I_o_H_wh[t]
            state['store'][:,0] = self.I_o_H_st[:,t]

            # add the ordering decisions to the pipeline (if we're after the lead time period, otherweise add as much as we have)
            if t >= wh_lt:
                state['warehouse'][-wh_lt:] = self.wh_actions_log[t-wh_lt:t]
            else:
                state['warehouse'][-t:] = self.wh_actions_log[:t]
            if t >= st_lt: 
                state['store'][:,-st_lt:] = self.st_actions_log[:,t-st_lt:t]
            else:
                state['store'][:,:-t] = self.st_actions_log[:,:t]


        return state
    
    def _STEP(self, action):
        '''
            Take a step forward in time in the inventory problem
        '''

        # TODO: make sure transhipments are acceptable with the current levels at store (same with warehouse not allowing negative orders)- i think this is covered in the ordering decision
        # Order requests (the action is calculated using a wrapped function so will give the allocation for each warehouse and store)
        orders = np.maximum(action[0], 0).astype(int)
        transhipments = np.maximum(action[1:,1:],0).astype(int)

        t = self.t
        I_o_H_wh = self.I_o_H_wh[t]
        I_o_H_st = self.I_o_H_st[:,t]


        # Initial state (to return at the end)
        starting_state = {'warehouse': self.state['warehouse'].copy(), 'store': self.state['store'].copy()}

        # Update the action log and take off orders from the warehouse
        self.wh_actions_log[t] += orders[0].copy()
        self.st_actions_log[:,t] += orders[1:].copy()
        I_o_H_wh -= np.sum(orders[1:])
        
        # Check the inventory on hand is not negative at the warehouse, if so we need to let the user know an allocation policy is needed
        assert I_o_H_wh >= 0, "You're orders to the stores ({}). have exceeded on hand warehouse ({}). Please implement an allocation policy in your ordering wrapper.".format(np.sum(orders[1:]), starting_state['warehouse'][0])

        # Transhipment movements
        # Axis=1 is rows and Axis=0 will be columns. The rows are stuff out, and columns are stuff in
        self.st_actions_ts_log[:,t] = transhipments.sum(axis=0)
        transhipments_out = transhipments.sum(axis=1) 

        I_o_H_st -= transhipments_out
        if (self.ts_lt == 0): # Since transhipments can be zero lead-time we add to on hand if this is the case
            I_o_H_st += transhipments.sum(axis=0)
            


        state_after_inv_wh = I_o_H_wh.copy()
        state_after_inv_st = I_o_H_st.copy()




        # Generate demand
        D_wh = self.np_random.poisson(self.online_demand_means[t])
        D_st = self.np_random.poisson(self.store_demand_means[t])

        # Append to the demand vector
        self.D[:,t] = np.concatenate(([D_wh], D_st))

        # Realise demand
        I_o_H_wh -= D_wh
        I_o_H_st -= D_st

        # Calculate the DFW fulfilment for each store
        for store_idx, store in enumerate(I_o_H_st):
            if store < 0 :
                dfw_request = self.np_random.binomial(np.abs(store), self.p)

                # We can only give DFW from what's available in the warehouse
                dfw_request = np.minimum(dfw_request, np.maximum(I_o_H_wh,0))
                self.dfw_fulfillment[store_idx,t] = dfw_request    

                # Take the DFW from the warehouse and reduce store stockout
                I_o_H_wh -= dfw_request
                I_o_H_st[store_idx] += dfw_request
            else:
                self.dfw_fulfillment[store_idx,t] = 0

        # Calculate the period costs
        store_holding = self.co_s
        warehouse_holding = self.co_w
        penalty = self.cu
        dfw_cost = self.c_dfw
        transhipment_cost = self.c_ts

        # WH
        cost_wh = np.abs(I_o_H_wh)*penalty if I_o_H_wh<=0 else I_o_H_wh*warehouse_holding
        self.C_wh[t] = cost_wh

        # ST
        cost_st = []
        for store_idx, store in enumerate(I_o_H_st):
            cost_st.append(np.abs(store)*penalty if store<=0 else store*store_holding)
            cost_st[-1] += dfw_cost*self.dfw_fulfillment[store_idx,t]+transhipment_cost*transhipments_out[store_idx]
    
        self.C_st[:,t] = cost_st

        
        # Update pipeline movements (add orders and calculate I_o_H for next period)
        self.I_o_H_wh[t+1] = max(I_o_H_wh,0)

        self.I_o_H_st[:,t+1] = np.maximum(I_o_H_st,0)
 

        self.t = t+1
        self._update_state(type='move')

        # Set the reward to return (negative cost as RL works on maximising reward)
        reward = -(cost_wh+sum(cost_st))

        # Check if we are at the end of the simulation
        if self.t >= self.periods:
            terminated = True   
        else:
            terminated = False
        
        # Set truncated to False as we don't need this
        # Also return a dictionary with information on the simulation
        info = {
            'Starting Inv. Warehouse': starting_state['warehouse'],
            'Starting Inv. Store': starting_state['store'],
            'Transhipments': self.net_transhipments(action[1:,1:]),
            'Order': orders,
            'Post Order Warehouse': state_after_inv_wh,
            'Post Order Store': state_after_inv_st,
            'Demand': self.D[:,t], 
            'DFW': self.dfw_fulfillment[:,t], 
            'Ending Inventory Warehouse': self.state['warehouse'].copy(),
            'Ending Inventory Store': self.state['store'].copy(),
            'Period Cost': -reward
        }
        return self.state, reward, terminated, False, info
    def step(self, action):
        return self._STEP(action)
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed = seed)
        return self._RESET()
    
    def sample_action(self):
        return self.action_space.sample()

