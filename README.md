# An inventory gym environment 

Current Version: `0.1.3`

Used as a testing and evaluation environment for PhD research into stochastic inventory policies. The underlying problem is formulated as a Stochastic Dynamic Program, however, due to the curse of dimensionality, and to allow for high-fidelity modelling, we choose to use a simulation environment to test and run policies.

Requirements
============
```
"gymnasium==0.29.1"
"pandas==2.2.3"
"numpy==2.1.3"
"scipy==1.14"
```

Also need the Rust implmentation of some heuristic features to use the wrappers. Download from [here](https://github.com/BenSLowery/wrapper_alternative) and [use maturin to build the wheel](https://github.com/PyO3/maturin) for your system. 

Environments
=============
The environment is based on using the [gymnasium](https://gymnasium.farama.org/) API standard. To make it easier for those who wish to apply or compare the heuristics used with their own RL/DeepRL methods.

There is currently one available environment to be used: `inventorygyms/TwoEchelonPLSTS-v0`. This is a discrete event system operating with partial lost sales and optional transhipments. It has a default setting, however, it is advised to also enter a custom configuration when initialising. Namely, the following parameters
```python
config = {
        'periods': int // Number of periods //,
        'stores': int // Number of stores //,
        'lead_time': list(int) // [Warehouse lead time, store lead time, transhipment lead time]//,
        'ts_cost': int // Cost of a transhipment //,
        'dfw_cost': int // Cost of direct from warehouse fulfilment //,
        'penalty': int // Penalty cost per unit shortage//,
        'holding_warehouse': int // per unit holding cost //,
        'holding_store': int // per unit holding cost //,
        'initial_inventory': list(list(int)) // Initial inventory on hand and due to arrive at warehouse and store // ,
        'online_demand_params': list(int) // Online demand parameters for each period // ,
        'store_demand_params':  list(list(int)) // Store demand params, a list of lists where each list is the stores demand params for each period // ,
        'demand_distribution': list(String) // The demand distribution for the warehouse and then each store // ,
        'dfw_chance': float // The chance of direct from warehouse // ,
    }
```
There are some (not exhaustive) `assert` statements to prevent incorrect inputs. 

To create a new instance:
```python 
import gymnasium as gym
import inventorygyms

env = gym.make('inventorygyms/TwoEchelonPLSTS-v0', **config)
```

Wrappers
===========
To implement policies we need to wrap the environment (you shouldn't really run the environment without using one of the wrappers, as there is less error checking in the base environment. i.e. allocation policies to prevent negative inventory). These wrappers allow you to generate actions, which are the ordering policies for the warehouse, stores and transhipments. 

#### Ordering/Transhipments (Action Space)
The action space takes the form of a $(N+1)\times(N+1)$ matrix; where $N$ is the number of stores. For a row $i$ and column $j$, a pair of co-ordinates $(i,j)$ gives the movement from place $i$ to place $j$. When $i=0$ then this is all deliveries, with (0,0) being from the supplier to the warehouse and $(0,1), (0,2),..., (0,N)$ gives the orders for store $j$. Note, that orders cannot be delivered to themselves so $i\neq j$ if $i\neq 0$. Further for all the heuristics, stock cannot be moved back upstream to the store. So, $(i,0)$ is 0. for $i\neq 0$. 

There are 3 wrappers offered for transhipment decisions. If you do not need a transhipment, Use the `ESR_rust` wrapper and set `transhipments=False` in the argument when generating an action.

### Ordering policies
The ordering policies are baked into the transhipment wrappers. We assume for `demand_tie` and `ESR_rust` an echelon base-stock policy is implemented with the option of order caps for each store. Whilst the `lookahead` generates on the go ordering decisions, so only requires an echelon warehouse level.

### Transhipment Policies
* `demand_tie` - An implementation of Transshipment Inventory Equalisation (TIE). There are 3 arguments in the action with the following input options/structure
  ```python
  generate_action(
    ordering_type=['EchBS','Capped'],
    ordering_action={'warehouse': int, 'store': list(int), 'r': list(int)},
    transhipment=[True, False]
  )
  ```
* `ESR_rust` - An implementation of Expected Shortage Reduction (ESR), with some hot code implemented in Rust for optimisation. The structure of `generate_action()` is the same as with `demand_tie`.
* `lookahead` - An implementation of lookahead policy which combines (ESR) with an optimisation step. There is less actions here, as we only need if there's a transhipment and the warehouse echelon base-stock level.
  ```python 
  generate_action(
    warehouse_order=int,
    transhipment=[True, False]
  )
  ```
* <span style="color:red">!Deprecated</span> `ESR` - A pure python implementation of the expected shortage reduction. Not intended to be used. And probably doesn't even work. Kept for clarity if you want to know how it works but not have to read Rust code.

### Example
```python
# Test the dfw model w/ Transhipments works: namely ESR
import gymnasium as gym
import inventorygyms
import inventorygyms.wrappers.transhipment.ESR_rust as ESR
import numpy as np

params = {
    'periods': 10, 
    'stores': 4,
    'lead_time': [1,1,0],
    'ts_cost': 1,
    'penalty': 9,
    'initial_inventory': [[10,10],[12,0],[7,0],[6,0], [9,0]],
    'online_demand_params': [0 for i in range(10)],
    'store_demand_params': [[(6,0.375) for i in range(10)] for t in range(4)],
    'dfw_chance': 0.8,
    'demand_distribution': ['Poisson'] + ['Negative Binomial' for i in range(4)],
}
out = {'warehouse': 100, 'store': [10,10,7,8], 'r': [6,7,8,9]}
env_2 = gym.make('inventorygyms/TwoEchelonPLSTS-v0', **params)

trans_env = ESR.ts_ESR(env_2)

terminated = False
cost = []

while not terminated:
    print('Period: {}'.format(j))
    j+=1
    # Generate action
    action = trans_env.generate_action('Capped', out, True)

    # Take a step in time.
    observation, reward, terminated, truncated, info = la_env.step(action)
```    
Outputs
====
Notice in the example, we save 4 variables. These are default for `gymnasium` environment but in general they provide:
* `observation`: Returns the state.
* `reward`: Return the period reward, note because we normally like to maximise in RL, this is returned as a negative value so to minimise costs take the negative of this output when saving
* `terminated`: if we've finished with the simulation, this becomes `true` once we exhausted number of periods
* `info`: All the information you need to know about the DES, in a dictionary format. Reccomend to addpend this to a list each time then wrap that as a [pandas](https://pandas.pydata.org/) dataframe. 

Misc.
======
### Distributions Supported
**Poisson:** $\lambda \leq 28$

**Negative Binomial:** If $x\sim NB(r,p)$, then, $\mathbb{P}[x\geq 50]\leq 0.001$

**Binomial:** $n \leq 50$

### To-dos
- [ ] Allow ringfencing for online demand
- [ ] Add more distributions to support
- [ ] Give example of using DeepRL for the problem
- [ ] Comprehensive documentation