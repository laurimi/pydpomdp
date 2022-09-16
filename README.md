# PyDPOMDP
A minimal package providing a Python API to decentralized partially observable Markov process (Dec-POMDP) problems in the `.dpomdp` file format.

## Installation
```
git clone git@github.com:laurimi/pydpomdp.git
pip install pydpomdp
```

## Examples
Before starting, download a problem file, for example: 
```
wget https://www.st.ewi.tudelft.nl/mtjspaan/decpomdp/dectiger.dpomdp
```

The following script shows the functionality of this package.
```python
from pydpomdp import DecPOMDP
d = DecPOMDP("dectiger.dpomdp")

# Basic properties
d.num_agents()
d.num_joint_actions()
d.num_joint_observations()
d.num_states()
d.discount()

# Initial belief state
d.initial_belief_at(state=0)

# Individual agent/state/action/observation properties
d.num_actions(agent=0)
d.num_observations(agent=1)

d.agent_name(agent=1)
d.state_name(state=0)
d.action_name(agent=0, action=1)
d.observation_name(agent=1, observation=1)

# Converting between joint and individual action/observation indices
individual_actions = [0, 1]
ja = d.individual_to_joint_action_indices(individual_actions)
d.joint_to_individual_action_indices(ja) # equals individual_actions

joint_observation = 3
individual_observations = d.joint_to_individual_observation_indices(joint_observation)
d.individual_to_joint_observation_indices(individual_observations) # equals joint_observation

# Rewards, state transition and observation probabilities.
# s is current state, s' is new state after taking joint action a, 
# z' is joint observation received in new state
state = 0
joint_action = 1
new_state = 1

d.reward(state, joint_action) # immediate reward

d.transition_probability(new_state, state, joint_action) # P(s' | s, a)

joint_observation = 0
d.observation_probability(joint_observation, new_state, joint_action) # P(z' | s', a)

# Sampling new states and observations
import random
new_state = d.sample_next_state(state, joint_action, random.random())
d.sample_observation(new_state, joint_action, random.random())
```


## Notes
* This package does not provide problem files such as `dectiger.dpomdp`. Many classic Dec-POMDPs from the research literature may be found [here](http://masplan.org/problem_domains).
* This package works for "flat" Dec-POMDPs, and support for factored or transition/observation independent Dec-POMDPs is not implemented so far. If you require a parser for these problems, refer to the [MADP toolbox](https://github.com/MADPToolbox/MADP).
* Indices for states and (joint) actions and (joint) observations are `int` on the Python side, and `unsigned int` in C++ - you may have issues with overflows with very large Dec-POMDPs


## Acknowledgments
This project depends on the parser from the [MADP toolbox](https://github.com/MADPToolbox/MADP), which is gratefully acknowledged.