import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha = 0.33):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}
        self.alpha = 0
        self.success = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def max_action_for_state(self, state):
        max_action = None
        max_value = float("-inf")
        actions = [None, 'forward', 'left', 'right']
        random.shuffle(actions)
        for action in actions:
            if (state, action) not in self.Q:
                self.Q[(state, action)] = 0
            if self.Q[(state, action)] > max_value:
                max_value = self.Q[(state, action)]
                max_action = action
        return max_action

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (('light', inputs['light']),
                      ('oncoming', inputs['oncoming']),
                      ('right', inputs['right']),
                      ('left', inputs['left']),
                      ('next_waypoint', self.next_waypoint))

        # TODO: Select action according to your policy
        actions = ["forward", "right",  "left", None]
        action = random.choice(actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        if (self.state, action) not in self.Q:
            self.Q[(self.state, action)] = 0
        self.Q[(self.state, action)] = (1 - self.alpha) * self.Q[(self.state, action)] \
                                       + self.alpha * reward

        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]
        if location == destination:
            self.success = self.success + 1
            

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, \
                                                                                                    inputs, \
                                                                                                    action, \
                                                                                                    reward)  # [debug]
        print "Location = {}, Destination = {}, Success = {}".format(location, destination, location == destination)                                                                                                   
        print "Alpha = {}, Total success = {}".format(self.alpha, self.success)        

def run(alpha):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    n_trials = 100
    sim.run(n_trials)  # press Esc or close pygame window to quit

    return float(a.success) / n_trials


if __name__ == '__main__':
    # alphas = np.arange(0, 1.1, 0.1)
    alphas = [0.0]
    couples = []
    for alpha in alphas:
        success = run(alpha)
        couples.append([alpha, success])
        print "Current alpha: {}, success={}".format(alpha, success)
    print ""
    print "Alpha vs. success table:"
    print np.array(couples)
    plt.plot(np.array(couples)[:,0], np.array(couples)[:,1], 'ro')
    plt.xlabel('alphas')
    plt.ylabel('success')
    plt.show()
