import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

random.seed(42)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha = 0.33, epsilon = 0.1):
        # sets self.env = env, state = None, next_waypoint = None,
        # and a default color
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # TODO: Initialize any additional variables here
        self.possible_actions = [None, 'forward', 'left', 'right']
        self.Q = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.success_table = []
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def max_action_for_state(self, state):
        max_action = None
        max_value = float("-inf")
        random.shuffle(self.possible_actions)
        for action in self.possible_actions:
            if (state, action) not in self.Q:
                self.Q[(state, action)] = 0
            if self.Q[(state, action)] > max_value:
                max_value = self.Q[(state, action)]
                max_action = action
        return max_action

    def update(self, t):
        # Gather inputs from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (('light', inputs['light']),
                      ('oncoming', inputs['oncoming']),
                      ('right', inputs['right']),
                      ('left', inputs['left']),
                      ('next_waypoint', self.next_waypoint))

        # TODO: Select action according to your policy
        # Choose action with epsilon-greedy exploration, i.e. pick best action
        # with probability of (1 - epsilon) or random action with probability of
        # epsilon
        if random.randint(0, int(1. / self.epsilon)) == 0:
            action = random.choice(self.possible_actions)
        else:
            action = self.max_action_for_state(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # TODO: Learn policy based on state, action, reward
        # If state not explored yet initialize Q-value = 10
        # Purpose: try optimism in the face of uncertainty so that agent leans
        # towards exploration rather than exploitation at the beginning
        if (self.state, action) not in self.Q:
            self.Q[(self.state, action)] = 10

        # If state explored already, compute Q-value via Bellman equation
        self.Q[(self.state, action)] = (1 - self.alpha) * self.Q[(self.state,
                                                                  action)] \
                                       + self.alpha * reward

        # Compare destination to current location
        # If location == destination, then it means success
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]
        is_success = location == destination
        
        self.success_table.append([self.alpha,
                                   self.epsilon,
                                   self.env.n_trial,
                                   is_success,
                                   deadline])
            
        print "LearningAgent.update(): deadline = {}, inputs = {},"\
              "action = {}, reward = {}".format(deadline,
                                                inputs,
                                                action,
                                                reward)  # [debug]
        print "Location = {}, Destination = {}, Success = {}"\
              .format(location, destination, is_success)                                                                                                   
        print "Alpha = {}, Epsilon = {}, Trial number = {}"\
              .format(self.alpha, self.epsilon, self.env.n_trial)

        # Generate log to study agent's behaviour after simulation is completed
        with open("q.log", "a") as q_log:
            q_log.write("Alpha: {}, Epsilon: {}, Trial: {}, Deadline: {}\n"\
                        .format(self.alpha,
                                self.epsilon,
                                self.env.n_trial,
                                deadline))
            q_log.write("Current trial success: {}\n".format(is_success))
            q_log.write("State: {}\n".format(self.state))
            q_log.write("Action: {}, Next waypoint: {}, Reward: {},"\
                        "Q-value: {}\n".format(action,
                                               self.next_waypoint,
                                               reward,
                                               self.Q[(self.state, action)]))
            q_log.write("Q(state, action) = (1 - {}) * {} + {} * {} = {} + {}"\
                        "= {}\n\n".format(self.alpha,
                                          self.Q[(self.state, action)],
                                          self.alpha,
                                          reward,
                                          (1 - self.alpha) * self.Q[(self.state,
                                                                     action)],
                                          self.alpha * reward,
                                          (1 - self.alpha) * self.Q[(self.state,
                                                                     action)]
                                          + self.alpha * reward))



def run(alphas, epsilons, n_trials):
    """
    Run the agent for a finite number of trials.
    alphas is meant to be a list of different learning rates, as many as you
    would like to test.
    n_trials is the number of trials you want the agent to perform for each
    different learning rate.
    """

    successes = np.zeros((len(epsilons), len(alphas)))
    
    for j, epsilon in enumerate(epsilons):
        for i, alpha in enumerate(alphas):

            # Set up environment and agent
            # create environment (also adds some dummy traffic)
            e = Environment()
            a = e.create_agent(LearningAgent, alpha, epsilon)  # create agent
            e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
            # Now simulate it
            # reduce update_delay to speed up simulation
            sim = Simulator(e, update_delay=0)
            sim.run(n_trials)  # press Esc or close pygame window to quit
            # Convert self.success_table to array to process it with ease
            a.success_table = np.array(a.success_table)
            
            # for debugging purposes
            # with open("success_table.log", "a") as success_table:
            #    success_table.write("{}\n".format(a.success_table))

            # Subset end row of each trial,
            # i.e. rows where deadline = 0 or success = 1
            is_success = a.success_table[:, 3] == 1
            is_deadline_reached = a.success_table[:, 4] == 0
            a.end_of_trial_table = a.success_table[(is_success |
                                                    is_deadline_reached), :]

            # for debugging purposes
            # with open("end_of_trial_table.log", "a") as end_of_trial_table:
            #    end_of_trial_table.write("{}\n".format(a.end_of_trial_table))
            
            trial_success = np.sum(a.end_of_trial_table[:, 3])
            successes[j][i] = trial_success

    best_result = np.max(successes)
    is_best_result = successes == best_result
    best_alpha = alphas[is_best_result][0]
    best_epsilon = epsilons[is_best_result][0]
    best_success_rate = best_result / n_trials
    avg_success_rate = np.mean(successes) / n_trials

    print("\n")
    print("Best alpha: {}".format(best_alpha))
    print("Best epsilon: {}".format(best_epsilon))
    print("Number of times agent reaches destination with best parameters: {}"\
          .format(best_result))
    print("Best success rate: {}%".format(best_success_rate * 100))
    print("Average success rate: {}%".format(avg_success_rate * 100))
    
            
    return successes

 
if __name__ == '__main__':

    # set parameters
    alphas = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90])
    epsilons = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    n_trials = 100

    # launch simulation
    successes = run(alphas, epsilons, n_trials)

    # plot surface to visualize optimal parameter set
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    alphas, epsilons = np.meshgrid(alphas, epsilons)
    surf = ax.plot_surface(alphas, epsilons, successes, rstride=1, cstride=1,
                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.xlabel('alphas')
    plt.ylabel('epsilons')
    plt.show()
