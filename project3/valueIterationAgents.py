# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # choose the optimal action for each state with the maxinum value
        def getMaxValueAction(singleState):
            # avoid the terminal state
            if self.mdp.isTerminal(singleState) == False:
                values_allActions = [] # Try to get the optimal action with maximum value later
                # calculate the value of each action using the formula
                for possible_action in self.mdp.getPossibleActions(singleState):
                    value_singleAction = sum(
                        prob_specific_state * (self.mdp.getReward(singleState, possible_action, possible_state) + self.discount * values_modified[possible_state])
                        for possible_state, prob_specific_state in
                        self.mdp.getTransitionStatesAndProbs(singleState, possible_action))
                    values_allActions.append(value_singleAction)
                # update the optimal action to the single state
                self.values[singleState] = sorted(values_allActions, reverse=True)[0]
                return singleState

        # iterate specific times to update the values
        i = 0
        while i < self.iterations:
            # Just in case the values are modified during the iteration
            values_modified = self.values.copy()
            # Update the values of all states
            finished_state = [getMaxValueAction(singleState) for singleState in self.mdp.getStates() ]
            i += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # calculate the value of each action using the formula
        return sum(prob_specific_state * (self.mdp.getReward(state, action, possible_state) + self.discount * self.values[possible_state])
            for possible_state, prob_specific_state in self.mdp.getTransitionStatesAndProbs(state, action))


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get the optimal action with maximum value
        return next(iter(sorted(((self.computeQValueFromValues(state, possible_action), possible_action)
                                 for possible_action in self.mdp.getPossibleActions(state)), key=lambda x: x[0], reverse=True)), (None, None))[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # iterate specific times to update the values
        i = 0
        # initialize the relevant states and prepare the asynchronous state
        all_possible_states = self.mdp.getStates()
        denominator_reminder = len(all_possible_states)
        while i < self.iterations:
            asynchronous_state = all_possible_states[i % denominator_reminder]
            if self.mdp.isTerminal(asynchronous_state) == False:
                # update the optimal action to the single state with asynchronous value iteration
                self.values[asynchronous_state] = sorted([self.computeQValueFromValues(asynchronous_state, single_action)
                                 for single_action in self.mdp.getPossibleActions(asynchronous_state)], reverse=True)[0]
            i += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        prepared_status = {}
        # make prepared status for each state
        for possible_status in self.mdp.getStates():
            if self.mdp.isTerminal(possible_status) == False:
                for single_possible_action in self.mdp.getPossibleActions(possible_status):
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(possible_status, single_possible_action):
                        if probability > 0:
                            prepared_status.setdefault(nextState, set()).add(possible_status)
        prepared_queue = util.PriorityQueue()
        # If the absolute value is greater than theta, push p into the priority queue with priority -diff
        for possible_status in self.mdp.getStates():
            push_result = (not self.mdp.isTerminal(possible_status)) and (prepared_queue.push(possible_status, - abs(self.values[possible_status] - sorted([self.computeQValueFromValues(possible_status, single_action) for single_action in self.mdp.getPossibleActions(possible_status)], reverse=True)[0])))

        iteration = 0
        while iteration < self.iterations:
            if prepared_queue.isEmpty():
                break  # Terminate if the priority queue is empty.
            # Pop a state s off the priority queue.
            possible_status = prepared_queue.pop()
            # Try to update the value of s in self.values.
            self.values[possible_status] = sorted([self.computeQValueFromValues(possible_status, action) for action in self.mdp.getPossibleActions(possible_status)], reverse=True)[0] if not self.mdp.isTerminal(possible_status) else self.values[possible_status]
            for possible_s in prepared_status.get(possible_status, []):
                # If the absolute value is greater than theta, push p into the priority queue with priority -diff
                diff = abs(self.values[possible_s] - sorted([self.computeQValueFromValues(possible_s, action) for action in self.mdp.getPossibleActions(possible_s)], reverse=True)[0])
                queue_result = (diff > self.theta) and prepared_queue.update(possible_s, -diff)
            iteration += 1



