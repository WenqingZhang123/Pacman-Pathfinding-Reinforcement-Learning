# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    visited_node_list = []

    # Define a function to implement DFS using recursion
    def dfs_exloper(state, actions):
        # Judge if the state has been visited
        if state in visited_node_list:
            return None
        # If this node is first visited, add it to the visited list
        visited_node_list.append(state)
        # Judge if the state is the goal state
        if problem.isGoalState(state):
            return actions
        # Get the successors of the state and iterate over them
        for next_state, action, action_cost in problem.getSuccessors(state):
            # imitate the process of DFS, which is a recursive process
            result = dfs_exloper(next_state, actions + [action])
            # If the result is not None, it means that the goal state has been found
            if result is not None:
                return result
        # If the goal state is not found, return None
        return None

    # Start the DFS process
    return dfs_exloper(problem.getStartState(), [])


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # Store the visited nodes
    visited_node_list = []
    # Store the nodes to be expanded
    to_expand_nodes = util.Queue()
    # Get the start state and put it into the queue
    startState = problem.getStartState()
    to_expand_nodes.push((startState, []))  # 将初始状态和空操作序列放入队列中
    # Start the BFS process
    while True:
        # If the queue is empty, return an empty action sequence
        if to_expand_nodes.isEmpty():
            return []
        # Pop the first node in the queue
        state, actions = to_expand_nodes.pop()
        # If the state is the goal state, return the action sequence
        if problem.isGoalState(state):
            return actions
        # If the state has not been visited, add it to the visited list
        if state not in visited_node_list:
            visited_node_list.append(state)
            successors = problem.getSuccessors(state)
            # Iterate over the successors of the state
            for successor, action, action_cost in successors:
                new_actions = actions + [action]
                to_expand_nodes.push((successor, new_actions))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Initialize the priority queue with the start state and a cost of 0
    start_state = problem.getStartState()
    # Initialize a priority queue to store the nodes to be expanded
    to_expand_nodes = util.PriorityQueue()
    # （state, actions, cost）
    to_expand_nodes.push((start_state, [], 0), 0)
    # Initialize a set to keep track of visited states
    visited_node_set = set()
    while True:
        if to_expand_nodes.isEmpty():
            return []
        # Pop the node with the lowest cost
        state, actions, cost = to_expand_nodes.pop()
        # If the state is the goal state, return the action sequence
        if problem.isGoalState(state):
            return actions
        # If the state has not been visited, add it to the visited list
        if state not in visited_node_set:
            visited_node_set.add(state)
            successors = problem.getSuccessors(state)
            for successor, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                to_expand_nodes.push((successor, new_actions, new_cost), new_cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # To store the nodes to be expanded
    to_expand_nodes = util.PriorityQueue()
    start_state = problem.getStartState()
    # To store the visited nodes
    visited_node_set = set()
    # To initialize the priority queue with the start state and a cost of 0
    to_expand_nodes.push((start_state, [], 0), 0 + heuristic(start_state, problem))
    # To start the A* search process
    while True:
        if to_expand_nodes.isEmpty():
            return []
        state, actions, cost = to_expand_nodes.pop()
        if problem.isGoalState(state):
            return actions
        if state not in visited_node_set:
            visited_node_set.add(state)
            # To get the successors of the state and iterate over them
            for successor, action, step_cost in problem.getSuccessors(state):
                new_actions = actions + [action]
                new_cost = cost + step_cost
                priority = new_cost + heuristic(successor, problem)
                to_expand_nodes.push((successor, new_actions, new_cost), priority)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
