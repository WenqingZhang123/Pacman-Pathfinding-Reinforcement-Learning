# multiAgents.py
# --------------
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

# WUSTL 412A: Wenqing Zhang & Tae Won Kim

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Initialize the evaluation score with the successor game state score
        totalScore = successorGameState.getScore()

        # Strategy1：Adjust the score based on the distance to the ghosts and their scared state
        for i in range(len(newGhostStates)):
            ghostState = newGhostStates[i]
            ghostScaredTime = newScaredTimes[i]

            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            # If the ghost is afraid, let pacman take advantage of this opportunity and find the best solution.
            # If ghosts are too close, try your best to avoid encounters with ghosts
            totalScore += ((300 - ghostDistance) if ghostScaredTime > 0 else 0) - (600 if (ghostDistance <= 1) else 0)

        # Strategy2：Adjust the score based on the distance to the closest food
        # allFoodDistance = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min([manhattanDistance(newPos, food) for food in newFood.asList()] ) if len(newFood.asList()) != 0 else 0
        nonZero_minFoodDistance = minFoodDistance + 1  # ensure allFoodDistance is not 0 as the divisor
        # Motivating pacman to find the nearest food
        totalScore += 2 / nonZero_minFoodDistance # The closer the distance, the higher the score

        # Strategy3：Adjust the score based on no movement to avoid pacman doing nothing
        # Prevents pacman stopping
        totalScore -= 12 if (action == Directions.STOP or successorGameState.getPacmanPosition() == currentGameState.getPacmanPosition()) else 0

        return totalScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimaxAlgorithm(agentIndex, depth, gameState):
            #   Judge if the game is over, or if the maximum depth has been reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's decision as Maximizer
            if agentIndex == 0:
                # Get all possible actions of pacman and find the maximum score
                return max(minimaxAlgorithm(1, depth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))
            # Ghost's decision as Minimizer
            if agentIndex >= 1:
                # If the last ghost has been reached, the next agent is pacman, and the depth is increased by 1
                lastGhostAgentNumChecker = [0, depth+1] if (agentIndex + 1) == gameState.getNumAgents() else [(agentIndex + 1),depth]
                # Get all possible actions of ghost and find the minimum score
                return min(minimaxAlgorithm(lastGhostAgentNumChecker[0], lastGhostAgentNumChecker[1], gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

        
        # Initialize the optimal action and score of pacman
        optimal_score_action = [float("-inf"), None]
        # Get all possible actions of pacman
        possible_actions = gameState.getLegalActions(0)
        # Find the action with the largest score among all possible actions of pacman
        for i in range(len(possible_actions)):
            # The new state generated by a potential action of Pacman evaluates the maximum value of the next ghost.
            maximumScore_action = minimaxAlgorithm(1, 0, gameState.generateSuccessor(0, possible_actions[i]))
            # Get the maximum value after the minmax process of this pacman and this action
            optimal_score_action = [maximumScore_action, possible_actions[i]] if maximumScore_action > optimal_score_action[0] else optimal_score_action
        # Return the action with the largest score
        return optimal_score_action[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaSearch(agentIndex, depth, gameState, alpha, beta):
            #   Judge if the game is over, or if the maximum depth has been reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's decision as Maximizer
            if agentIndex == 0:
                # Initialize the maximum value to negative infinity
                v_Maximizer = float("-inf")
                # Iterate through all possible actions of each current agent and find the maximum score
                currentAgent_actions = gameState.getLegalActions(agentIndex)
                for i in range(len(currentAgent_actions)):
                    v_Maximizer = max(v_Maximizer, alphaBetaSearch(1, depth, gameState.generateSuccessor(agentIndex, currentAgent_actions[i]), alpha, beta))
                    # Implement alpha-beta pruning, which only includes cases greater than but not equal to beta
                    if v_Maximizer > beta:
                        return v_Maximizer
                    alpha = max(alpha, v_Maximizer)
                return v_Maximizer
            # Ghost's decision as Minimizer
            if agentIndex >= 1:
                # Initialize the maximum value to positive infinity
                v_Minimizer = float("inf")
                # If the last ghost has been reached, the next agent is pacman, and the depth is increased by 1
                lastGhostAgentNumChecker = [0, depth + 1] if (agentIndex + 1) == gameState.getNumAgents() else [
                    (agentIndex + 1), depth]
                # Iterate through all possible actions of each current agent and find the maximum score
                currentAgent_actions = gameState.getLegalActions(agentIndex)
                for i in range(len(currentAgent_actions)):
                    v_Minimizer = min(v_Minimizer, alphaBetaSearch(lastGhostAgentNumChecker[0], lastGhostAgentNumChecker[1],
                                                                   gameState.generateSuccessor(agentIndex, currentAgent_actions[i]), alpha, beta))
                    # Implement alpha-beta pruning, which only includes cases smaller than but not equal to beta
                    if v_Minimizer < alpha:
                        return v_Minimizer
                    beta = min(beta, v_Minimizer)
                return v_Minimizer

        # Body of getAction starts here
        alpha = float("-inf")
        beta = float("inf")
        # Initialize the optimal action and score of pacman
        optimal_score_action = [float("-inf"), None]
        # Get all possible actions of pacman
        possible_actions = gameState.getLegalActions(0)
        # Find the action with the largest score among all possible actions of pacman
        for i in range(len(possible_actions)):
            # Start from Pacman with agentIndex 0 and depth 0
            # The new state generated by a potential action of Pacman evaluates the maximum value of the next ghost.
            maximumScore_action = alphaBetaSearch(1, 0, gameState.generateSuccessor(0, possible_actions[i]), alpha, beta)
            # Get the maximum value after the minmax process of this pacman and this action
            optimal_score_action = [maximumScore_action, possible_actions[i]] if maximumScore_action > optimal_score_action[0] else optimal_score_action
            # Apply alpha-beta pruning and update alpha values
            alpha = max(alpha, optimal_score_action[0])
        # Return the action with the largest score
        return optimal_score_action[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            #   Judge if the game is over, or if the maximum depth has been reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # Pacman's decision as Maximizer
            if agentIndex == 0:
                return max(expectimax(1, depth, gameState.generateSuccessor(agentIndex, newState)) for newState in
                           gameState.getLegalActions(agentIndex))
            # Ghost's decision as Expecter
            if agentIndex >= 1:
                # If the last ghost has been reached, the next agent is pacman, and the depth is increased by 1
                lastGhostAgentNumChecker = [0, depth + 1] if (agentIndex + 1) == gameState.getNumAgents() else [
                    (agentIndex + 1), depth]
                # Calculate the average value of all possible actions of the current ghost
                cumulative_Sum = sum(expectimax(lastGhostAgentNumChecker[0], lastGhostAgentNumChecker[1], gameState.generateSuccessor(agentIndex, newState)) for newState in
                           gameState.getLegalActions(agentIndex))
                # Calculate the average value of all possible actions of the current ghost
                numOfActions = float(len(gameState.getLegalActions(agentIndex)))
                # cumulative_Sum / numOfActions is the average value of all possible actions of the current ghost
                return cumulative_Sum / numOfActions # It means ghost will choose the action randomly or the same probability.

        # Initialize the optimal action and score of pacman
        optimal_score_action = [float("-inf"), None]
        # Get all possible actions of pacman
        possible_actions = gameState.getLegalActions(0)
        # Find the action with the largest score among all possible actions of pacman
        for i in range(len(possible_actions)):
            # The new state generated by a potential action of Pacman evaluates the expectedMax value of the next ghost.
            expectiMaxScore_action = expectimax(1, 0, gameState.generateSuccessor(0, possible_actions[i]))
            # Get the expectiMax value after the minmax process of this pacman and this action
            optimal_score_action = [expectiMaxScore_action, possible_actions[i]] if expectiMaxScore_action > optimal_score_action[0] else optimal_score_action
        return optimal_score_action[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''
    Food-based choice strategy: the goal is to eat all the food as soon as possible
    '''
    # Find the relationship between pacman and food, using the function provided by getAgent in MinimaxAgent
    pacman_position = currentGameState.getPacmanPosition()
    food_position = currentGameState.getFood()
    # Find the shortest distance between pacman and food
    minFoodDistance = min([manhattanDistance(pacman_position, food) for food in food_position.asList()]) if len(food_position.asList()) != 0 else 0
    # Encourage eating the closest food and eating all the food quickly
    # Score initialization bit current score - target reduces distance - target eats all the food quickly
    totalScore = currentGameState.getScore() - 2 * minFoodDistance -2 * len(food_position.asList())

    # Encourage eating the last few items of food, which avoids pacman from running around or stopping or turning around
    # Add bonus for the last piece of food, which the more last food, the more bonus
    nonZero = minFoodDistance + 1  # ensure minFoodDistance is not 0 as the divisor
    encurageLastFood = {1: 200 / nonZero, 2: 10 / nonZero, 3: 5 / nonZero}
    for i in range(len(encurageLastFood)):
        totalScore += encurageLastFood[i + 1] if len(food_position.asList()) == i + 1 else 0
    return totalScore

# Abbreviation
better = betterEvaluationFunction
