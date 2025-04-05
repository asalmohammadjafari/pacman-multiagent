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

from util import manhattanDistance
from game import Directions
import random, util, math
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """
    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) 
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        The evaluation function takes in the current and proposed successor GameStates
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = 0
        foods = sum(1 for i in range(newFood.width) for j in range(newFood.height) if newFood[i][j])
        
        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    dist = abs(i - newPos[0]) + abs(j - newPos[1])
                    score += (100 / dist) - (3 * dist) if dist > 1 else 100
        
        score -= 1000 * foods
        
        for ghost in newGhostStates:
            dist = abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1])
            if dist <= 3:
                score -= 100000 * (4 - dist)
            else:
                score += dist
        
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """ This default evaluation function just returns the score of the state. """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides common elements to all multi-agent searchers.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='3'):
        self.index = 0 
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """ Your minimax agent """
    def getAction(self, gameState: GameState):
        return self.minimax(gameState, 1, self.depth, 0)[1]

    def minimax(self, gameState: GameState, currentDepth, targetDepth, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"
        
        if currentDepth == targetDepth and agentIndex == gameState.getNumAgents() - 1:
            legalMoves = gameState.getLegalActions(agentIndex)
            scores = [self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)) for action in legalMoves]
            return min(scores), random.choice(legalMoves)
        
        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            scores = [self.minimax(gameState.generateSuccessor(agentIndex, action), currentDepth, targetDepth, agentIndex + 1)[0] for action in legalMoves]
            return max(scores), random.choice(legalMoves)
        else:
            scores = [self.minimax(gameState.generateSuccessor(agentIndex, action), currentDepth + (agentIndex == gameState.getNumAgents() - 1), targetDepth, (agentIndex + 1) % gameState.getNumAgents())[0] for action in legalMoves]
            return min(scores), random.choice(legalMoves)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """ Your alpha-beta pruning agent """
    def getAction(self, gameState: GameState):
        return self.alphabeta(gameState, 1, self.depth, 0, -math.inf, math.inf)[1]

    def alphabeta(self, gameState: GameState, currentDepth, targetDepth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"
        
        if currentDepth == targetDepth and agentIndex == gameState.getNumAgents() - 1:
            legalMoves = gameState.getLegalActions(agentIndex)
            scores = [self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)) for action in legalMoves]
            return min(scores), random.choice(legalMoves)
        
        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            bestScore = -math.inf
            bestMove = None
            for action in legalMoves:
                score = self.alphabeta(gameState.generateSuccessor(agentIndex, action), currentDepth, targetDepth, agentIndex + 1, alpha, beta)[0]
                if score > bestScore:
                    bestScore = score
                    bestMove = action
                alpha = max(alpha, bestScore)
                if beta <= alpha:
                    break
            return bestScore, bestMove
        else:
            bestScore = math.inf
            bestMove = None
            for action in legalMoves:
                score = self.alphabeta(gameState.generateSuccessor(agentIndex, action), currentDepth + (agentIndex == gameState.getNumAgents() - 1), targetDepth, (agentIndex + 1) % gameState.getNumAgents(), alpha, beta)[0]
                if score < bestScore:
                    bestScore = score
                    bestMove = action
                beta = min(beta, bestScore)
                if beta <= alpha:
                    break
            return bestScore, bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """ Your expectimax agent """
    def getAction(self, gameState: GameState):
        return self.expectimax(gameState, 1, self.depth, 0)[1]

    def expectimax(self, gameState: GameState, currentDepth, targetDepth, agentIndex):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"
        
        if currentDepth == targetDepth and agentIndex == gameState.getNumAgents() - 1:
            legalMoves = gameState.getLegalActions(agentIndex)
            scores = [self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)) for action in legalMoves]
            return sum(scores) / len(scores), random.choice(legalMoves)
        
        legalMoves = gameState.getLegalActions(agentIndex)
        if agentIndex == 0:
            scores = [self.expectimax(gameState.generateSuccessor(agentIndex, action), currentDepth, targetDepth, agentIndex + 1)[0] for action in legalMoves]
            return max(scores), random.choice(legalMoves)
        else:
            scores = [self.expectimax(gameState.generateSuccessor(agentIndex, action), currentDepth + (agentIndex == gameState.getNumAgents() - 1), targetDepth, (agentIndex + 1) % gameState.getNumAgents())[0] for action in legalMoves]
            return sum(scores) / len(scores), random.choice(legalMoves)

def betterEvaluationFunction(currentGameState: GameState):
    """
    A better evaluation function that considers both food and ghosts more comprehensively.
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    
    # Calculate food distance
    foodDistances = [manhattanDistance(pacmanPosition, foodPosition) for foodPosition in food.asList()]
    foodScore = -sum(foodDistances) if foodDistances else 0  # Sum of distances to all food
    
    # Ghost evaluation (Pacman should avoid ghosts)
    ghostScores = 0
    for ghost, ghostDistance in zip(ghosts, [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghosts]):
        if ghost.scaredTimer > 0:
            # Bonus for being near scared ghosts (Pacman can eat them)
            ghostScores += 100 / (ghostDistance + 1) if ghostDistance <= 3 else 0
        else:
            # Penalize for being too close to non-scared ghosts
            ghostScores -= 1000 / (ghostDistance + 1) if ghostDistance <= 3 else 0
    
    # Combine the scores
    return currentGameState.getScore() + foodScore + ghostScores

