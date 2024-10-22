from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    def getAction(self, gameState: GameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
        score = 0
        foods = 0
        for i in range(newFood.width):
            for j in range(newFood.height):
                if(newFood[i][j] == True):
                    foods += 1
                    if(abs(i - newPos[0]) + abs(j - newPos[1]) == 1):
                        score += 100
                    else:
                        score += (100 / (abs(i - newPos[0]) + abs(j - newPos[1]))) - (3 * (abs(i - newPos[0]) + abs(j - newPos[1])))

        score -= 1000 * foods

        for ghost in newGhostStates:
            #print(ghost.getPosition()[0])
            if(abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1]) <= 3):
                score -= 100000 * (4 - abs(ghost.getPosition()[0] - newPos[0]) - abs(ghost.getPosition()[1] - newPos[1]))
            else:
                score += 1 * abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1])
        return score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    
    def getAction(self, gameState: GameState):
        
        return self.minimax(gameState, 1, self.depth, 0)[1]

        util.raiseNotDefined()

    def minimax(self, gameState: GameState, currentDepth, targetDepth, agentIndex):
        if(gameState.isWin()):
            return (self.evaluationFunction(gameState),"STOP")
        if(gameState.isLose()):
            return (self.evaluationFunction(gameState), "STOP")

        if(currentDepth == targetDepth and agentIndex == gameState.getNumAgents() - 1):
            legalMoves = gameState.getLegalActions(agentIndex)
            gameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
            scores = [self.evaluationFunction(newGameState) for newGameState in gameStates]
            worstScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == worstScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (worstScore, legalMoves[chosenIndex])
        
        if(agentIndex == 0):
            legalMoves = gameState.getLegalActions(agentIndex)
            gameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
            scores = [self.minimax(newGameState, currentDepth, targetDepth, agentIndex + 1)[0] for newGameState in gameStates]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (bestScore, legalMoves[chosenIndex])
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            gameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
            if(agentIndex == gameState.getNumAgents() - 1):
                scores = [self.minimax(newGameState, currentDepth + 1, targetDepth, 0)[0] for newGameState in gameStates]
            else:
                scores = [self.minimax(newGameState, currentDepth, targetDepth, agentIndex + 1)[0] for newGameState in gameStates]
            worstScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == worstScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (worstScore, legalMoves[chosenIndex])
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    
    def getAction(self, gameState: GameState):
        return self.minimax(gameState, 1, self.depth, 0, -math.inf, math.inf)[1]

        util.raiseNotDefined()

    def minimax(self, gameState: GameState, currentDepth, targetDepth, agentIndex, alpha, beta):
        if(gameState.isWin()):
            return (self.evaluationFunction(gameState), "STOP")
        if(gameState.isLose()):
            return (self.evaluationFunction(gameState), "STOP")

        if(currentDepth == targetDepth and agentIndex == gameState.getNumAgents() - 1):
            legalMoves = gameState.getLegalActions(agentIndex)
            score = math.inf
            for action in legalMoves:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                if(self.evaluationFunction(newGameState) < score):
                    score = self.evaluationFunction(newGameState)
                    bestAction = action
                if(score < alpha):
                    return(score, action)
                beta = min(beta, score)
            return (score, bestAction)
        
        if(agentIndex == 0):
            legalMoves = gameState.getLegalActions(agentIndex)
            score = -math.inf
            for action in legalMoves:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                if(self.minimax(newGameState, currentDepth, targetDepth, agentIndex + 1, alpha, beta)[0] > score):
                    score = self.minimax(newGameState, currentDepth, targetDepth, agentIndex + 1, alpha, beta)[0]
                    bestAction = action
                if score > beta:
                    return(score, action)
                alpha = max(alpha, score)
            return(score, bestAction)      
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            score = math.inf
            for action in legalMoves:
                newGameState = gameState.generateSuccessor(agentIndex, action)
                if(agentIndex == gameState.getNumAgents() - 1):
                    if(self.minimax(newGameState, currentDepth + 1, targetDepth, 0, alpha, beta)[0] < score):
                        score = self.minimax(newGameState, currentDepth + 1, targetDepth, 0, alpha, beta)[0]
                        bestAction = action
                else:
                    if(self.minimax(newGameState, currentDepth, targetDepth, agentIndex + 1, alpha, beta)[0] < score):
                        score = self.minimax(newGameState, currentDepth, targetDepth, agentIndex + 1, alpha, beta)[0]
                        bestAction = action
                if(score < alpha):
                    return(score, action)
                beta = min(beta, score)
            return (score, bestAction)
        
        


class ExpectimaxAgent(MultiAgentSearchAgent):
    
    def getAction(self, gameState: GameState):
        
        return self.expectimax(gameState, 1, self.depth, 0)[1]

        util.raiseNotDefined()

    def expectimax(self, gameState: GameState, currentDepth, targetDepth, agentIndex):
        if(gameState.isWin()):
            return (self.evaluationFunction(gameState),"STOP")
        if(gameState.isLose()):
            return (self.evaluationFunction(gameState), "STOP")

        if(currentDepth == targetDepth and agentIndex == gameState.getNumAgents() - 1):
            legalMoves = gameState.getLegalActions(agentIndex)
            gameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
            scores = [self.evaluationFunction(newGameState) for newGameState in gameStates]
            return (sum(scores) / len(scores), "STOP")
        
        if(agentIndex == 0):
            legalMoves = gameState.getLegalActions(agentIndex)
            gameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
            scores = [self.expectimax(newGameState, currentDepth, targetDepth, agentIndex + 1)[0] for newGameState in gameStates]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (bestScore, legalMoves[chosenIndex])
        else:
            legalMoves = gameState.getLegalActions(agentIndex)
            gameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
            if(agentIndex == gameState.getNumAgents() - 1):
                scores = [self.expectimax(newGameState, currentDepth + 1, targetDepth, 0)[0] for newGameState in gameStates]
            else:
                scores = [self.expectimax(newGameState, currentDepth, targetDepth, agentIndex + 1)[0] for newGameState in gameStates]
            return (sum(scores) / len(scores), "STOP")

def betterEvaluationFunction(currentGameState: GameState):
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()



    score = 0
    foods = 0
    for i in range(newFood.width):
        for j in range(newFood.height):
            if(newFood[i][j] == True):
                foods += 1
                if(abs(i - newPos[0]) + abs(j - newPos[1]) == 1):
                    score += 10
                else:
                    score += (100 / (abs(i - newPos[0]) + abs(j - newPos[1]))) - (3 * (abs(i - newPos[0]) + abs(j - newPos[1])))

    score -= 1000 * foods
    for posCapsuls in newCapsules:
        score += (500 / (abs(posCapsuls[0] - newPos[0]) + abs(posCapsuls[1] - newPos[1]))) - (3 * (abs(posCapsuls[0] - newPos[0]) + abs(posCapsuls[1] - newPos[1])))
        score -= 1000

    for ghost in newGhostStates:
        if(ghost.scaredTimer <= 30 and ghost.scaredTimer >= 5):
            score -= 10000 * abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1])
        elif(abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1]) <= 4):
            score -= 100000 * (5 - abs(ghost.getPosition()[0] - newPos[0]) - abs(ghost.getPosition()[1] - newPos[1]))
        else:
            score += abs(ghost.getPosition()[0] - newPos[0]) + abs(ghost.getPosition()[1] - newPos[1])
   
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
