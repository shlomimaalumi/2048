import math
import random
from pprint import pprint
from typing import Tuple, Optional, Any, Union
from pprint import pprint
import numpy as np
from enum import Enum
import abc
import game_state
import util
from game import Agent, Action


class AgentType(Enum):
    Player = 0
    Opponent = 1


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        legal_moves = game_state.get_agent_legal_actions()
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        return legal_moves[random.choice([index for index in range(len(scores)) if scores[index] == best_score])]

    @staticmethod
    def evaluation_function(current_game_state, action):
        """
        Evaluate the given game state based on various factors and return a score.

        Args:
            current_game_state (GameState): The current game state.
            action (int): The action to be taken.

        Returns:
            float: The evaluation score of the game state.
        """
        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        board_size = len(board) * len(board[0])

        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        free_tiles = successor_game_state.get_empty_tiles()
        free_tiles = len(free_tiles[0])
        taken_tiles = board_size - free_tiles
        return 0.8 * (score / taken_tiles) + score + max_tile + 0.8 * free_tiles ** 2 + len(
            successor_game_state.get_legal_actions(0))


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state: game_state.GameState):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        val = self.minmax(game_state, self.depth, AgentType.Player)
        return val[0]

    def minmax(self, game_state: game_state.GameState, depth, agent: AgentType) -> Tuple[Optional[Action], float]:
        """
        Returns the maximum value for the current state
        """
        legal_moves = game_state.get_agent_legal_actions() if agent == AgentType.Player \
            else game_state.get_opponent_legal_actions()
        if depth == 0 or not legal_moves:
            return Action.STOP, self.evaluation_function(game_state)

        if agent == AgentType.Player:  # maximize
            return self.maximize(depth, game_state, legal_moves)

        else:  # minimize
            return self.minimize(depth, game_state, legal_moves)

    def maximize(self, depth, game_state, legal_moves):
        value = float('-inf')
        best_action = Action.STOP
        for action in legal_moves:
            successor_state = game_state.generate_successor(agent_index=AgentType.Player.value, action=action)
            _, successor_value = self.minmax(successor_state, depth - 1, AgentType.Opponent)
            if successor_value > value:
                value = successor_value
                best_action = action
        return best_action, value

    def minimize(self, depth, game_state, legal_moves):
        value = float('inf')
        best_action = Action.STOP
        for action in legal_moves:
            successor_state = game_state.generate_successor(agent_index=AgentType.Opponent.value, action=action)
            _, successor_value = self.minmax(successor_state, depth, AgentType.Player)
            if successor_value < value:
                value = successor_value
                best_action = action
        return best_action, value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action, val = self.alpha_beta_pruning(game_state, self.depth, AgentType.Player)
        return action

    def alpha_beta_pruning(self, game_state, depth, agent, alpha=float("-inf"), beta=float("inf")):
        """
        Returns the maximum value for the current state
        """
        legal_moves = game_state.get_agent_legal_actions() if agent == AgentType.Player \
            else game_state.get_opponent_legal_actions()
        if depth == 0 or not legal_moves:
            return Action.STOP, self.evaluation_function(game_state)

        if agent == AgentType.Player:  # maximize
            return self.maximize_alpha_beta(depth, game_state, legal_moves, alpha, beta)

        else:  # minimize
            return self.minimize_alpha_beta(depth, game_state, legal_moves, alpha, beta)

    def maximize_alpha_beta(self, depth, game_state, legal_moves, alpha, beta):
        value = float('-inf')
        best_action = Action.STOP
        new_alpha = alpha
        for action in legal_moves:
            successor_state = game_state.generate_successor(agent_index=AgentType.Player.value, action=action)
            _, successor_value = self.alpha_beta_pruning(successor_state, depth - 1, AgentType.Opponent, new_alpha, beta)
            if successor_value > value:
                value = successor_value
                best_action = action
            new_alpha = max(new_alpha, value)
            if new_alpha >= beta:
                break
        return best_action, value

    def minimize_alpha_beta(self, depth, game_state, legal_moves, alpha, beta):
        value = float('inf')
        best_action = Action.STOP
        new_beta = beta
        for action in legal_moves:
            successor_state = game_state.generate_successor(agent_index=AgentType.Opponent.value, action=action)
            _, successor_value = self.alpha_beta_pruning(successor_state, depth, AgentType.Player, alpha, new_beta)
            if successor_value < value:
                value = successor_value
                best_action = action
            new_beta = min(new_beta, value)
            if new_beta <= alpha:
                break
        return best_action, value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        action, _ = self.expectimax(game_state, self.depth, AgentType.Player)
        return action

    def expectimax(self, game_state: game_state.GameState, depth, agent: AgentType) -> Tuple[Optional[Action], float]:
        """
        Returns the maximum value for the current state
        """
        legal_moves = game_state.get_agent_legal_actions() if agent == AgentType.Player \
            else game_state.get_opponent_legal_actions()
        if depth == 0 or not legal_moves:
            return Action.STOP, self.evaluation_function(game_state)

        if agent == AgentType.Player:  # maximize
            return self.maximize(depth, game_state, legal_moves)

        else:  # minimize
            return self.expectation(depth, game_state, legal_moves)

    def maximize(self, depth, game_state, legal_moves):
        value = float('-inf')
        best_action = Action.STOP
        for action in legal_moves:
            successor_state = game_state.generate_successor(agent_index=AgentType.Player.value, action=action)
            _, successor_value = self.expectimax(successor_state, depth - 1, AgentType.Opponent)
            if successor_value > value:
                value = successor_value
                best_action = action
        return best_action, value

    def expectation(self, depth, game_state, legal_moves):
        value = 0
        prob = 1 / len(legal_moves)  # uniform distribution
        for action in legal_moves:
            successor_state = game_state.generate_successor(agent_index=AgentType.Opponent.value, action=action)
            _, successor_value = self.expectimax(successor_state, depth, AgentType.Player)
            value += prob * successor_value
        return Action.STOP, value


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: 
    This evaluation function aims to balance multiple aspects of the game state to provide a more
    comprehensive evaluation. It considers the following factors:
    1. The score of the game state.
    2. The number of empty tiles available.
    3. The maximum tile on the board.
    4. The smoothness of the board (how similar adjacent tiles are).
    5. The monotonicity of the board (whether values are consistently increasing or decreasing).
    6. The clustering of tiles (penalizes isolated high-value tiles).

    Args:
        current_game_state (GameState): The current game state.

    Returns:
        float: The evaluation score of the game state.
    """
    board = current_game_state.board
    score = current_game_state.score
    max_tile = current_game_state.max_tile
    free_tiles = len(current_game_state.get_empty_tiles()[0])

    def smoothness(board):
        smoothness_score = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != 0:
                    value = math.log(board[i][j], 2)
                    # Check adjacent cells
                    for direction in [(0, 1), (1, 0)]:
                        x, y = i + direction[0], j + direction[1]
                        if x < len(board) and y < len(board[i]) and board[x][y] != 0:
                            smoothness_score -= abs(value - math.log(board[x][y], 2))
        return smoothness_score

    def monotonicity(board):
        monotonicity_score = 0
        for i in range(len(board)):
            for j in range(1, len(board[i])):
                if board[i][j] > board[i][j - 1]:
                    monotonicity_score += 1
                else:
                    monotonicity_score -= 1

        for j in range(len(board[0])):
            for i in range(1, len(board)):
                if board[i][j] > board[i - 1][j]:
                    monotonicity_score += 1
                else:
                    monotonicity_score -= 1

        return monotonicity_score

    def clustering(board):
        clustering_score = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != 0:
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        x, y = i + direction[0], j + direction[1]
                        if 0 <= x < len(board) and 0 <= y < len(board[i]):
                            clustering_score += abs(board[i][j] - board[x][y])
        return clustering_score

    smoothness_score = smoothness(board)
    monotonicity_score = monotonicity(board)
    clustering_score = clustering(board)

    # Combine all the factors into the final evaluation score
    evaluation_score = (score +
                        max_tile +
                        free_tiles * 2 +
                        smoothness_score * 0.1 +
                        monotonicity_score * 1.5 -
                        clustering_score * 0.5)

    return evaluation_score


# Abbreviation
better = better_evaluation_function
