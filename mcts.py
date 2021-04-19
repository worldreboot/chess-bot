import gym
from gym_chess.alphazero import MoveEncoding
import random
import math
import copy
import numpy as np
import tensorflow as tf
import chess
import chess.pgn
import chess.svg
from tensorflow import keras
from tensorflow.keras import layers

class MCTS:
    pieceTypes = {"None" : -1, "P" : 0, "N" : 1, "B": 2, "R" : 3 , "Q" : 4, "K" : 5, "p" : 6, "n" : 7, "b" : 8, "r" : 9, "q" : 10, "k" : 11}
    
    env = gym.make('ChessAlphaZero-v0')
    
    c_puct = 5
    n_thr = 2

    def __init__(self, nnet):
        self.nnet = nnet
        self.N = {}
        self.W = {}
        self.P = {}
        self.Q = {}
        self.moves = []
        self.next_state = 0
        self.next_move = 0
        self.turn = True # white = True, black = False
        self.states = {}
        self.parents = []
        self.children = []
        self.visited = set()
        self.env.reset()
        self.root = 0
        self.MOVES_PLAYED = []
        self.DEPTH = 1
        self.CURRENT_DEPTH = 0
    
    def boardToBitBoard(self, board):
        bitBoard = np.zeros(12 * 8 * 8)
        bitBoard = np.reshape(bitBoard, (12, 8, 8))
        counter = 0
        for i in range(8):
            for j in range(8):
                pieceType = self.pieceTypes[str(board.piece_at(counter))]
                if not pieceType == -1:
                    bitBoard[pieceType][i][j] = 1
                counter += 1
        return bitBoard

    def getBestCompMove(self, board):
        bitBoard = self.boardToBitBoard(board)
        bitBoard = np.asarray(bitBoard)
        bitBoard = bitBoard.reshape(1, 12, 8, 8)
        pp = self.nnet(bitBoard, training=False)
        predictions = pp[0][0].numpy()
        maxMove = np.argmax(predictions)
        while not maxMove in self.env.legal_actions:
            predictions[maxMove] = 0
            maxMove = np.argmax(predictions)
        move = chess.Move.from_uci(str(self.env.decode(maxMove)))
        return move, maxMove

    def convert_board_to_tuple(self, board):
        l = [0 for x in range(64)]
        index = 0
        for i in range(8):
            for j in range(8):
                for piece in range(12):
                    if board[piece][i][j] != 0:
                        l[index] = piece + 1
                index += 1
        return tuple(l)
    
    def add_child(self, parent, action, board):
        self.states[self.next_state] = board.fen() # self.convert_board_to_tuple(self.boardToBitBoard(board))
        self.N[self.next_state] = {}
        self.W[self.next_state] = {}
        self.P[self.next_state] = {}
        self.Q[self.next_state] = {}
        for a in board.legal_moves:
            self.moves.append(a)
            self.N[self.next_state][self.next_move] = 0
            self.W[self.next_state][self.next_move] = 0
            self.P[self.next_state][self.next_move] = 1 # TODO: replace with predicted probability from network
            self.Q[self.next_state][self.next_move] = 0
            self.next_move += 1
        if parent != -1:
            self.parents.append(parent)
            self.children[parent].append((self.next_state, action))
            self.children.append([])
        else:
            # if root
            self.parents.append(parent)
            self.children.append([])
        self.next_state += 1
    
    def already_child(self, s, a):
        for s_a in self.children[s]:
            if s_a[1] == a:
                return True
        return False

    def selection(self, s, board, rollout):
        if board.is_game_over():
            return s, rollout, board

        next_s = 0
        max_a = 0
        max_value = 0
        sq_term = 0
        first = True

        if s not in self.visited or len(self.children[s]) == 0:
            for a in self.N[s]:
                sq_term += self.N[s][a]
            sq_term = math.sqrt(sq_term)

            for a in self.N[s]:
                if first:
                    max_a = a
                    first = False
                value = self.Q[s][a]
                value += self.c_puct * self.P[s][a] * (sq_term / (1 + self.N[s][a]))
                if value >= max_value:
                    max_value = value
                    max_a = a
            rollout.append((s, max_a))
            board.push(self.moves[max_a])
            self.env.step(self.env.encode(self.moves[max_a]))

            if s not in self.visited:
                self.visited.add(s)
            
            return s, rollout, board

        for a in self.children[s]:
            sq_term += self.N[s][a[1]]
        sq_term = math.sqrt(sq_term)

        for a in self.children[s]:
            if first:
                next_s = a[0]
                max_a = a[1]
                first = False
            value = self.Q[s][a[1]]
            value += self.c_puct * self.P[s][a[1]] * (sq_term / (1 + self.N[s][a[1]]))
            if value >= max_value:
                max_value = value
                max_a = a[1]
                next_s = a[0]
        rollout.append((s, max_a))
        board.push(self.moves[max_a])
        self.env.step(self.env.encode(self.moves[max_a]))
        return self.selection(next_s, board, rollout)

    def evaluation(self, board):
        if board.is_game_over():
            z = 0.0
            result = board.result()
            if result == "1-0":
                if self.turn:
                    z = 1.0
                else:
                    z = -1.0
            elif result == "0-1":
                if self.turn:
                    z = -1.0
                else:
                    z = 1.0
            return z
        elif self.CURRENT_DEPTH == self.DEPTH:
            bitBoard = self.boardToBitBoard(board)
            bitBoard = np.asarray(bitBoard)
            bitBoard = bitBoard.reshape(1, 12, 8, 8)
            pp = self.nnet(bitBoard, training=False)
            z = pp[1][0][0].numpy()
            if not self.turn:
                z *= -1

            return z
        move, move_alpha = self.getBestCompMove(board)
        self.env.step(move_alpha)
        board.push(move)
        self.CURRENT_DEPTH += 1
        return self.evaluation(board)
    
    def play_move(self, board, move):
        if len(self.children) == 0:
            self.add_child(-1, -1, board)
        for a in self.N[self.root]:
            s = self.root
            if self.moves[a] != move:
                continue
            max_a = a
            found_next_root = False
            for s_a in self.children[self.root]:
                if s_a[1] == max_a:
                    self.root = s_a[0]
                    found_next_root = True
                    break
            if not found_next_root:
                next_root = self.next_state
                temporary_board = copy.copy(board)
                temporary_board.push(self.moves[max_a])
                self.add_child(self.root, max_a, temporary_board)
                self.root = next_root
                break
        self.MOVES_PLAYED.append(move)

    # mcts: returns action with maximum visit count from current board
    #       state equal to board after total_simulations
    def mcts(self, board, total_simulations, whites_turn):
        # set root
        if len(self.children) == 0:
            self.add_child(-1, -1, board)

        self.turn = whites_turn

        simulation = 0

        while simulation < total_simulations:
            self.env.reset()
            for move in self.MOVES_PLAYED:
                self.env.step(self.env.encode(move))
            temporary_board = copy.copy(board)
            leaf, rollout, leaf_board = self.selection(self.root, temporary_board, [])
            z = self.evaluation(leaf_board)
            self.CURRENT_DEPTH = 0
            for s_a in rollout:
                s = s_a[0]
                a = s_a[1]
                self.N[s][a] += 1
                self.W[s][a] += z
                self.Q[s][a] = self.W[s][a] / self.N[s][a]
                # expansion
                if self.N[s][a] == (self.n_thr + 1):
                    temporary_board = chess.Board(self.states[s])
                    temporary_board.push(self.moves[a])
                    self.add_child(s, a, temporary_board)
            simulation += 1
        
        # select action with max visit count from root
        max_a = 0
        max_visit_count = 0
        last_state = self.root
        for a in self.N[last_state]:
            s = last_state
            if self.N[s][a] > max_visit_count:
                max_visit_count = self.N[s][a]
                max_a = a
        found_next_root = False
        for s_a in self.children[self.root]:
            if s_a[1] == max_a:
                self.root = s_a[0]
                found_next_root = True
                break
        if not found_next_root:
            next_root = self.next_state
            temporary_board = copy.copy(board)
            temporary_board.push(self.moves[max_a])
            self.add_child(self.root, max_a, temporary_board)
            self.root = next_root

        policy = np.zeros(8 * 8 * 73)
        total_visits = 0
        for a in self.N[last_state]:
            total_visits += self.N[last_state][a]
        for a in self.N[last_state]:
            try:
                move_index = self.env.encode(self.moves[a])
                policy[move_index] = self.N[last_state][a] / total_visits
            except:
                continue
        
        self.MOVES_PLAYED.append(self.moves[max_a])
        return self.moves[max_a], policy