import gym
import gym_chess
import random
import numpy as np
import tensorflow as tf
import chess
import chess.pgn
import chess.svg
import copy
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
#stockfish bot
from stockfish import Stockfish
#chess engine
import chess.engine
engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish_13_win_x64_bmi2")

from mcts import MCTS

stockfish = Stockfish("stockfish/stockfish_13_win_x64_bmi2")

input = keras.Input(shape=(12, 8, 8), name='board')

env = gym.make('ChessAlphaZero-v0')

# hidden layers
x = layers.Conv2D(128, 5, padding='same', activation='relu')(input)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

# output layers
tmp = layers.Conv2D(73, 1, padding='valid', activation='relu')(x)
p = layers.Dense(8*8*73, activation='softmax', name='p')(keras.layers.Flatten()(tmp))
v = layers.Dense(1, activation='tanh', name='v')(keras.layers.Flatten()(x))

# stockfish settings #
model_file = "./models/modelStockfish2000EloGames_score.h5"
accuracy_file = "./accuracy/move_accuracy_2000elo_score.txt"
stockfish_elo = 2000
####################

model = keras.Model(
    inputs=[input],
    outputs=[p, v],
)

model.compile(
    optimizer='adam',
    loss=[
        keras.losses.CategoricalCrossentropy(),
        keras.losses.MeanSquaredError(),
    ],
)

mfile = Path(model_file)
if mfile.is_file():
    model = keras.models.load_model(model_file)

afile = Path(accuracy_file)
if not mfile.is_file():
    open(accuracy_file, 'w').close()

def getResult(str):
    if str == "0-1":
        return -1
    elif str == "1-0":
        return 1
    else:
        return 0

def generate_policy(val):
    policy = np.zeros(8 * 8 * 73)
    policy[val] = 1.0
    return policy

def board_score(fen):
    score = 3
    for e in range(len(fen)):
        piece = fen[e]
        if(piece == 'P'):
            score += 1
        elif(piece == 'N' or piece == 'B'):
            score += 3
        elif(piece == 'R'):
            score += 5
        elif(piece == 'Q'):
            score += 7
        elif(piece == 'p'):
            score -= 1
        elif(piece == 'n' or piece == 'b'):
            score -= 3
        elif(piece == 'r'):
            score -= 5
        elif(piece == 'q'):
            score -= 7
    return score

def board_eval(board, m, depth):
    bitBoard = player.boardToBitBoard(board)
    bitBoard = np.asarray(bitBoard)
    bitBoard = bitBoard.reshape(1, 12, 8, 8)
    moves = model.predict(bitBoard)
    picked = []
    scores = []
    while True:
        env.reset()
        for move in player.MOVES_PLAYED:
            env.step(env.encode(move))
        maxIdx = np.argmax(moves[0][0])
        while not maxIdx in env.legal_actions:
            moves[0][0][maxIdx] = 0
            maxIdx = np.argmax(moves[0][0])
        decoded_move = str(env.decode(maxIdx))
        maxMove = chess.Move.from_uci(decoded_move)
        picked.append(maxMove)
        #go depth moves in
        for i in range(depth):
            #create copy of board
            board_copy = copy.copy(board)
            #play previously found best move
            board_copy.push(maxMove)
            #find next move
            bitBoard = player.boardToBitBoard(board_copy)
            bitBoard = np.asarray(bitBoard)
            bitBoard = bitBoard.reshape(1, 12, 8, 8)
            moves = model.predict(bitBoard)
            maxIdx = np.argmax(moves[0][0])
            while not maxIdx in env.legal_actions:
                moves[0][0][maxIdx] = 0
                maxIdx = np.argmax(moves[0][0])
            decoded_move = str(env.decode(maxIdx))
            maxMove = chess.Move.from_uci(decoded_move)
        #done with this iteration, calcualte board score
        fen = str(board_copy.fen())
        score = board_score(fen)
        scores.append(score)
        #enough moves, end
        if len(picked) >= m:
            break

    # pick best score 
    maxScore = scores.index(max(scores))
    maxMove = picked[maxScore]
    return maxMove

player = MCTS(model)

board = chess.Board()
turn = 1
correct_moves = 0
scores = []
allscores = []

training_positions = []
training_policies = []
training_results = []

while True:
    print("-------------------------------------")
    print("TURN: ", turn)
    white_move, policy = player.mcts(board, 50, True)
    training_positions.append(player.boardToBitBoard(board))
    training_policies.append(policy)

    stockfish.set_fen_position(board.fen())
    stockfish.set_elo_rating(1500)
    best_move1500 = stockfish.get_best_move()
    best_move1500 = chess.Move.from_uci(best_move1500)

    stockfish.set_elo_rating(2000)
    best_move2000 = stockfish.get_best_move()
    best_move2000 = chess.Move.from_uci(best_move2000)

    if(best_move1500 == white_move or best_move2000 == white_move):
        correct_moves += 1

    # giving the move a score based on engine score
    score = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = score['score']
    white_score_before = str(score.white())

    board.push(white_move)
    score = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = score['score']
    white_score_after = str(score.white())

    pseudo_score = 0
    if '#' not in white_score_before and '#' not in white_score_after:
        score_diff = int(white_score_after) - int(white_score_before)
        scores.append(score_diff)
        allscores.append(score_diff)
    elif '#' in white_score_before and '#' not in white_score_after:
        if '#-' in white_score_before:
            pseudo_score = 300
        else:
            pseudo_score = -300
        scores.append(pseudo_score)
        allscores.append(pseudo_score)
    elif '#' not in white_score_before and '#' in white_score_after:
        if '#-' in white_score_after:
            pseudo_score = -500
        else:
            pseudo_score = 1000
        scores.append(pseudo_score)
        allscores.append(pseudo_score)
    elif '#' in white_score_before and '#' in white_score_after:
        if '#-' in white_score_before:
            pseudo_score = -200
        else:
            pseudo_score = 200
        scores.append(pseudo_score)
        allscores.append(pseudo_score)

    print('white move ', white_move)
    print("White:")
    print(board)

    if board.is_game_over():
        print(board.result())
        # calculate accurate move %
        move_percentage = round((correct_moves/turn) * 100, 2)
        print("Correct move %: ", move_percentage)
        # calculate move +/-
        avg = int(np.mean(scores))
        # write move accuracy to file
        f = open(accuracy_file, 'r+')
        lines = f.readlines()
        matches = len(lines)
        # format is: match:turns:w/l:move+/-:moveaccuracy
        f.write('\n'+str(matches)+':'+str(turn)+':'+str(board.result())+':'+str(avg)+':'+str(move_percentage))
        f.close()

        result = getResult(board.result())
        training_results = allscores

        training_positions = np.asarray(training_positions)
        training_policies = np.asarray(training_policies)
        training_results = np.asarray(training_results)

        if(len(training_policies) == len(training_results)):
            model.fit(x=training_positions, y={"p": training_policies, "v": training_results}, epochs=7)
            model.save(model_file)

        player = MCTS(model)
        board = chess.Board()
        turn = 0
        correct_moves = 0
        scores = []
        training_positions = []
        training_policies = []
        allscores = []
        continue

    stockfish.set_fen_position(board.fen())
    stockfish.set_elo_rating(stockfish_elo)
    black_move = stockfish.get_best_move()
    black_move = chess.Move.from_uci(black_move)
    try:
        black_move_a = player.env.encode(black_move)
        training_positions.append(player.boardToBitBoard(board))
        training_policies.append(generate_policy(black_move_a))
    except:
        print('move not decoding: ', black_move)
    print('black move ', black_move)
    player.play_move(board, black_move)

    # giving the move a score based on engine score
    score = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = score['score']
    black_score_before = str(score.black())

    board.push(black_move)

    score = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = score['score']
    black_score_after = str(score.black())

    pseudo_score = 0
    if '#' not in black_score_before and '#' not in black_score_after:
        score_diff = int(black_score_after) - int(black_score_before)
        allscores.append(score_diff)
    elif '#' in black_score_before and '#' not in black_score_after:
        if '#-' in black_score_before:
            pseudo_score = 300
        else:
            pseudo_score = -300
        allscores.append(pseudo_score)
    elif '#' not in black_score_before and '#' in black_score_after:
        if '#-' in black_score_after:
            pseudo_score = -400
        else:
            pseudo_score = 500
        allscores.append(pseudo_score)
    elif '#' in black_score_before and '#' in black_score_after:
        if '#-' in black_score_before:
            pseudo_score = -200
        else:
            pseudo_score = 200
        allscores.append(pseudo_score)

    print("Black:")
    print(board)
    

    if board.is_game_over():
        print(board.result())
        # calculate accurate move %
        move_percentage = round((correct_moves/turn) * 100, 2)
        print("Correct move %: ", move_percentage)
        # calculate move +/-
        avg = int(np.mean(scores))
        # write move accuracy to file
        f = open(accuracy_file, 'r+')
        lines = f.readlines()
        matches = len(lines)
        # format is: match:turns:w/l:move+/-:moveaccuracy
        f.write('\n'+str(matches)+':'+str(turn)+':'+str(board.result())+':'+str(avg)+':'+str(move_percentage))
        f.close()

        training_results = allscores

        training_positions = np.asarray(training_positions)
        training_policies = np.asarray(training_policies)
        training_results = np.asarray(training_results)

        if(len(training_policies) == len(training_results)):
            model.fit(x=training_positions, y={"p": training_policies, "v": training_results}, epochs=7)
            model.save(model_file)

        player = MCTS(model)
        board = chess.Board()
        turn = 0
        correct_moves = 0
        scores = []
        training_positions = []
        training_policies = []
        allscores = []
        continue

    turn += 1