import gym
import gym_chess
import random
import numpy as np
import tensorflow as tf
import chess
import chess.pgn
import chess.svg
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, Response, request
from pathlib import Path

pieceTypes = {"None" : -1, "P" : 0, "N" : 1, "B": 2, "R" : 3 , "Q" : 4, "K" : 5, "p" : 6, "n" : 7, "b" : 8, "r" : 9, "q" : 10
, "k" : 11}
def boardToBitBoard(board):
    bitBoard = np.zeros(12 * 8 * 8)
    bitBoard = np.reshape(bitBoard, (12, 8, 8))
    counter = 0
    for i in range(8):
        for j in range(8):
            pieceType = pieceTypes[str(board.piece_at(counter))]
            if not pieceType == -1:
                bitBoard[pieceType][i][j] = 1
            counter += 1
    return bitBoard

def generate_random_board():
    board = np.zeros(12 * 8 * 8)
    board = np.reshape(board, (12, 8, 8))
    board[0][0][5] = 1.0
    board[11][1][1] = 1.0
    return board

def generate_policy(val):
    policy = np.zeros(8 * 8 * 73)
    policy[val] = 1.0
    return policy

def getResult(str):
    if str == "0-1":
        return -1
    elif str == "1-0":
        return 1
    else:
        return 0

env = gym.make('ChessAlphaZero-v0')


# input layer
input = keras.Input(shape=(12, 8, 8), name='board')

# hidden layers
x = layers.Conv2D(128, 5, padding='same', activation='relu')(input)
x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

# output layers
tmp = layers.Conv2D(73, 1, padding='valid', activation='relu')(x)
p = layers.Dense(8*8*73, activation='softmax', name='p')(keras.layers.Flatten()(tmp))
v = layers.Dense(1, activation='tanh', name='v')(keras.layers.Flatten()(x))

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

# if model file already exists, load model
model_file = "./models/modelGmGames.h5"
mfile = Path(model_file)
if mfile.is_file():
    model = keras.models.load_model(model_file)

gm_games = ["./GM_games/RichardRapport.pgn", "./GM_games/AdolfAnderssen.pgn", "./GM_games/SamuelShankland.pgn", "./GM_games/AntonSmirnov.pgn", "./GM_games/TaniaSachdev.pgn" ]
gm_game = 0
while True:
    print('Training on GM games: ', gm_games[gm_game])
    env.reset()
    turn = True # True = White's turn, False = Black's turn

    #Get training data and save game moves to array 
    pgn = open(gm_games[gm_game])
    games1 = []
    while True:
        try: 
            newGame = chess.pgn.read_game(pgn)
            games1.append(newGame)
            print(newGame.mainline_moves())
        except:
            break

    print("Training on: ", len(games1), " games")
    input_boards = []
    input_labels = []
    input_results = []
    gameCounter = 0
    for game in games1:
        print(gameCounter)
        gameCounter += 1
        board = chess.Board()
        bitBoard = boardToBitBoard(board)
        intermediate_input_boards = []
        intermediate_input_labels = []
        intermediate_input_results = []
        try:
            for move in game.mainline_moves():
                try:
                    board.push(move)        
                    intermediate_input_boards.append(bitBoard)
                    intermediate_input_labels.append(generate_policy(env.encode(move)))
                    intermediate_input_results.append(getResult(game.headers["Result"]))
                    bitBoard = boardToBitBoard(board)
                except: 
                    break #If a move doesn't work then scrap the whole game 
        except:
            continue
        #If the whole game goes through properly we can add the results to our real inputs 
        input_boards += intermediate_input_boards
        input_labels += intermediate_input_labels
        input_results += intermediate_input_results


    #stockfish bot
    from stockfish import Stockfish
    #chess engine for board evaluation
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish_13_win_x64_bmi2")

    from mcts import MCTS

    while len(input_boards) != len(input_labels):
        input_boards.pop()

    iteration = 0
    match_every = 5000
    while len(input_boards) > (match_every*(iteration+1)):
        boards = np.asarray(input_boards[iteration*match_every:(iteration+1)*match_every])
        labels = np.asarray(input_labels[iteration*match_every:(iteration+1)*match_every])
        results = np.asarray(input_results[iteration*match_every:(iteration+1)*match_every])

        model.fit(x=boards, y={"p": labels, "v": results}, epochs=1)

        ############################ MATCH IS PLAYED ############################

        stockfish = Stockfish("stockfish/stockfish_13_win_x64_bmi2")
        accuracy_file = "./accuracy/move_accuracy_GM.txt"
        stockfish_elo = 1500

        afile = Path(accuracy_file)
        if not mfile.is_file():
            open(accuracy_file, 'w').close()

        player = MCTS(model)

        board = chess.Board()
        turn = 1
        correct_moves = 0
        scores = []

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

            # play move
            board.push(white_move)

            score = engine.analyse(board, chess.engine.Limit(time=0.1))
            score = score['score']
            white_score_after = str(score.white())

            pseudo_score = 0
            if '#' not in white_score_before and '#' not in white_score_after:
                score_diff = int(white_score_after) - int(white_score_before)
                scores.append(score_diff)
            elif '#' in white_score_before and '#' not in white_score_after:
                if '#-' in white_score_before:
                    pseudo_score = 400
                else:
                    pseudo_score = -300
                scores.append(pseudo_score)
            elif '#' not in white_score_before and '#' in white_score_after:
                if '#-' in white_score_after:
                    pseudo_score = -500
                else:
                    pseudo_score = 1000
                scores.append(pseudo_score)
            elif '#' in white_score_before and '#' in white_score_after:
                if '#-' in white_score_before:
                    pseudo_score = -200
                else:
                    pseudo_score = 300
                    scores.append(pseudo_score)

            print('white move ', white_move)
            print("White:")
            print(board)

            if board.is_game_over():
                print(board.result())
                # calculate accurate move %
                move_percentage = round((correct_moves/turn) * 100, 2)
                # calculate move +/-
                avg = int(np.mean(scores))
                # write move accuracy to file
                f = open(accuracy_file, 'r+')
                lines = f.readlines()
                matches = len(lines)
                # format is: match:turns:w/l:move+/-:moveaccuracy
                f.write('\n'+str(matches)+':'+str(turn)+':'+str(board.result())+':'+str(avg)+':'+str(move_percentage))
                f.close()

                player = MCTS(model)
                board = chess.Board()
                turn = 0
                correct_moves = 0
                scores = []
                break

            stockfish.set_fen_position(board.fen())
            stockfish.set_elo_rating(stockfish_elo)
            black_move = stockfish.get_best_move()
            black_move = chess.Move.from_uci(black_move)
            print('black move ', black_move)
            player.play_move(board, black_move)
            board.push(black_move)

            print("Black:")
            print(board)
            

            if board.is_game_over():
                print(board.result())
                # calculate accurate move %
                move_percentage = round((correct_moves/turn) * 100, 2)
                # calculate move +/-
                avg = int(np.mean(scores))
                # write move accuracy to file
                f = open(accuracy_file, 'r+')
                lines = f.readlines()
                matches = len(lines)
                # format is: match:turns:w/l:move+/-:moveaccuracy
                f.write('\n'+str(matches)+':'+str(turn)+':'+str(board.result())+':'+str(avg)+':'+str(move_percentage))
                f.close()

                player = MCTS(model)
                board = chess.Board()
                turn = 0
                correct_moves = 0
                scores = []
                break

            turn += 1

        iteration += 1
        model.save(model_file)


    gm_game += 1
    if gm_game >= 5:
        gm_game = 0


    env.close()