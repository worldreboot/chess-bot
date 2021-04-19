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
import time

env = gym.make('ChessAlphaZero-v0')
env.reset()
model = keras.models.load_model("./models/modelGmGames.h5")
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

def getBestCompMove(board):
	bitBoard = boardToBitBoard(board)
	bitBoard = np.asarray(bitBoard)
	bitBoard = bitBoard.reshape(1, 12, 8, 8)
	pp = model.predict(bitBoard)
	move = ""
	try:
		maxMove = np.argmax(pp[0][0])
		move = chess.Move.from_uci(str(env.decode(np.argmax(pp[0][0]))))
	except: 
		pp[0][0][maxMove] = 0
	while not move in board.legal_moves:
		try:
			maxMove = np.argmax(pp[0][0])
			move = chess.Move.from_uci(str(env.decode(np.argmax(pp[0][0]))))
			pp[0][0][maxMove] = 0
		except: 
			pp[0][0][maxMove] = 0
	return move


app = Flask(__name__)
@app.route("/")
def hello():
	ret = '<html><body><img width = 600 height = 600 src="board.svg"></img>'
	ret += '<form action="/move" autocomplete="off"><input name = "move" type="text"></input><input type ="submit" value="human move"></form> '
	ret += '<form action = "/newgame"><input type ="submit" value="New Game"></form>'
	return ret

@app.route("/newgame")
def newGame():
	board.reset()
	return hello()

def computer_move():
  	# computer move
	move = getBestCompMove(board)
	board.push(move)

@app.route("/board.svg")
def board():
	return Response(chess.svg.board(board), mimetype='image/svg+xml')

# move given in algebraic notation
@app.route("/move")
def move():
	is_uci = False
	move = ""
	try:
		move = board.parse_uci(str(request.args.get("move", default ="")))
		is_uci = True
	except:
		try:
			move = board.parse_san(str(request.args.get("move", default ="")))
			is_uci = False
		except:
			return hello()
	move = board.san(move)
	if not move in str(board.legal_moves):
		return hello()
	if is_uci:
		board.push_uci(request.args.get("move", default =""))
	else:
		board.push_san(request.args.get("move", default =""))
	if board.is_game_over():
		time.sleep(1.5)
		return gameOver()
	computer_move()
	if board.is_game_over():
		time.sleep(1.5)
		return gameOver()
	return hello()

@app.route("/gameover")
def gameOver():
	ret = '<html><body><h1>Game Over</h1>'
	ret += '<form action = "/newgame"><input type ="submit" value="New Game"></form>'
	return ret

if __name__ == "__main__":
	board = chess.Board()
	app.run()

env.close()