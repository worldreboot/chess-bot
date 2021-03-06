import gym
import gym_chess
import random

env = gym.make('Chess-v0')
print(env.render())

env.reset()
turn = True # True = White's turn, False = Black's turn

while True:
    if not env._ready:
        print(env._board.result())
        break
    if turn:
        print('White to play. White has ' + str(len(env.legal_moves)) + ' legal moves.')
    else:
        print('Black to play. Black has ' + str(len(env.legal_moves)) + ' legal moves.')
    action = random.choice(env.legal_moves)
    env.step(action)
    print(env.render(mode='unicode'))
    print('-----------------------------------')
    turn = not turn

env.close()