import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1  # Player 1 starts
        self.winner = None

    def is_valid_move(self, row, col):
        return self.board[row, col] == 0
    
    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.switch_player()

    def switch_player(self):
        self.current_player = 3 - self.current_player  # Toggle between players (1 and 2)

    def is_game_over(self):
        for player in [1, 2]:
            for i in range(3):
                # Check rows, columns, and diagonals
                if (np.all(self.board[i, :] == player) or
                        np.all(self.board[:, i] == player) or
                        np.all(np.diag(self.board) == player) or
                        np.all(np.diag(np.fliplr(self.board)) == player)):
                    self.winner = player
                    return True
        # Check for a draw
        if not 0 in self.board:
            self.winner = 0
            return True
        return False
    
    def get_state(self):
        return tuple(self.board.reshape(9))
    
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.3, gamma=0.9):
        self.q_table = {}  # Q-table to store state-action values
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

    def choose_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            max_q_value = -float("inf")
            best_action = None
            for action in valid_moves:
                q_value = self.q_table.get((state, action), 0)
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action
            return best_action
        
    def learn(self, state, action, reward, next_state):
        current_q_value = self.q_table.get((state, action), 0)
        max_next_q_value = max([self.q_table.get((next_state, next_action), 0) for next_action in range(9)]
                               if next_state != () else [0])
        
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_next_q_value)
        self.q_table[(state, action)] = new_q_value

def train_q_learning_agent(agent, episodes=1000):
    for episode in range(episodes):
        game = TicTacToe()

        while not game.is_game_over():
            state = game.get_state()
            valid_moves = [i for i in range(9) if game.is_valid_move(i // 3, i % 3)]
            action = agent.choose_action(state, valid_moves)     
            game.make_move(action // 3, action % 3)
            next_state = game.get_state()
            if game.is_game_over():
                if game.winner == 1:
                    reward = 1
                elif game.winner == 2:
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0
            agent.learn(state, action, reward, next_state)

def test_q_learning_agent(agent):
    game = TicTacToe()

    while not game.is_game_over():
        state = game.get_state()
        valid_moves = [i for i in range(9) if game.is_valid_move(i // 3, i % 3)]
        if game.current_player == 1:
            # Q-learning agent's turn
            action = agent.choose_action(state, valid_moves)
        else:
            # Human player's turn
            print("Current board:")
            print(game.board)
            action = int(input("Enter your move (0-8): "))
        game.make_move(action // 3, action % 3)
    print("Game over. Winner:", game.winner)
    print("Final board:")
    print(game.board)

if __name__ == "__main__":
    agent = QLearningAgent()
    train_q_learning_agent(agent)
    test_q_learning_agent(agent)
# "