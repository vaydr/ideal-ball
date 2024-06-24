import numpy as np
from objects import *

# Define the basketball environment
class BasketballEnv:
    def __init__(self):
        self.court = Court()
        self.basket1 = Basket(0.45)
        self.basket2 = Basket(0.45)
        self.team1 = RLTeam(22, 128, 3)  # Adjusted input size for more state info
        self.team2 = RLTeam(22, 128, 3)  # Adjusted input size for more state info
        self.state = self.reset()

    def reset(self):
        # Reset player positions (10 players) and ball position
        self.state = np.zeros((11, 2))  # 10 players + 1 ball, x and y positions
        return self.state

    def step(self, actions):
        rewards = [0, 0]  # Team 1 and Team 2 rewards
        new_state = self.state.copy()

        for i in range(5):
            if actions[i] == 0:  # Dribble
                direction = np.random.uniform(-1, 1, 2)  # Random direction
                new_state[i] = self.team1.agents[i].dribble(self.state[i], direction)
            elif actions[i] == 1:  # Pass
                new_state[i], success = self.team1.agents[i].pass_ball(self.state[i])
                rewards[0] += success
            elif actions[i] == 2 and self.state[i][1] <= self.court.length / 2:  # Shoot only if in their half
                success = self.team1.agents[i].shoot(self.state[i])
                rewards[0] += success

        for i in range(5, 10):
            if actions[i] == 0:  # Dribble
                direction = np.random.uniform(-1, 1, 2)  # Random direction
                new_state[i] = self.team2.agents[i-5].dribble(self.state[i], direction)
            elif actions[i] == 1:  # Pass
                new_state[i], success = self.team2.agents[i-5].pass_ball(self.state[i])
                rewards[1] += success
            elif actions[i] == 2 and self.state[i][1] >= self.court.length / 2:  # Shoot only if in their half
                success = self.team2.agents[i-5].shoot(self.state[i])
                rewards[1] += success

        self.state = new_state
        return self.state, rewards

# Train the agents
env = BasketballEnv()
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    for t in range(100):  # Each episode lasts for 100 time steps
        actions = env.team1.select_actions(state) + env.team2.select_actions(state)
        new_state, rewards = env.step(actions)
        for agent in env.team1.agents:
            agent.rewards.append(rewards[0])
        for agent in env.team2.agents:
            agent.rewards.append(rewards[1])
        state = new_state

    env.team1.update_policies()
    env.team2.update_policies()

print("Training completed.")
