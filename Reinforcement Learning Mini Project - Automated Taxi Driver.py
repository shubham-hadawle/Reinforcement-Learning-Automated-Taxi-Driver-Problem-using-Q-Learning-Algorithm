import numpy as np
import gym


# Define the environment
env = gym.make("Taxi-v3", render_mode = 'human').env

# Initialize the q-table with zero values
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # learning-rate
gamma = 0.7  # discount-factor
epsilon = 0.1  # explor vs exploit

# Random generator
rng = np.random.default_rng()

# Perform 100,000 episodes
for i in range(50):
    print('Current iteration = ',i)
    # Reset the environment
    state = env.reset()
    state = state[0]

    done = False

    # Loop as long as the game is not over, i.e. done is not True
    while not done:
        if rng.random() < epsilon:
            action = env.action_space.sample()  # Explore the action space
            print('...........Explore')
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
            print('Exploit')

        # Apply the action 
        # next_state, reward, done, info = env.step(action)
        step_result = env.step(action)
        next_state = step_result[0]
        reward = step_result[1]
        done = step_result[2]
        info = step_result[3]

        # current Q-value for the state/action couple
        current_value = q_table[state, action]
        next_max = np.max(q_table[next_state])  # next best Q-value

        # Compute the new Q-value with the Bellman equation
        q_table[state, action] = (1 - alpha) * current_value + alpha * (reward + gamma * next_max)
        state = next_state
print('The q-table is :')
print(q_table)