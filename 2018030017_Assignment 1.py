import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log

# Stochastic Bandits Environment
class StochasticBanditsEnvironment:
    """
    Stochastic bandits environment where there are k arms and each arm gives a random reward based on a uniform distribution
    with a different range [a_i, b_i].
    """
    def __init__(self, k):
        self.k = k
        # Initialize ranges for each arm
        self.a = np.random.rand(k)
        self.b = np.random.rand(k)
        # Make sure no two arms have the same range
        while any(self.a == self.b):
            self.a = np.random.rand(k)
            self.b = np.random.rand(k)
        
    def step(self, action):
        """
        Sample a reward for the selected arm
        :param action: the index of the selected arm
        :return: the reward obtained from the selected arm
        """
        reward = np.random.uniform(self.a[action], self.b[action])
        return reward

# Epsilon-Greedy Algorithm
def epsilon_greedy(env, T):
    """
    Epsilon-greedy algorithm that explores randomly with probability epsilon and exploits the best arm with probability 1-epsilon.
    :param env: the environment with the k arms
    :param T: the number of iterations
    :return: the regret and reward history for each iteration
    """
    k = env.k
    # Decay rate of exploration probability
    epsilon = lambda t: (t ** (-1/3)) * (k * log(t)) ** (1/3)
    # Initialize action-value function and count of each action
    Q = np.zeros(k)
    N = np.zeros(k)
    # Initialize regret and reward history
    regret = np.zeros(T)
    reward_list = np.zeros(T)
    # Compute optimal reward
    optimal_reward = max(env.b)
    
    # Iterate for T steps
    for t in range(1, T+1):
        # Choose a random action with probability epsilon, otherwise select the best action
        if np.random.rand() < epsilon(t):
            action = np.random.randint(k)
        else:
            action = np.argmax(Q)
        
        # Sample a reward from the selected arm and update the action-value function and count for the selected action
        reward = env.step(action)
        # Update the estimated Q-value for the selected action.
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        
        # Calculate the current reward as the average of all rewards so far,
        # if there have been any, or 0 otherwise.
        current_reward = reward_list[:t].mean() if t > 1 else 0
        
        # Calculate the regret as the difference between the optimal reward
        # and the current reward.
        regret[t-1] = optimal_reward - current_reward
        
        # Store the observed reward for the current time step.
        reward_list[t-1] = reward
    
    # Return the cumulative regret and reward arrays.
    return regret, reward_list


#UCB Algorithm
def UCB(env, T):
    """"
    Define an Upper Confidence Bound (UCB) algorithm that selects the action with the highest
    UCB value, which is the sum of the estimated Q-value and a confidence bound that
    depends on the number of times the action has been selected.
    """
    k = env.k
    # Initialize arrays to store the estimated mean reward (mu_hat) and the number of times
    # each action has been selected (Q).
    mu_hat = np.zeros(k)
    Q = np.ones(k)
    regret = np.zeros(T)
    reward_list = np.zeros(T)
    # Calculate the maximum possible reward in the environment.
    optimal_reward = max(env.b)

    # Loop over each time step.
    for t in range(1, T + 1):
        # Calculate the UCB values for each action.
        UCB_values = mu_hat + np.sqrt(np.log(T) / Q)
        # Choose the action with the highest UCB value.
        action = np.argmax(UCB_values)
        # Take a step in the environment with the chosen action.
        reward = env.step(action)
        # Update the number of times the chosen action has been selected.
        Q[action] += 1
        # Update the estimated mean reward for the chosen action.
        mu_hat[action] += (reward - mu_hat[action]) / Q[action]
        # Calculate the cumulative regret at the current time step.
        current_reward = reward_list[:t].mean() if t > 1 else 0
        regret[t - 1] = optimal_reward - current_reward
        # Add the reward from the current time step to the list of rewards.
        reward_list[t - 1] = reward

    # Return the array of cumulative regret and the array of rewards.
    return regret, reward_list


def plot_regrets_and_rewards(T_values, k_values):
    # create a 3x3 grid of subplots with a size of 12x8 inches
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    
    # loop over all combinations of T and k values
    for i, (T, k) in enumerate(zip(T_values, k_values)):
        # create a StochasticBanditsEnvironment with k arms
        env = StochasticBanditsEnvironment(k)
        
        # run epsilon-greedy and UCB algorithms on the environment for T time steps
        eg_regret, eg_rewards = epsilon_greedy(env, T)
        ucb_regret, ucb_rewards = UCB(env, T)
        
        # calculate the expected regret for each algorithm at each time step
        t_values = np.arange(1, T + 1)
        eg_expected_regret = np.array([t**(2/3) * (k * log(t))**(1/3) for t in t_values])
        ucb_expected_regret = np.array([sqrt(t) for t in t_values])
        
        # plot the cumulative regret for each algorithm
        axs[i][0].plot(np.cumsum(eg_regret), label="ε-Greedy")
        axs[i][0].plot(np.cumsum(ucb_regret), label="UCB")
        # plot the expected regret for each algorithm
        axs[i][0].plot(eg_expected_regret, label="ε-Greedy Expected Regret", linestyle='--')
        axs[i][0].plot(ucb_expected_regret, label="UCB Expected Regret", linestyle='--')
        # set labels and title for the subplot
        axs[i][0].set_xlabel("Time")
        axs[i][0].set_ylabel("Cumulative Regret")
        axs[i][0].set_title(f"T = {T}, k = {k}")
        # show a legend on the subplot
        axs[i][0].legend()
        
        # plot the cumulative reward for each algorithm
        axs[i][1].plot(np.cumsum(eg_rewards), label="ε-Greedy")
        axs[i][1].plot(np.cumsum(ucb_rewards), label="UCB")
        # set labels and title for the subplot
        axs[i][1].set_xlabel("Time")
        axs[i][1].set_ylabel("Cumulative Reward")
        axs[i][1].set_title(f"T = {T}, k = {k}")
        # show a legend on the subplot
        axs[i][1].legend()
        
        # plot the reward per iteration for each algorithm
        axs[i][2].plot(eg_regret, label="ε-Greedy")
        axs[i][2].plot(ucb_regret, label="UCB")
        # set labels and title for the subplot
        axs[i][2].set_xlabel("Time")
        axs[i][2].set_ylabel("Regret per Iteration")
        axs[i][2].set_title(f"T = {T}, k = {k}")
        # show a legend on the subplot
        axs[i][2].legend()
    
    # adjusts the spacing between subplots
    plt.tight_layout()
    # displays the figure with all the subplots
    plt.show()
# number of iterations
T_values = [1000, 1000, 10000]
# number of arms
k_values = [10, 80, 10]

plot_regrets_and_rewards(T_values, k_values)