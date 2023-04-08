import os
import numpy as np
import pandas as pd
from classes.competitor import Competitor
from classes.market import Market
from classes.deep_q_learning import DQN, ReplayMemory, Transition

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


product = 'men_street_footwear'
df_product = pd.read_csv(os.path.join('dataset', '{}.csv'.format(product)), sep=';')
df_product['week_of_year'] = df_product['week_of_year'].astype(str)

# Initializing Market
class_market = Market(df_product=df_product)
encoder, model_poisson = class_market.model_poisson()

# Deep Q-Learning Settings
range_actions = torch.arange(start=class_market.price_min, end=class_market.price_max + 1, step=1)
range_states = torch.arange(start=class_market.price_min, end=class_market.price_max + 1, step=1)
batch_size = 32
competitors = ['type_1', 'type_2', 'type_3']
discount = 0.9
epsilons = [0.05, 0.1, 0.25]
hidden_size = 32
learning_rate = 0.001
tau = 0.001
n_episodes = 500
n_trials = 100
n_weeks = 52
n_actions = len(range_actions)
n_states = len(range_states)


def select_action(state, epsilon):
    if np.random.uniform() < epsilon:
        action = torch.randint(low=0, high=n_actions, size=(1,), device=device).view(1, -1)
    else:
        action = nn_policy(state).max(1)[1].view(1, -1)

    return action


def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size=batch_size)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.state_next)), device=device, dtype=torch.bool)
    non_final_states_next = torch.cat([s for s in batch.state_next if s is not None])
    batch_state = torch.cat(batch.state)
    batch_action = torch.cat(batch.action)
    batch_reward = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = nn_policy(batch_state).gather(1, batch_action)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = nn_target(non_final_states_next).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * discount) + batch_reward

    # Compute MSE loss
    criterion = torch.nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return


for competitor in competitors:
    print('\nCompetitor: {}'.format(competitor))

    # Declaring arrays that will store the results
    # Third dimension: price, sales and profit
    result_agent = np.zeros(shape=(len(epsilons), n_trials, n_episodes, n_weeks, 3))
    result_competitor = np.zeros(shape=(len(epsilons), n_trials, n_episodes, n_weeks, 3))

    # Initializing competitor
    class_competitor = Competitor(type=competitor)

    for epsilon_index, epsilon in enumerate(epsilons):
        print('Epsilon: {}'.format(epsilon))
        for trial in range(n_trials):
            if trial % 10 == 0:
                print('Trial: {}'.format(trial))

            torch.cuda.empty_cache()

            nn_policy = DQN(n_states=n_states, n_actions=n_actions, hidden_size=hidden_size).to(device)
            nn_target = DQN(n_states=n_states, n_actions=n_actions, hidden_size=hidden_size).to(device)
            nn_target.load_state_dict(nn_policy.state_dict())

            optimizer = optim.Adam(nn_policy.parameters(), lr=learning_rate)
            memory = ReplayMemory(10000)

            for episode in range(n_episodes):
                # Initializing state - Competitor's price in the first week
                # For the mimic competitor, it is considered that the last agent's price is equal to price_mean
                price_competitor = class_competitor.get_price(price_agent=class_market.price_mean,
                                                              price_min=class_market.price_min,
                                                              price_max=class_market.price_max,
                                                              price_mean=class_market.price_mean,
                                                              price_std=class_market.price_std)

                # Vector of dimension n_states. Zeros in all elements except the index of competitor's action
                state = (range_states == price_competitor).float().view(1, -1)

                # Looping over the 52 weeks of a year
                for week in range(n_weeks):
                    action = select_action(state=state, epsilon=epsilon)

                    # Getting the prices
                    price_agent = np.array(range_actions)[action]
                    price_market = class_market.price_market(price_agent=price_agent, price_competitor=price_competitor)

                    # Getting the sales
                    sales_market, sales_agent, sales_competitor = class_market.get_sales(encoder=encoder,
                                                                                         model_poisson=model_poisson,
                                                                                         price_market=price_market,
                                                                                         price_agent=price_agent,
                                                                                         price_competitor=price_competitor,
                                                                                         week=week)

                    # Getting the profits
                    profit_agent, profit_competitor = class_market.get_profits(sales_agent=sales_agent,
                                                                               price_agent=price_agent,
                                                                               sales_competitor=sales_competitor,
                                                                               price_competitor=price_competitor)

                    # Storing the results
                    result_agent[epsilon_index, trial, episode, week] = np.array([price_agent, sales_agent, profit_agent])
                    result_competitor[epsilon_index, trial, episode, week] = np.array([price_competitor, sales_competitor, profit_competitor])

                    # If it is the last step, there is no next state
                    if week < n_weeks - 1:
                        # Declaring reward (company's revenue)
                        reward = torch.tensor(profit_agent).view(1)

                        # Next state (competitor's price of the following week)
                        price_competitor = class_competitor.get_price(price_agent=price_agent,
                                                                      price_min=class_market.price_min,
                                                                      price_max=class_market.price_max,
                                                                      price_mean=class_market.price_mean,
                                                                      price_std=class_market.price_std)
                        state_next = (range_states == price_competitor).float().view(1, -1)

                        # Store the transition in memory
                        memory.push(state, action, state_next, reward)

                        # Move to the next state
                        state = state_next

                        # Perform one step of the optimization (on the policy network)
                        optimize_model()

                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        nn_target_state_dict = nn_target.state_dict()
                        nn_policy_state_dict = nn_policy.state_dict()
                        for key in nn_policy_state_dict:
                            nn_target_state_dict[key] = nn_policy_state_dict[key] * tau + nn_target_state_dict[key] * (1 - tau)

                        nn_target.load_state_dict(nn_target_state_dict)

    # Save results
    np.savez(os.path.join('results', 'dqn_{}'.format(competitor)), result_agent, result_competitor)