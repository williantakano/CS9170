import os
import numpy as np
import pandas as pd
from classes.competitor import Competitor
from classes.market import Market


product = 'men_street_footwear'
df_product = pd.read_csv(os.path.join('dataset', '{}.csv'.format(product)), sep=';')
df_product['week_of_year'] = df_product['week_of_year'].astype(str)

# Initializing Market
class_market = Market(df_product=df_product)
encoder, model_poisson = class_market.model_poisson()

# Q-Learning Settings
range_actions = np.arange(start=class_market.price_min, stop=class_market.price_max + 1, step=1)
range_states = np.arange(start=class_market.price_min, stop=class_market.price_max + 1, step=1)
competitors = ['type_1']
discount = 0.9
epsilons = [0.05, 0.10, 0.25]
n_episodes = 500
n_trials = 100
n_weeks = 52
n_actions = len(range_actions)
n_states = len(range_states)


def select_action(state, epsilon):
    if np.random.uniform() < epsilon:
        action = np.random.choice(np.arange(start=0, stop=n_actions, step=1))
    else:
        action = np.argmax(Q_table[state, :])

    return action


for competitor in competitors:
    print('\nCompetitor: {}'.format(competitor))

    # Declaring arrays that will store the results
    # Third dimension: price, sales and profit
    result_agent = np.zeros(shape=(len(epsilons), n_trials, n_episodes, n_weeks, 3))
    result_competitor = np.zeros(shape=(len(epsilons), n_trials, n_episodes, n_weeks, 3))
    result_q_table = np.zeros(shape=(len(epsilons), n_trials, n_states, n_actions))

    # Initializing competitor
    class_competitor = Competitor(type=competitor)

    for epsilon_index, epsilon in enumerate(epsilons):
        print('Epsilon: {}'.format(epsilon))
        for trial in range(n_trials):
            if trial % 10 == 0:
                print('Trial: {}'.format(trial))

            # Resetting Q-Learning tables
            Q_table = np.zeros(shape=(n_states, n_actions))
            count_table = np.zeros(shape=(n_states, n_actions))

            for episode in range(n_episodes):
                # Initializing state - Competitor's price in the first week
                # For the mimic competitor, it is considered that the last agent's price is equal to price_mean
                price_competitor = class_competitor.get_price(price_agent=class_market.price_mean,
                                                              price_min=class_market.price_min,
                                                              price_max=class_market.price_max,
                                                              price_mean=class_market.price_mean,
                                                              price_std=class_market.price_std)
                state, = np.where(range_states == price_competitor)[0]

                # Iterating over the 52 weeks of a year
                for week in range(n_weeks):
                    action = select_action(state=state, epsilon=epsilon)

                    # Getting the prices
                    price_agent = range_actions[action]
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
                        # Declaring reward (agent's revenue)
                        reward = profit_agent

                        # Next state (competitor's price of the following week)
                        price_competitor = class_competitor.get_price(price_agent=price_agent,
                                                                      price_min=class_market.price_min,
                                                                      price_max=class_market.price_max,
                                                                      price_mean=class_market.price_mean,
                                                                      price_std=class_market.price_std)
                        state_next, = np.where(range_states == price_competitor)[0]

                        # Updating counts
                        count_table[state, action] += 1

                        # Learning rate
                        alpha = 1/count_table[state, action]

                        # Updating Q-Table
                        Q_table[state][action] = Q_table[state][action] + alpha * (reward + discount * max(Q_table[state_next, :]) - Q_table[state][action])

                        # Updating state
                        state = state_next

            result_q_table[epsilon_index, trial] = Q_table

    # Save results
    np.savez(os.path.join('results', 'q_learning_{}'.format(competitor)), result_agent, result_competitor, result_q_table)