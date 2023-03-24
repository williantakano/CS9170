import os
import shutil
import numpy as np
import pandas as pd
from classes.market import Market


product = 'men_street_footwear'
df_product = pd.read_csv(os.path.join('dataset', '{}.csv'.format(product)), sep=';')
df_product['week_of_year'] = df_product['week_of_year'].astype(str)

# Create folder
folder = os.path.join('results', product, 'q_learning')
if os.path.exists(folder):
    shutil.rmtree(folder)

os.makedirs(folder)

class_market = Market(df_product=df_product)
encoder, model_poisson = class_market.model_poisson()

# Q-Learning Settings
range_actions = np.arange(start=class_market.price_min, stop=class_market.price_max + 1, step=1)
range_states = np.arange(start=class_market.price_min, stop=class_market.price_max + 1, step=1)
competitors = ['type_1', 'type_2']
discount = 0.9
epsilons = [0.05, 0.1, 0.25]
n_episodes = 200
n_trials = 100
n_weeks = 52
n_actions = len(range_actions)
n_states = len(range_states)


def select_action(state, epsilon):
    # Checking exploration vs exploitation
    if np.random.uniform() < epsilon:
        action = np.random.choice(np.arange(start=0, stop=n_actions, step=1))
    else:
        action = np.argmax(Q_table[state, :])

    return action


for competitor in competitors:
    print('\nCompetitor: {}'.format(competitor))

    if competitor == 'type_1':
        prices_competitor = class_market.prices_competitor1(n_weeks=n_weeks)
    else:
        prices_competitor = class_market.prices_competitor2(n_weeks=n_weeks)

    # Declaring arrays that will store the results
    # Third dimension: price, sales and profit
    result_company = np.zeros(shape=(len(epsilons), n_trials, n_episodes, n_weeks, 3))
    result_competitor = np.zeros(shape=(len(epsilons), n_trials, n_episodes, n_weeks, 3))

    # Results by taking the optimal policy
    result_optimal_company = np.zeros(shape=(len(epsilons), n_trials, n_weeks, 3))
    result_optimal_competitor = np.zeros(shape=(len(epsilons), n_trials, n_weeks, 3))

    for epsilon_index, epsilon in enumerate(epsilons):
        print('Epsilon: {}'.format(epsilon))
        for trial in range(n_trials):
            if trial % 20 == 0:
                print('Trial: {}'.format(trial))

            # Resetting Q-Learning tables
            Q_table = np.zeros(shape=(n_states, n_actions))
            count_table = np.zeros(shape=(n_states, n_actions))

            for episode in range(n_episodes):
                # Initializing state - Competitor's price in the first week
                state, = np.where(range_states == prices_competitor[0])[0]

                # Looping over the 52 weeks of a year
                for week in range(n_weeks):
                    action = select_action(state=state, epsilon=epsilon)

                    # Getting the prices
                    price_company, price_competitor, price_market = class_market.get_prices(range_actions=range_actions,
                                                                                            action=action,
                                                                                            prices_competitor=prices_competitor,
                                                                                            week=week)

                    # Getting the sales
                    sales_market, sales_company, sales_competitor = class_market.get_sales(encoder=encoder,
                                                                                           model_poisson=model_poisson,
                                                                                           price_market=price_market,
                                                                                           price_company=price_company,
                                                                                           price_competitor=price_competitor,
                                                                                           week=week)

                    # Getting the profits
                    profit_company, profit_competitor = class_market.get_profits(sales_company=sales_company,
                                                                                 price_company=price_company,
                                                                                 sales_competitor=sales_competitor,
                                                                                 price_competitor=price_competitor)

                    # Storing the results
                    # cumulative_discount_reward = (discount ** week) * reward
                    result_company[epsilon_index, trial, episode, week] = np.array([price_company, sales_company, profit_company])
                    result_competitor[epsilon_index, trial, episode, week] = np.array([price_competitor, sales_competitor, profit_competitor])

                    # If it is the last step, there is no next state
                    if week < n_weeks - 1:
                        # Declaring reward (company's revenue) and next state (competitor's price of the following week)
                        reward = profit_company
                        state_next, = np.where(range_states == prices_competitor[week + 1])[0]

                        # Updating counts
                        count_table[state, action] += 1

                        # Learning rate
                        alpha = 1/count_table[state, action]

                        # Updating Q-Table
                        Q_table[state][action] = Q_table[state][action] + alpha * (reward + discount * max(Q_table[state_next, :]) - Q_table[state][action])

                        # Updating state
                        state = state_next

            policy = np.argmax(Q_table, axis=1)

            # Simulate the results if we follow the optimal policy
            for week in range(n_weeks):
                state, = np.where(range_states == prices_competitor[week])[0]
                action = policy[state]

                # Getting the prices
                price_company, price_competitor, price_market = class_market.get_prices(range_actions=range_actions,
                                                                                        action=action,
                                                                                        prices_competitor=prices_competitor,
                                                                                        week=week)

                # Getting the sales
                sales_market, sales_company, sales_competitor = class_market.get_sales(encoder=encoder,
                                                                                       model_poisson=model_poisson,
                                                                                       price_market=price_market,
                                                                                       price_company=price_company,
                                                                                       price_competitor=price_competitor,
                                                                                       week=week)

                # Getting the profits
                profit_company, profit_competitor = class_market.get_profits(sales_company=sales_company,
                                                                             price_company=price_company,
                                                                             sales_competitor=sales_competitor,
                                                                             price_competitor=price_competitor)

                # Storing the results
                result_optimal_company[epsilon_index, trial, week] = np.array([price_company, sales_company, profit_company])
                result_optimal_competitor[epsilon_index, trial, week] = np.array([price_competitor, sales_competitor, profit_competitor])

    # Save results
    np.savez(os.path.join(folder, 'q_learning_{}'.format(competitor)), result_company, result_competitor)
    np.savez(os.path.join(folder, 'q_learning_optimal_{}'.format(competitor)), result_optimal_company, result_optimal_competitor)