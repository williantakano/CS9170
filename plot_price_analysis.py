import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

epsilon_index = 1
competitors = ['type_1', 'type_2']
models = ['q_learning', 'dqn']
products = ['men_street_footwear', 'men_athletic_footwear', 'men_apparel', 'women_street_footwear',
            'women_athletic_footwear', 'women_apparel']

figsize = (7, 4)
font_size_axis = 11
font_size_title = 12


def price_per_week(ax, epsilon_index, competitor, results_q_learning_agent, results_q_learning_competitor, results_dqn_agent, results_dqn_competitor):
    price_q_learning_agent = np.mean(results_q_learning_agent[epsilon_index, :, :, 0], axis=0)
    price_q_learning_competitor = np.mean(results_q_learning_competitor[epsilon_index, :, :, 0], axis=0)
    price_dqn_agent = np.mean(results_dqn_agent[epsilon_index, :, :, 0], axis=0)
    price_dqn_competitor = np.mean(results_dqn_competitor[epsilon_index, :, :, 0], axis=0)

    ax.plot(price_q_learning_agent, label='Q-Learning', color='C0')
    ax.plot(price_q_learning_competitor, label='Competitor', color='C1')
    ax.plot(price_dqn_agent, label='DQN', color='C2')
    ax.plot(price_dqn_competitor, label='Competitor', color='C3')

    return


def main(product):
    # Create folder
    folder = os.path.join('plots', product, 'price_analysis')
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)

    for competitor in competitors:
        results_q_learning = np.load(os.path.join('results', product, 'q_learning', 'q_learning_optimal_{}.npz'.format(competitor)))
        results_q_learning_agent = results_q_learning['arr_0']
        results_q_learning_competitor = results_q_learning['arr_1']

        results_dqn = np.load(os.path.join('results', product, 'dqn', 'dqn_optimal_{}.npz'.format(competitor)))
        results_dqn_agent = results_dqn['arr_0']
        results_dqn_competitor = results_dqn['arr_1']

        fig = plt.figure(figsize=figsize)

        gs = GridSpec(1, 2, figure=fig, wspace=0.15)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])

        price_per_week(ax=ax0,
                       epsilon_index=epsilon_index,
                       competitor=competitor,
                       results_q_learning_agent=results_q_learning_agent,
                       results_q_learning_competitor=results_q_learning_competitor,
                       results_dqn_agent=results_dqn_agent,
                       results_dqn_competitor=results_dqn_competitor)

        plt.show()


if __name__ == '__main__':
    for product in products:
        main(product=product)