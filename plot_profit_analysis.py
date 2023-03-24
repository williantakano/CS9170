import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


competitors = ['type_1', 'type_2']
models = ['q_learning', 'dqn']
dict_range_products = {'men_street_footwear': np.arange(start=0, stop=4.5 * 10 ** 6, step=0.5 * 10 ** 6),
                       'men_athletic_footwear': np.arange(start=0, stop=3 * 10 ** 6, step=0.5 * 10 ** 6),
                       'men_apparel': np.arange(start=0, stop=2.5 * 10 ** 6, step=0.5 * 10 ** 6),
                       'women_street_footwear': np.arange(start=0, stop=2.5 * 10 ** 6, step=0.5 * 10 ** 6),
                       'women_athletic_footwear': np.arange(start=0, stop=2.5 * 10 ** 6, step=0.5 * 10 ** 6),
                       'women_apparel': np.arange(start=0, stop=3.5 * 10 ** 6, step=0.5 * 10 ** 6)}

figsize = (11, 4)
font_size_axis = 11
font_size_title = 12


def profit_per_episode(epsilon_index, competitor, result_company, result_competitor, ax):
    # Profit over the 52 weeks
    profit_company = np.sum(result_company[epsilon_index, :, :, :, 2], axis=2)
    profit_competitor = np.sum(result_competitor[epsilon_index, :, :, :, 2], axis=2)

    # Profit considering the trials
    profit_company_mean = np.mean(profit_company, axis=0)
    profit_company_std = np.std(profit_company, axis=0)
    profit_competitor_mean = np.mean(profit_competitor, axis=0)
    profit_competitor_std = np.std(profit_competitor, axis=0)

    ax.plot(profit_company_mean, label='Agent', color='b')
    ax.fill_between(range(len(profit_company_mean)),
                    profit_company_mean - profit_company_std,
                    profit_company_mean + profit_company_std,
                    color='b', alpha=0.3)
    ax.plot(profit_competitor_mean, label='Competitor {}'.format(competitor.split('_')[1]), color='g')
    ax.fill_between(range(len(profit_competitor_mean)),
                    profit_competitor_mean - profit_competitor_std,
                    profit_competitor_mean + profit_competitor_std,
                    color='g', alpha=0.3)

    return


def main(product):
    # Create folder
    folder = os.path.join('plots', product, 'profit_analysis')
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)

    for model in models:
        for competitor in competitors:
            results = np.load(os.path.join('results', product, model, '{}_{}.npz'.format(model, competitor)))
            result_company = results['arr_0']
            result_competitor = results['arr_1']

            fig = plt.figure(figsize=figsize)

            gs = GridSpec(1, 3, figure=fig, wspace=0.15)

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
            ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)

            profit_per_episode(epsilon_index=0, competitor=competitor, result_company=result_company, result_competitor=result_competitor, ax=ax0)
            profit_per_episode(epsilon_index=1, competitor=competitor, result_company=result_company, result_competitor=result_competitor, ax=ax1)
            profit_per_episode(epsilon_index=2, competitor=competitor, result_company=result_company, result_competitor=result_competitor, ax=ax2)

            ax0.yaxis.set_ticks(dict_range_products[product])

            ax0.set_ylabel('Reward', fontsize=font_size_axis)

            ax0.set_xlabel('', fontsize=font_size_axis)
            ax1.set_xlabel('Episodes', fontsize=font_size_axis)
            ax2.set_xlabel('', fontsize=font_size_axis)

            ax0.set_title('Epsilon 0.05', fontsize=font_size_title)
            ax1.set_title('Epsilon 0.10', fontsize=font_size_title)
            ax2.set_title('Epsilon 0.25', fontsize=font_size_title)

            ax0.spines[['right', 'top']].set_visible(False)
            ax1.spines[['right', 'top']].set_visible(False)
            ax2.spines[['right', 'top']].set_visible(False)

            ax1.tick_params(labelleft=False)
            ax2.tick_params(labelleft=False)

            plt.legend(loc='lower right', fancybox=True)
            plt.savefig(os.path.join(folder, '{}_{}.png'.format(model, competitor)), bbox_inches='tight', format='png', dpi=300)
            plt.close()


if __name__ == '__main__':
    for product in dict_range_products.keys():
        main(product=product)