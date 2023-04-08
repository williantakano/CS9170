import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


product = 'men_street_footwear'
competitors = ['type_1', 'type_2', 'type_3']
models = ['q_learning']
dict_color = {'agent': '#1f77b4',
              'competitor': '#ff7f0e'}

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

    ax.plot(profit_company_mean, label='Agent', color=dict_color['agent'])
    ax.fill_between(range(len(profit_company_mean)),
                    profit_company_mean - profit_company_std,
                    profit_company_mean + profit_company_std,
                    color=dict_color['agent'], alpha=0.3)
    ax.plot(profit_competitor_mean, label='Competitor {}'.format(competitor.split('_')[1]), color=dict_color['competitor'])
    ax.fill_between(range(len(profit_competitor_mean)),
                    profit_competitor_mean - profit_competitor_std,
                    profit_competitor_mean + profit_competitor_std,
                    color=dict_color['competitor'], alpha=0.3)

    return


def main():
    for model in models:
        for competitor in competitors:
            results = np.load(os.path.join('results', '{}_{}.npz'.format(model, competitor)))
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

            ax0.yaxis.set_ticks(np.arange(start=0, stop=4.5 * 10 ** 6, step=0.5 * 10 ** 6))

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
            plt.savefig(os.path.join('plots', '{}_{}.png'.format(model, competitor)), bbox_inches='tight', format='png', dpi=300)
            plt.close()


if __name__ == '__main__':
    main()