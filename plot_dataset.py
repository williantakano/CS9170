import os
import pandas as pd
import matplotlib.pyplot as plt

product = 'men_street_footwear'

font_size_axis = 11
font_size_tick = 10


def price_x_sales(df_product, folder):
    df_product['week_of_year'] = df_product['week_of_year'].astype(str).str.pad(2, fillchar='0')
    df_product['year'] = df_product['year'].astype(str)
    df_product['date'] = df_product['year'] + '-' + df_product['week_of_year']

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax2 = ax1.twinx()

    line1 = ax1.plot(df_product['date'], df_product['price_unit'], color='C0', label='Price')
    line2 = ax2.plot(df_product['date'], df_product['units_sold']/1000, color='C1', label='Sales')

    ax1.set_xlabel('Year-Week')
    ax1.set_ylabel('Price per unit (R$)', size=font_size_axis)
    ax1.tick_params(axis='both', which='major', labelsize=font_size_tick)
    ax1.set_xticks(df_product['date'].values[::12])
    ax1.set_xticklabels(df_product['date'].values[::12], rotation=30)

    ax2.set_ylabel('Units sold per week ($10^3$)', size=font_size_axis)
    ax2.tick_params(axis='both', which='major', labelsize=font_size_tick)

    lines = line1 + line2
    labels = [label.get_label() for label in lines]
    ax1.legend(lines, labels, loc='upper left', fancybox=True, ncol=1)

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'price_x_sales.png'), bbox_inches='tight', format='png', dpi=300)
    plt.close()

    return


def price_histogram(df_product, folder):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(df_product['price_unit'], bins=20, ec='black')

    ax.set_ylabel('Count', size=font_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'histogram_price.png'), bbox_inches='tight', format='png', dpi=300)
    plt.close()

    return


def sales_histogram(df_product, folder):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.hist(df_product['units_sold'], bins=15, ec='black')

    ax.set_ylabel('Count', size=font_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=font_size_tick)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'histogram_sales.png'), bbox_inches='tight', format='png', dpi=300)
    plt.close()

    return


if __name__ == "__main__":
    folder = os.path.join('plots')
    df_product = pd.read_csv(os.path.join('dataset', '{}.csv'.format(product)), sep=';')

    price_x_sales(df_product=df_product, folder=folder)
    price_histogram(df_product=df_product, folder=folder)
    sales_histogram(df_product=df_product, folder=folder)