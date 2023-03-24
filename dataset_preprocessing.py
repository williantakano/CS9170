import os
import numpy as np
import pandas as pd


file = os.path.join('dataset', 'Adidas US Sales Datasets.csv')

df = pd.read_csv(file,
                 decimal=',',
                 names=['date', 'product', 'price_unit', 'units_sold'],
                 sep=';',
                 skiprows=1,
                 thousands='.',
                 usecols=[3, 7, 8, 9])

df.replace({"Men's Street Footwear": 'men_street_footwear',
            "Men's Athletic Footwear": 'men_athletic_footwear',
            "Men's Apparel": 'men_apparel',
            "Women's Street Footwear": 'women_street_footwear',
            "Women's Athletic Footwear": 'women_athletic_footwear',
            "Women's Apparel": 'women_apparel'}, inplace=True)


df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y').dt.date
df['year'] = pd.DatetimeIndex(df['date']).year
df['week_of_year'] = pd.DatetimeIndex(df['date']).week

df['price_unit'] = df['price_unit'].str[2:]
df['price_unit'] = df['price_unit'].str.replace(',', '.')
df['price_unit'] = pd.to_numeric(df['price_unit'])

for product in list(set(df['product'])):
    df_product = df.loc[df['product'] == product]
    df_product = df_product.groupby(['year', 'week_of_year'], as_index=False).agg(units_sold=('units_sold', np.sum),
                                                                                  price_unit=('price_unit', np.mean))
    df_product.sort_values(by=['year', 'week_of_year'], inplace=True, ignore_index=True)
    df_product.to_csv(os.path.join('dataset', '{}.csv'.format(product)), sep=';', index=False)