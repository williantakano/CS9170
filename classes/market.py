import math
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn import linear_model


class Market(object):

    def __init__(self, df_product):
        self.df_product = df_product
        self.cost_production = int(df_product['price_unit'].min())
        self.price_min = int(df_product['price_unit'].min())
        self.price_max = int(df_product['price_unit'].max())
        self.price_mean = int(df_product['price_unit'].mean())
        self.price_std = int(df_product['price_unit'].std())

    def model_poisson(self):
        # Poisson Regression Model
        # - Model used to simulate the market
        # - Given the market average price, the model return the number of units sold considering the whole market
        # - BinaryEncoder is used to avoid sparse data (52 week in a year)
        encoder = ce.BinaryEncoder()
        X_week_of_year = encoder.fit_transform(self.df_product['week_of_year']).values
        X_price_unit = self.df_product[['price_unit']].values
        X = np.concatenate((X_price_unit, X_week_of_year), axis=1)
        y = self.df_product['units_sold'].values

        model_poisson = linear_model.PoissonRegressor()
        model_poisson.fit(X, y)

        return encoder, model_poisson

    def prices_competitor1(self, n_weeks):
        return np.full(shape=n_weeks, fill_value=self.price_mean)

    def prices_competitor2(self, n_weeks):
        # np.random.seed(0)
        # prices = np.random.normal(loc=self.price_mean, scale=self.price_std, size=n_weeks).astype(int)
        # prices = np.clip(prices, a_min=self.price_min, a_max=self.price_max)

        return np.random.randint(low=self.price_min, high=self.price_max, size=n_weeks)

    @staticmethod
    def get_prices(range_actions, action, prices_competitor, week):
        price_company = range_actions[action]
        price_competitor = prices_competitor[week]
        price_market = (price_company + price_competitor) / 2

        return price_company, price_competitor, price_market

    @staticmethod
    def get_sales(encoder, model_poisson, price_market, price_company, price_competitor, week):
        # Market sales = competitor sales + company sales
        # Company price x company sales = competitor price x competitor sales
        X_price_market = np.array([[price_market]])
        X_week_of_year = encoder.transform(pd.DataFrame(data={'week_of_year': [str(week)]})).values
        X = np.concatenate((X_price_market, X_week_of_year), axis=1)

        sales_market = int(model_poisson.predict(X))
        sales_company = math.ceil(sales_market / (1 + (price_company / price_competitor)))
        sales_competitor = math.floor(sales_market / (1 + (price_competitor / price_company)))

        return sales_market, sales_company, sales_competitor

    def get_profits(self, sales_company, sales_competitor, price_company, price_competitor):
        profit_company = sales_company * (price_company - self.cost_production)
        profit_competitor = sales_competitor * (price_competitor - self.cost_production)

        return profit_company, profit_competitor
