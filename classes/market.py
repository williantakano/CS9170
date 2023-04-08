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

    @staticmethod
    def price_market(price_agent, price_competitor):
        return (price_agent + price_competitor) / 2

    @staticmethod
    def get_sales(price_market, price_agent, price_competitor, encoder, model_poisson, week):
        # Market sales = competitor sales + agent sales
        # Agent price x agent sales = competitor price x competitor sales
        X_price_market = np.array([[price_market]])
        X_week_of_year = encoder.transform(pd.DataFrame(data={'week_of_year': [str(week)]})).values
        X = np.concatenate((X_price_market, X_week_of_year), axis=1)

        sales_market = int(model_poisson.predict(X))
        sales_agent = math.ceil(sales_market / (1 + (price_agent / price_competitor)))
        sales_competitor = math.floor(sales_market / (1 + (price_competitor / price_agent)))

        return sales_market, sales_agent, sales_competitor

    def get_profits(self, sales_agent, sales_competitor, price_agent, price_competitor):
        profit_agent = sales_agent * (price_agent - self.cost_production)
        profit_competitor = sales_competitor * (price_competitor - self.cost_production)

        return profit_agent, profit_competitor
