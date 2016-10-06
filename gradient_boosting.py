import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer

#reading and formatting part 1

def read_and_clean(filename):
    df = pd.read_csv(filename)
    df.columns = np.array([(str(x)).lower() for x in df.columns])
    df = df[['modelid', 'auctioneerid', 'yearmade', 'machinehourscurrentmeter', \
            'saledate', 'saleprice', 'state', 'productgroup']]
    df['sale_month'] = pd.to_datetime(df['saledate'], infer_datetime_format = True).dt.month
    df['sale_year'] = pd.to_datetime(df['saledate'], infer_datetime_format = True).dt.year
    df.drop('saledate', axis = 1, inplace = True)
    df.saleprice = df.saleprice.astype(float)

    #formatting part 2
    df_dummies = pd.get_dummies(df[['state', 'modelid']])
    df = pd.concat([df_dummies, df], axis=1).values
    print df.head()


if __name__ == '__main__':
    read_and_clean('data/Train.csv')
