if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
#from sklearn.metrics import mean_squared_error

@transformer
def preprocess(df):
    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    # if fit_dv:
    #     X = dv.fit_transform(dicts)
    # else:
    #     X = dv.transform(dicts)
    dv = DictVectorizer()

    train_dicts = df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = df[target].values
    # y_val = df_val[target].values
    # y_test = df_test[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(lr.intercept_)

    # y_pred = lr.predict(X_val)
    # mean_squared_error(y_val, y_pred, squared=False)
   
    return lr, dv



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'