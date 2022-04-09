from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
DATA_PATH = 'C:/Users/omrys/Git/IML.HUJI/datasets/house_prices.csv'
OUTPUT_PATH = 'C:/Users/omrys/Git/IML.HUJI/exercises/ex2_graphs'


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename, error_bad_lines=False)
    return __preprocess(df)


def __preprocess(df):
    df.dropna(0, inplace=True)
    # positive values
    above_zero = {'condition', 'grade', 'yr_built', 'sqft_living', 'sqft_lot',
                  'sqft_lot15', 'sqft_living15', 'price'}
    for feature in above_zero:
        df = df[(df[feature] > 0)]

    prices = df['price']
    df['yr_renovated'] = np.where(df['yr_renovated'] == 0,
                                  df['yr_built'], df['yr_renovated'])

    # drop
    to_drop = {'price', 'date', 'id', 'zipcode'}
    for feature in to_drop:
        df.drop(feature, inplace=True, axis=1)

    return df, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col_name in X.columns:
        col = X[col_name]
        pears = __pearson_correlation(col, y)
        fig = px.scatter(x=col, y=y, labels={'x': col_name, 'y': "Price"},
                         title="Pearson Correlation: " + str(pears))
        fig.write_image(output_path + "/" + col_name + ".jpeg")


def __pearson_correlation(x, y):
    return (np.cov(x, y)[0][1] / (np.std(x) * np.std(y)))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    mat, vec = load_data(DATA_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(mat, vec, OUTPUT_PATH)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(mat, vec, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lin_reg = LinearRegression()
    p_range = np.arange(0.1, 1.01, 0.01)
    mean_loss = []
    std_loss = []
    for p in p_range:
        p_loss = []
        for i in range(10):
            train_Xi = train_X.sample(frac=p)
            train_yi = train_y[train_Xi.index]
            lin_reg.fit(train_Xi.to_numpy(), train_yi.to_numpy())
            p_loss.append(lin_reg.loss(test_X.to_numpy(), test_y.to_numpy()))
        mean_loss.append(np.mean(p_loss))
        std_loss.append(np.std(p_loss))
    down_bound = np.array(mean_loss) - 2 * np.array(std_loss)
    up_bound = np.array(mean_loss) + 2 * np.array(std_loss)
    fig = go.Figure([
        go.Scatter(x=p_range, y=mean_loss, mode='markers', name='Mean Loss'),
        go.Scatter(x=p_range, y=up_bound, line=dict(color='lightblue'),
                   fill='tonexty', showlegend=False),
        go.Scatter(x=p_range, y=down_bound, line=dict(color='lightblue'),
                   fill='tonexty', showlegend=False)
    ])
    fig.write_image(OUTPUT_PATH + "/q4.jpeg")

