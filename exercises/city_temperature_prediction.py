import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

DATA_PATH = 'C:/Users/omrys/Git/IML.HUJI/datasets/City_Temperature.csv'
OUTPUT_PATH = 'C:/Users/omrys/Git/IML.HUJI/exercises/ex2_graphs/part2'


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'], error_bad_lines=False)
    return __preprocess(df)


def __preprocess(df):
    df.dropna(0, inplace=True)
    df = df[(df['Temp'] > -71)]
    df['dayofyear'] = df['Date'].dt.dayofyear
    df.drop('Date', inplace=True, axis=1)

    return df


def __temps_doy_plot(df):
    df = df.copy()
    df['Year'] = df['Year'].astype(str)
    fig = px.scatter(x=df['dayofyear'], y=df['Temp'], color=df['Year'],
                     labels={'x': "Day of year", 'y': "Temperature"},
                     title="Temperature as a function of day of year (Israel)")
    fig.write_image(OUTPUT_PATH + "/israel_temps.jpeg")


def __months_bar_plot(df):
    df = df.copy()
    std = df.groupby('Month').Temp.agg(np.std)
    fig = px.bar(std, x=std.index, y=std.values, labels={'x': "Month",
                                                         'y': "STD"},
                 title="STD as a function of month")
    fig.write_image(OUTPUT_PATH + "/months_bar_plot.jpeg")


def __country_comparison(df):
    df = df.copy()
    g = df.groupby(['Month', 'Country'])
    avg = g.mean().reset_index()
    std = g.std().reset_index()
    fig = px.line(avg, x='Month', y='Temp', error_y=std['Temp'], color='Country',
                  labels={'x': "Month", 'y': "Average Temperature"},
                  title="Average Temperature as a function of month")

    fig.write_image(OUTPUT_PATH + "/country_comparison.jpeg")


def __k_plot(df):
    df = df.copy()
    df.sample(frac=1)
    temps = df['Temp']
    df.drop('Temp', inplace=True, axis=1)
    train_X, train_y, test_X, test_y = split_train_test(df, temps, .75)
    k_loss = []
    k_range = range(1, 11)
    for k in k_range:
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X['dayofyear'].to_numpy(), train_y.to_numpy())
        k_loss.append(round(
            poly_fit.loss(test_X['dayofyear'].to_numpy(), test_y.to_numpy()),
            2))
        print("Error for k=" + str(k) + ": " + str(k_loss[k - 1]))
    fig = px.bar(x=k_range, y=k_loss, labels={'x': "k", 'y': "Error"},
                 title="Error as a function of k")
    fig.write_image(OUTPUT_PATH + "/k_comparison.jpeg")


def __country_eval(df, israel):
    poly_fit = PolynomialFitting(k=5)
    poly_fit.fit(israel['dayofyear'], israel['Temp'])
    errors = []
    countries = df['Country'].unique()
    countries = np.delete(countries, np.where(countries == 'Israel')[0],
                          axis=0)
    for country in countries:
        cur = df[df['Country'] == country]
        l = poly_fit.loss(cur['dayofyear'], cur['Temp'])
        errors.append(l)
    fig = px.bar(x=countries, y=errors,
                 labels={'x': 'Countries', 'y': "Error"},
                 title="Error by country")
    fig.write_image(OUTPUT_PATH + "/country_evaluation.jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(DATA_PATH)

    # Question 2 - Exploring data for specific country
    isr_df = data[data['Country'] == 'Israel']
    __temps_doy_plot(isr_df)
    __months_bar_plot(isr_df)

    # Question 3 - Exploring differences between countries
    __country_comparison(data)

    # Question 4 - Fitting model for different values of `k`
    __k_plot(isr_df)

    # Question 5 - Evaluating fitted model on different countries
    __country_eval(data, isr_df)
