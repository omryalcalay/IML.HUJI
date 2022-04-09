from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"

NO_OF_SAMPLES = 1000
MU = 10
VAR = 1


def test_univariate_gaussian():
    uni = UnivariateGaussian()
    # Question 1 - Draw samples and print fitted model
    q1arr = np.random.normal(MU, VAR, NO_OF_SAMPLES)
    uni.fit(q1arr)
    print(f"({uni.mu_}, {uni.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    n_samples = [[], []]
    for i in range(10, NO_OF_SAMPLES + 1, 10):
        n_samples[0].append(i)
        uni.fit(q1arr[:i])
        n_samples[1].append(np.abs(uni.mu_ - MU))
    plt.figure(1)
    plt.scatter(n_samples[0], n_samples[1])
    plt.xlabel('Number of samples')
    plt.ylabel('Distance between the estimated and true expectation')
    plt.title('Abs dist of estimated and true expectation as a'
              'function of no. of samples')

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.figure(2)
    plt.scatter(q1arr, uni.pdf(q1arr))
    plt.xlabel('Sample value')
    plt.ylabel('PDF value')
    plt.title('PDF as a function of sample')

    plt.show()


def test_multivariate_gaussian():
    multi = MultivariateGaussian()

    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma = np.matrix([[1, 0.2, 0, 0.5],
                       [0.2, 2, 0, 0],
                       [0, 0, 1, 0],
                       [0.5, 0, 0, 1]])
    q4arr = np.random.multivariate_normal(mu, sigma, NO_OF_SAMPLES)
    multi.fit(q4arr)
    print(multi.mu_)
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    data = np.empty((200, 200))
    for i in range(f1.size):
        for j in range(f3.size):
            cur_mu = np.array([f1[i], 0, f3[j], 0])
            data[i][j] = multi.log_likelihood(cur_mu, sigma, q4arr)
    fig = go.Figure(data=go.Heatmap(x=f1, y=f3, z=data))
    fig.update_layout(title='Log-Likelihood heatmap', xaxis_title='f1',
                      yaxis_title='f3')
    # fig.show()


    # Question 6 - Maximum likelihood
    cord = np.unravel_index(np.argmax(data), data.shape)
    a=f1[cord[0]]
    b=f3[cord[1]]
    print(format(a), ".3f")
    print(format(b), ".3f")

def quiz():
    x = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
                  -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    uni = UnivariateGaussian()
    print(uni.log_likelihood(1,1,x))
    print(uni.log_likelihood(10,1,x))



if __name__ == '__main__':
    np.random.seed(0)
    quiz()
    # test_univariate_gaussian()
    # test_multivariate_gaussian()
