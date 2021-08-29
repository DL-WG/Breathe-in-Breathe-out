from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import inv
from Dataset import *

"""
Implementation of the Optimal Interpolation algorithm
"""


class OptInterpolation:
    def __init__(self, observations, simulations, idx):
        """
        :param observations: observed values
        :param simulations: modelled values
        :param idx: index
        """
        self.scaler = StandardScaler()
        self.sim = self.scaler.fit_transform(simulations)
        self.obs = self.scaler.transform(observations)
        self.idx = idx
        self.C = np.identity(1)
        self.R = self.covariance_matrix(self.obs.T - np.dot(self.C, self.sim.T))
        self.P = self.covariance_matrix(np.array(self.sim).T)
        self.range = range(len(self.obs))

    def kalman_gain(self, P, C, R):
        """
        Method for the Kalman gain calculation
        :param P:
        :param C:
        :param R:
        :return: Kalman gain matrix K
        """
        tempInv = inv(R + np.dot(C, np.dot(P, C.transpose())))
        res = np.dot(P, np.dot(C.transpose(), tempInv))

        return res

    def update_prediction(self, x, K, C, y):
        """
        Method that implements analysis step of the OI --
        assimilates modelled value with the observations
        :param x: modelled value
        :param K: Kalman gain
        :param C: linear observation operator (equal to identity matrix in our case)
        :param y: observed value
        :return: updated value
        """
        cx = np.dot(C, x)
        inovation = np.dot(K, (y - cx))
        res = x + inovation

        return res

    def covariance_matrix(self, X):
        """
        Method that calculates cov matrix of data
        :return: cov matrix
        """
        means = np.array([np.mean(X, axis=1)]).transpose()
        # print(means)
        dev_matrix = X - means
        # print(dev_matrix)
        #     print(X.shape)
        res = np.dot(dev_matrix, dev_matrix.transpose())  # /(X.shape[1]-1)

        return res

    def assimilate(self, R=None, plot=False, plot_length=120,
                   labels=['Optimal interpolation', 'Observations', 'Simulations']):
        """
        Method responsible for the full assimilation process
        :param R:
        :param plot: plot the assimilation graph (modelled vs observed vs assimilated values)
        :param plot_length: length of the plot
        :param labels: graph axis names
        :return:
        """
        if R is None:
            R = self.covariance_matrix(self.obs.T - np.dot(self.C, self.sim.T))

        self.updated = []
        for i in self.range:
            self.updated.append(
                self.update_prediction(self.sim[i], self.kalman_gain(self.P, self.C, R), self.C, self.obs[i]))

        self.updated = self.scaler.inverse_transform(self.updated)

        if plot:
            self.plot(plot_length, labels=labels)

        return self.updated

    def plot(self, plot_length, labels):
        """
        Method responsible for the assimilation plotting (modelled vs observed vs assimilated values)
        :param plot_length: length of the plot
        :param labels: graph axis names
        """
        obs = self.scaler.inverse_transform(self.obs)
        sim = self.scaler.inverse_transform(self.sim)

        plt.plot(pd.DataFrame(self.updated, index=self.idx)[:plot_length], 'b--', label=labels[0])
        plt.plot(pd.DataFrame(obs, index=self.idx)[:plot_length], 'r-', label=labels[1])
        plt.plot(pd.DataFrame(sim, index=self.idx)[:plot_length], 'g-', label=labels[2])

        plt.legend()
        plt.rcParams["figure.figsize"] = (18, 10)

        plt.show()

        return
