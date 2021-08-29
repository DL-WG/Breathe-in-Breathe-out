import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

"""
Class responsible for the correlation measures
between two sets
"""


class PearsonCorr:
    def __init__(self, x, y):
        """
        Object initialised with the two sets that need to be compared
        """
        self.x = np.array(x)
        self.y = np.array(y)

    def correlation(self):
        """
        Pearson's correlation coefficient calculation
        """
        self.pear_corr, _ = pearsonr(self.x, self.y)

        return self.pear_corr

    def plot_corr(self, labels, title=None, years=[2007, 2010, 2013]):
        """
        Method responsible for the correlation graph plotting
        :param labels: axis labels
        :param title: graph title
        :param years: labels of the points in the graph
        """
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, color='r', s=600)

        m, b = np.polyfit(self.x, self.y, 1)

        plt.plot(self.x, m * self.x + b, linewidth=10)

        for i, txt in enumerate(years):
            ax.annotate(txt, (self.x[i], self.y[i]), fontsize=36)

        plt.xlabel(labels[0], fontdict={'size': 36})
        plt.ylabel(labels[1], fontdict={'size': 36})

        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        if title is not None:
            plt.title(title, {'fontsize': 40})

        plt.show()

        return
