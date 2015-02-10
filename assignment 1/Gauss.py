# -*- encoding-utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D


class Gauss:

    @staticmethod
    def draw_gauss_one(mean, deviation):
        N = lambda x: (1 / (2 * np.pi * deviation ** 2) ** 0.5) * np.exp((-1 / (2 * deviation ** 2)) * (x - mean) ** 2)

        xs = np.arange(-10., 10., 0.1)
        ys = map(lambda x: N(x), xs)

        plt.figure('Guassian distribution')
        plt.plot(xs, ys)
        plt.axis([-5, 5, -5, 5])
        plt.show()

    @staticmethod
    def sample_gauss(mean, covariance, size):
        sample = []
        for i in range(0, size):
            z = np.random.randn(2, 1)
            sample += [mean + np.dot(np.linalg.cholesky(covariance), z)]
        return sample

    @staticmethod
    def draw_gauss_multi(mean, covariance, likelihood_mean_function=None):

        sample = Gauss.sample_gauss(mean, covariance, 100)

        xs = map(lambda x: x.item(0), sample)
        ys = map(lambda x: x.item(1), sample)

        fig = plt.figure('Guassian distribution')
        plt.plot(xs, ys, 'ro')
        plt.plot(mean.item(0), mean.item(1), 'b^', label='Distribution mean')

        if likelihood_mean_function is not None:
            sample_mean = likelihood_mean_function(sample)

            # plot maximum likelihood sample mean, and the deviation from distrubition mean
            plt.plot(sample_mean.item(0), sample_mean.item(1), 'bo', label='Sample mean')
            plt.plot(
                [mean.item(0), sample_mean.item(0)],
                [mean.item(1), sample_mean.item(1)],
                'b-',
                label="Mean deviation: " + str(np.linalg.norm((mean-sample_mean)))
            )

        return fig

    @staticmethod
    def draw_likelihood(mean, covariance):

        likelyhood_mean = lambda N: 1/len(N) * sum(N)
        fig = Gauss.draw_gauss_multi(mean, covariance, likelyhood_mean)
        plt.legend(loc='upper left')
        plt.show()

    @staticmethod
    def draw_eigenvectors(distribution_mean, distribution_covariance):

        sample = Gauss.sample_gauss(distribution_mean, distribution_covariance, 100)

        likelyhood_mean = lambda N: 1/len(N) * sum(N)
        sample_mean = likelyhood_mean(sample)

        likehood_covariance = lambda N: (
            1/len(N) *
            sum(map(
                lambda x: ((x - sample_mean)*(x - sample_mean).T),
                N
            ))
        )

        sample_covariance = likehood_covariance(sample)

        eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)

        scale_eigenvector = lambda eigenvalue, eigenvector: \
            distribution_mean + np.sqrt(eigenvalue)*np.linalg.norm(eigenvector)

        translated_eigenvector1 = scale_eigenvector(eigenvalues[0], eigenvectors[0])
        translated_eigenvector2 = scale_eigenvector(eigenvalues[1], eigenvectors[1])

        fig = Gauss.draw_gauss_multi(distribution_mean, distribution_covariance, likelyhood_mean)

        plt.plot(
            translated_eigenvector1[0],
            translated_eigenvector1[1], 'bx', label='translated eigenvector 1'
        )
        plt.plot(
            translated_eigenvector2[0],
            translated_eigenvector2[1], 'bx', label='translated eigenvector 2'
        )

        plt.legend(loc='upper left')
        plt.show()