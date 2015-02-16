# -*- encoding-utf-8 -*-
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D


class Gauss(object):

    @staticmethod
    def sample_gauss(mean, covariance, size):
        sample = []
        for i in range(0, size):
            z = np.random.randn(2, 1)
            sample += [mean + np.dot(np.linalg.cholesky(covariance), z)]
        return sample

    @staticmethod
    def draw_gauss_one(mean, deviation):
        N = lambda x: (1 / (2 * np.pi * deviation ** 2) ** 0.5) * np.exp((-1 / (2 * deviation ** 2)) * (x - mean) ** 2)

        xs = np.arange(-10., 10., 0.1)
        ys = map(lambda x: N(x), xs)

        fig = plt.figure('Guassian distribution')
        plt.plot(xs, ys)
        plt.axis([-5, 5, -5, 5])

        return fig

    @staticmethod
    def draw_gauss_multi(mean, covariance, likelihood_mean_function=None, sample=None, label='Label1'):

        if sample is None:
            sample = Gauss.sample_gauss(mean, covariance, 100)

        xs = map(lambda x: x.item(0), sample)
        ys = map(lambda x: x.item(1), sample)

        color = np.random.random(3)

        fig = plt.figure('Guassian distribution')
        plt.plot(xs, ys, 'o', color=color)
        plt.plot(mean.item(0), mean.item(1), '^', color=np.random.random(3), label=label + ' distribution mean: ' + str((mean.item(0), mean.item(1))))

        if likelihood_mean_function is not None:
            sample_mean = likelihood_mean_function(sample)

            # plot maximum likelihood sample mean, and the deviation from distrubition mean
            plt.plot(sample_mean.item(0), sample_mean.item(1), 's', color=color, label=label + ' mean: ' + str((sample_mean.item(0), sample_mean.item(1))))
            plt.plot(
                [mean.item(0), sample_mean.item(0)],
                [mean.item(1), sample_mean.item(1)],
                '-',
                color=color,
                label=label + " mean deviation: " + str(np.linalg.norm((mean-sample_mean)))
            )

        return fig

    @staticmethod
    def draw_likelihood(mean, covariance):

        likelyhood_mean = lambda N: 1/len(N) * sum(N)
        fig = Gauss.draw_gauss_multi(mean, covariance, likelyhood_mean)
        plt.legend(loc='upper left')

        return fig

    @staticmethod
    def draw_eigenvectors(distribution_mean, distribution_covariance, sample=None, label='Label1'):

        if sample is None:
            sample = Gauss.sample_gauss(distribution_mean, distribution_covariance, 100)

        likelyhood_mean = lambda N: 1/len(N) * sum(N)
        # maximum likelihood sample mean
        sample_mean = likelyhood_mean(sample)

        likehood_covariance = lambda N: (
            1/len(N) *
            sum(map(
                lambda x: np.dot((x - sample_mean), (x - sample_mean).T),  # should this be dot or product?
                N
            ))
        )

        # maximum likelihood sample covariance
        sample_covariance = likehood_covariance(sample)

        # eigenvalues are found by normalising the covariance
        eigenvalues, eigenvectors = np.linalg.eig(sample_covariance)

        scale_eigenvector = lambda eigenvalue, eigenvector: \
            distribution_mean + np.sqrt(eigenvalue)*eigenvector

        translated_eigenvector1 = scale_eigenvector(eigenvalues[0], eigenvectors[:, 0:1])
        translated_eigenvector2 = scale_eigenvector(eigenvalues[1], eigenvectors[:, 1:2])

        fig = Gauss.draw_gauss_multi(distribution_mean, distribution_covariance, likelyhood_mean, sample, label)

        color = np.random.random(3)

        plt.plot(
            translated_eigenvector1[0],
            translated_eigenvector1[1], 'x', color=color
        )
        plt.plot(
            translated_eigenvector2[0],
            translated_eigenvector2[1], 'x', color=color
        )

        # plot line from mean to eigenvectors
        plt.plot(
                [sample_mean.item(0), translated_eigenvector1.item(0)],
                [sample_mean.item(1), translated_eigenvector1.item(1)],
                '-',
                color=color,
                label=label + ' Translated eigenvector 1: ' + str(np.linalg.norm(sample_mean - translated_eigenvector1))
            )

        # plot line from mean to eigenvectors
        plt.plot(
                [sample_mean.item(0), translated_eigenvector2.item(0)],
                [sample_mean.item(1), translated_eigenvector2.item(1)],
                '-',
                color=color,
                label=label + ' Translated eigenvector 2: ' + str(np.linalg.norm(sample_mean - translated_eigenvector2))
            )

        # ensure that eigenvectors are orthogonal
        assert eigenvectors[:, 0:1][0] * eigenvectors[:, 0:1][1] \
              + eigenvectors[:, 1:2][0] * eigenvectors[:, 1:2][1] == 0

        plt.legend(loc='upper left')

        return translated_eigenvector1, translated_eigenvector2

    @staticmethod
    def draw_rotated_covariance(distribution_mean, distribution_covariance):

        sample_size = 10000

        sample = Gauss.sample_gauss(distribution_mean, distribution_covariance, sample_size)

        likelyhood_mean = lambda N: 1/len(N) * sum(N)
        # maximum likelihood sample mean
        sample_mean = likelyhood_mean(sample)

        likehood_covariance = lambda N: (
            1/len(N) *
            sum(map(
                lambda x: np.dot((x - sample_mean), (x - sample_mean).T),  # should this be dot or product?
                N
            ))
        )

        sample_covariance = likehood_covariance(sample)

        translated_eigenvectors = Gauss.draw_eigenvectors(distribution_mean, distribution_covariance, sample, '0 degrees')
        angle = Gauss.find_angle(translated_eigenvectors[0])[0]

        rotate_covariance_30 = Gauss.rotate_covariance((1/3)*np.pi/2, sample_covariance)
        rotate_covariance_60 = Gauss.rotate_covariance((2/3)*np.pi/2, sample_covariance)
        rotate_covariance_90 = Gauss.rotate_covariance(np.pi/2, sample_covariance)
        rotate_x_axis_covariance = Gauss.rotate_covariance(angle, sample_covariance)

        sample_30 = Gauss.sample_gauss(distribution_mean, rotate_covariance_30, sample_size)
        sample_60 = Gauss.sample_gauss(distribution_mean, rotate_covariance_60, sample_size)
        sample_90 = Gauss.sample_gauss(distribution_mean, rotate_covariance_90, sample_size)
        sample_angle = Gauss.sample_gauss(distribution_mean, rotate_x_axis_covariance, sample_size)

        Gauss.draw_eigenvectors(distribution_mean, rotate_covariance_30, sample_30, '30 degrees')
        Gauss.draw_eigenvectors(distribution_mean, rotate_covariance_60, sample_60, '60 degrees')
        Gauss.draw_eigenvectors(distribution_mean, rotate_covariance_90, sample_90, '90 degrees')
        Gauss.draw_eigenvectors(distribution_mean, rotate_x_axis_covariance, sample_angle, '0 degrees rotated along the x-axis')

        plt.legend(loc='upper left')
        plt.show()

    @staticmethod
    def rotate_covariance(phi, covariance):
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        return np.dot(np.dot(np.linalg.inv(R), covariance), R)

    @staticmethod
    def find_angle(vector):
        return np.arccos(vector[1]/(np.linalg.norm(vector)))