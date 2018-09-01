import numpy as np
from time import sleep, time

import skfuzzy as fuzz


class NetworkSom:
    """
    The SOM network is a 2-layer neural network that accepts N-dimensional patterns as input and maps them to a set
    of output neurons, which represents the space of the data to be grouped. The output map, which is typically
    two-dimensional, represents the positions of the neurons in relation to their neighbors. The idea is that
    topologically close neurons respond in a similar way to similar inputs. For this, all input-layer neurons are all
    connected to the output neurons.

    SOM Training Algorithm Steps:
        1- Initialize the weights with random values.
        2- Display the input pattern for the network.
        3- Choose the output neuron with the highest activation state (Neurôno "winner").
        4- Update the weights of neighboring neurons to the winning neuron using a learning
        factor (Generally based on neighborhood radius and learning rate).
        5- Reduce the learning factor monotonically (linearly).
        6- Reduce the neighborhood radius monotonically (linearly).
        7- Repeat the steps from step 2 until the weight refreshes are very low.
    """

    def __init__(self, data_input: np, data_output, grid=[10, 10], alpha=0.1, sigma=None):
        dim = data_input.shape[1]
        self.w_nodes = np.random.uniform(-1, 1, [grid[0], grid[1], dim])
        self.alpha0 = alpha
        if sigma is None:
            self.sigma0 = max(grid) / 2.0
        else:
            self.sigma0 = sigma

        self.dataIn = np.asarray(data_input)
        self.grid = grid
        self.data_output = data_output

    def fit(self, max_it=100, verbose=True, analysis=False, time_sleep=0.5, fuzzy=False):
        n_samples = self.dataIn.shape[0]
        m = self.w_nodes.shape[0]  # line
        n = self.w_nodes.shape[1]  # column

        # Processing time
        time_cte = (max_it / np.log(self.sigma0))
        if analysis:
            print('time_cte = ', time_cte)

        time_init = 0
        time_end = 0

        for epc in range(max_it):
            alpha = self.alpha0 * np.exp(-epc / time_cte)
            sigma = self.sigma0 * np.exp(-epc / time_cte)

            if verbose:
                print('Epoch: ', epc, ' - Processing time: ', (time_end - time_init) * (max_it - epc), ' sec')

            time_init = time()

            for k in range(n_samples):

                # Winning Node
                mat_dist = NetworkSom.distance_calculation(self.dataIn[k, :], self.w_nodes)
                pos_win = NetworkSom.get_win_node_pos(mat_dist)

                for i in range(m):
                    for j in range(n):
                        # Distance between two nodes
                        d_node = NetworkSom.get_distance_nodes([i, j], pos_win)

                        # neighborhood region
                        h = np.exp((-d_node ** 2) / (2 * sigma ** 2)) if not fuzzy else NetworkSom.relevance()

                        # update weights
                        delta_w = (alpha * h * (self.dataIn[k, :] - self.w_nodes[i, j, :]))
                        self.w_nodes[i, j, :] += delta_w

                        if analysis:
                            NetworkSom.print_atalysis(epc, k, alpha, sigma, h, (pos_win[0], pos_win[1]), (i, j), d_node,
                                                      delta_w, self.w_nodes[i, j, :], (self.w_nodes[i, j, :] + delta_w))
                            sleep(time_sleep)

            time_end = time()

    @staticmethod
    def print_atalysis(epc, k, alpha, sigma, h, w_position: tuple, current_node: tuple, d_node, delta_w,
                       pre_node: float,
                       pos_node: float):
        """
        Print analysis using variables
        """
        line = f'Epoch = {epc} \n Sample = {k} \n ------------------------------- \n alpha = {alpha} \n \ ' \
               f'sigma = {sigma} \n h = {h} '
        line += f'------------------------------- Node winning = {w_position} \n Current node = {current_node} \n ' \
                f'Distance between nodes = {d_node} \n '
        line += f'deltaW = {delta_w} \n wNode prev {pre_node} \n wNode after = {pos_node}\n'
        print(line)
        return line

    @staticmethod
    def distance_calculation(a, b):
        """
        Calculates the distance between the input and the weights
        """
        return np.sqrt(np.sum((a - b) ** 2, 2, keepdims=True))

    @staticmethod
    def get_distance_nodes(n1, n2):
        """
        Calculates the distance between two nodes
        """
        n1 = np.asarray(n1)
        n2 = np.asarray(n2)
        return np.sqrt(np.sum((n1 - n2) ** 2))

    @staticmethod
    def get_win_node_pos(dists):
        """
        Returns position of winning node
        :param dists:
        :return:
        """
        arg = dists.argmin()
        m = dists.shape[0]
        return arg // m, arg % m

    def get_centroid(self, data):
        """
        Returns centroids of data input
        """
        data = np.asarray(data)
        n = data.shape[0]
        centroids = []

        for k in range(n):
            mat_dist = self.distance_calculation(data[k, :], self.w_nodes)
            centroids.append(self.get_win_node_pos(mat_dist))

        return centroids

    def save_model(self, file_name='model.csv'):
        """
        Saves the updated weights obtained after training
        """
        np.savetxt(file_name, self.w_nodes)

    def load_model(self, file_name):
        """
        Loads trained model for prediction
        """
        self.w_nodes = np.loadtxt(file_name)

    def predict(self, sample):
        """
        Returns the position of the neuron that responded to the new example
        :param sample:
        :return: tuple y, x
        """
        x, y = self.get_win_node_pos(self.distance_calculation(sample, self.w_nodes))
        return y, x

    @staticmethod
    def relevance(distance: float):
        """
        Given the distance, it returns a degree of pertinence. Necessary to apply the fuzzy technique,
        in adjusting the weights. For distance 0, returns the degree of relevance 1
        """
        universe = np.arange(0, 100, 1)
        return fuzz.interp_membership(universe, fuzz.gaussmf(universe, 0.0, 5), distance)
