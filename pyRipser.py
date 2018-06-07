import os
import shutil
import uuid
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

try:
    is_ipython_notebook = get_ipython
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def serialize(arrays: list, directory: str):
    """Serialize numpy array to given dir in csv format. Supports single array and list of arrays.
    Returns (list of) name(s) of the serialized file

    :param arrays: single or list of numpy arrays
    :type arrays: list of np.ndaarray
    :param directory: target directory
    :type directory: str
    :return: list of file names of serialized arrays
    :rtype: list of str
    """
    filenames = []
    for array in tqdm(arrays, desc='Serializing to disk'):
        fname = os.path.join(directory, str(uuid.uuid4()))
        np.savetxt(fname, array)
        filenames.append(fname)

    return filenames


class Ripser(object):
    """Interface to the Ripser module"""

    def __init__(self, ripser_path: str, dim=1):
        """
        Initialize the Ripser interface

        :param ripser_path: path to the Ripser exectuable
        :type ripser_path: str
        :param dim: dimensions up to which the persistence diagrams are computed
        :type dim: int
        """
        self.cmd_path = ripser_path
        self.dim = dim

    @staticmethod
    def parse_output(output):
        # print(output.decode('utf-8').split())
        output = output.decode('utf-8').split('\n')


        persistence_pairs = {}
        for line in output[2:]:
            line = line.rstrip()

            if line.startswith('persistence intervals in dim'):
                dim = line[-2]
                persistence_pairs[dim] = []
                continue

            line = line.replace('[', '').replace(')', '').split(',')
            if len(line) ==  2 and line[1] != ' ':
                persistence_pairs[dim].append([float(line[0]), float(line[1])])

        persistence_pairs = {k:np.array(v) for k, v in persistence_pairs.items()}

        return PersistenceDiagram(persistence_pairs)

    def call(self, filename):
        result = subprocess.run([self.cmd_path, '--dim', str(self.dim), '--format', 'distance', filename],
                                stdout=subprocess.PIPE)
        return self.parse_output(result.stdout)

    def compute_pd(self, dist_matrices: Union[np.ndarray, list]):
        """
        Compute the persistence diagram from distance matrix using Ripser.

        `dist_matrix` can be either a single numpy 2d array or list of arrays.

        :param dist_matrix: single or a list of distance matrices
        :type dist_matrix: np.ndarray or list
        :return: PersistenceDiagram object or a list of PersistenceDiagram objects
        :rtype: PersistenceDiagram or list
        """

        # Convert to a one element list
        if isinstance(dist_matrices, np.ndarray):
            dist_matrices = [dist_matrices]

        # Make sure that provided matrix is a valid distance matrix
        # assert (dist_matrices >= 0).all(), "Not a valid distance matrix (contains negative values)"

        # Create a temporary directory for serializing the matrices
        temp_dir = '.tmp_serd'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=False)
        file_list = serialize(dist_matrices, temp_dir)

        pers_diags = []
        for mat_file in tqdm(file_list, desc='Running Ripser'):
            pers_diags.append(self.call(mat_file))

        shutil.rmtree(temp_dir)

        return pers_diags


class PersistenceDiagram(object):
    """Represent a persistence diagram"""

    def __init__(self, persistence_pairs):
        self.dims = sorted(persistence_pairs.keys())
        self.pairs = persistence_pairs

    def show_diagram(self):
        """Plot the persistence diagram"""
        fig, ax = plt.subplots()

        for idx, dim in enumerate(self.pairs):
            pair = self.pairs[dim]
            ax.scatter(pair[:, 0], pair[:, 1], c='C{}'.format(idx))

        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]

        # now plot both limits against each other
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.show()

    def to_np_array(self):
        """
        Convert the persistence diagram into a 3-column numpy array. The first column
        contains the dimension and the next 2 columns contain the x and y coordinate
        of the point

        :return: numpy array representation of the persistence diagram
        :rtype: numpy.ndarray
        """
        array = []
        for idx, dim in enumerate(self.pairs):
            dim_pd = self.pairs[dim]
            temp = np.zeros((len(dim_pd), 3))
            temp[:, 0] = dim
            temp[:, 1:] = dim_pd
            array.append(temp)
        return np.vstack(array)