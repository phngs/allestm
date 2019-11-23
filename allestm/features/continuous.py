"""
Module containing continuous input features as well as output targets.
"""
import math
import re
from collections import Counter
from collections import defaultdict

import itertools
from sklearn.base import TransformerMixin

import numpy as np
from sklearn.preprocessing import StandardScaler

from allestm.features.base import NoFitMixin, LengthMixin


class Continuous:
    """
    Using the linear activation function and MAE loss.
    """
    keras_activation = 'linear'
    keras_loss = 'mean_absolute_error'
    lightgbm_objective = 'regression'


class RawValues(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    Base class for directly using the database values without a specified query.
    """
    length = 1
    query = None

    @staticmethod
    def transform(x):
        """
        Just returns the original values from the database.
        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.

        """
        return x


class RsaComplex(RawValues):
    """
    The relative solvent accessibility of a chain in a complex.
    """
    query = 'SELECT acc_complex FROM raw_data WHERE id=? ORDER BY resi'
    #keras_activation = 'sigmoid'


class RsaChain(RawValues):
    """
    The relative solvent accessibility of a chain without the remaining complex.
    """
    query = 'SELECT acc_chain FROM raw_data WHERE id=? ORDER BY resi'
    #keras_activation = 'sigmoid'


class RsaDiff(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    The difference in relative solvent accessibility of the residues with and
    without the remaining complex.
    """
    length = 1
    query = 'SELECT acc_complex, acc_chain FROM raw_data WHERE id=? ORDER BY resi'
    #keras_activation = 'sigmoid'

    @staticmethod
    def transform(x):
        """
        Calcualte the absolute difference in RSA for residues with and without the
        remaining complex.
        Args:
            x: [n, 2]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.

        """
        return np.abs(x[:, 0] - x[:, 1])[:, None]


class PropertiesCont(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    Continuous properties:

    Hydropathy, weight.
    """
    _map = {
        'A': [1.8, 89.094],
        'R': [-4.5, 174.203],
        'N': [-3.5, 132.119],
        'D': [-3.5, 133.104],
        'C': [2.5, 121.154],
        'E': [-3.5, 147.131],
        'Q': [-3.5, 146.146],
        'G': [-0.4, 75.067],
        'H': [-3.2, 155.156],
        'I': [4.5, 131.175],
        'L': [3.8, 131.175],
        'K': [-3.9, 146.189],
        'M': [1.9, 149.208],
        'F': [2.8, 165.192],
        'P': [-1.6, 115.132],
        'S': [-0.8, 105.093],
        'T': [-0.7, 119.119],
        'W': [-0.9, 204.228],
        'Y': [-1.3, 181.191],
        'V': [4.2, 117.148]
    }

    length = 2
    query = 'SELECT seqres FROM raw_data WHERE id=? ORDER BY resi'

    def __init__(self):
        """
        Continuous properties feature.
        """

    def transform(self, x):
        """
        Returns a [n, num_features]-shaped NumPy array converting the amino acid
        letters to feature values.

        Args:
            x: Sequence as [n, 1]-shaped NumPy array.

        Returns:
            [n, num_features]-shaped NumPy array.
        """
        result = np.zeros((len(x), self.length))
        for i, aa in enumerate(x[:, 0]):
            try:
                result[i, :] = self._map[aa]
            except KeyError:
                pass

        return result


class Pssm(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    PSSM matrix calculated from a MSA.
    """
    _aas = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
            'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')  # , '-')
    _aa2index = {aa: index for index, aa in enumerate(_aas)}

    length = len(_aas)
    query = 'SELECT msa FROM alignments where id=?'

    @classmethod
    def transform(cls, x):
        """
        Transforms the [1, 1]-shaped NumPy array to a [n, 20]-shaped one representing
        the PSSM.

        Args:
            x: [1,1]-shaped NumPy array containing the alignment in a3m format as a binary str.
            If the array is longer in any dimension, these are ignored.

        Returns:
            [n, 20]-shaped NumPy array representing the PSSM.
        """
        # Remove lowercase letters (gaps in query), remove headers
        msa = [re.sub('[a-z]', '', row)
               for row in x[0, 0].strip().split('\n') if not row.startswith('>')]

        # Transpose alignment (rows to cols)
        msa = list(map(list, zip(*msa)))

        pssm = np.zeros((len(msa), len(cls._aas)))

        # Count
        background_counter = Counter()
        for col_index, col in enumerate(msa):
            col_counter = Counter({aa: 1 for aa in cls._aas})
            col_counter.update(col)
            background_counter.update(col_counter)

            col_total_aas = sum(
                [v for k, v in col_counter.items() if k in cls._aa2index])
            for aa, aa_index in cls._aa2index.items():
                pssm[col_index, aa_index] = col_counter[aa] / col_total_aas

        # Calculate log ratios and normalize using tanh.
        background_total_aas = sum(
            [v for k, v in background_counter.items() if k in cls._aa2index])
        for col_index in range(len(msa)):
            for aa, aa_index in cls._aa2index.items():
                pssm[col_index, aa_index] /= background_counter[aa] / \
                    background_total_aas
                pssm[col_index, aa_index] = cls._normalize(
                    pssm[col_index, aa_index])

        return pssm

    @staticmethod
    def _normalize(x):
        return 1 / (1 + math.exp(-1 * math.log2(x)))


class ZCoordinates(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    Normalized z-coordinates to a range of -1 to 1.
    """
    length = 1
    query = 'SELECT ca_z FROM raw_data where id=? ORDER BY resi'
    #keras_activation = 'tanh'

    def __init__(self, lower_bound=-25, upper_bound=25):
        """
        Constructor.

        Args:
            lower_bound: Lowest value for the z-coordinate.
            upper_bound: Highest value for the z-coordiante.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _transform_func(self, v):
        v = max(self.lower_bound, min(self.upper_bound, v))
        return -1 + ((v - self.lower_bound) * 2 / (self.upper_bound - self.lower_bound))

    def transform(self, x):
        """
        Transforms the z-coordinate to a value between -1 and 1.

        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.
        """

        return np.vectorize(self._transform_func)(x)

    def _inverse_transform_func(self, v):
        v = self.lower_bound + \
            (((v + 1) * (self.upper_bound - self.lower_bound)) / 2)
        return max(self.lower_bound, min(self.upper_bound, v))

    def inverse_transform(self, x):
        """
        Transforms transformed values back to original values (up to the lower and upper
        bounds).

        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.
        """

        return np.vectorize(self._inverse_transform_func)(x)


class Bfactors(TransformerMixin, LengthMixin, NoFitMixin, Continuous):
    """
    Scale bfactors according to z-score normalization
    """
    length = 1
    query = 'SELECT bfactor FROM raw_data WHERE id=? ORDER BY resi'

    def __init__(self, with_mean=True, with_std=True, copy=True):
        self.standard_scaler = StandardScaler(
            with_mean=with_mean, with_std=with_std, copy=copy)

    def transform(self, x):
        """
        Transform the bfactor values by standard scaling.

        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.
        """
        return self.standard_scaler.fit_transform(x)


class Angles(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    Base class for angle transformation without a specified query.
    """
    length = 1
    query = None
    #keras_activation = 'tanh'

    @staticmethod
    def transform(x):
        """
        Transforms values from -pi to pi into a range of -1 to 1.
        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.

        """
        return np.vectorize(lambda v: max(min(v / math.pi, 1.0), -1.0))(x)

    @staticmethod
    def inverse_transform(x):
        """
        Transforms values from -1 to 1 into a range of -180 to 180 degrees.
        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.

        """
        return np.vectorize(lambda v: max(min(v, 1.0), -1.0) * 180.0)(x)
        # return np.vectorize(lambda v: max(min(v, 1.0), -1.0) * math.pi)(x)


class PhiAngles(Angles, LengthMixin):
    """
    Implementation of the Angles feature including a query for phi angles.
    """
    query = 'SELECT phi FROM raw_data WHERE id=? ORDER BY resi'


class PsiAngles(Angles, LengthMixin):
    """
    Implementation of the Angles feature including a query for psi angles.
    """
    query = 'SELECT psi FROM raw_data WHERE id=? ORDER BY resi'


class Thickness(TransformerMixin, NoFitMixin, LengthMixin, Continuous):
    """
    Returns half thickness of the membrane.
    """
    length = 1
    query = 'SELECT thickness FROM raw_data JOIN proteins USING (id) where id=? ORDER BY resi'

    @staticmethod
    def transform(x):
        """
        Just returns the thicknesses.

        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.

        """
        return x
