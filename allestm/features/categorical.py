"""
Module containing categorical input features as well as output targets.
"""
from sklearn.base import TransformerMixin
import numpy as np

from allestm.features.base import NoFitMixin, LengthMixin


class Categorical:
    """
    Using the softmax activation function and the categorical crossentropy loss.
    """
    keras_activation = 'softmax'
    keras_loss = 'categorical_crossentropy'
    lightgbm_objective = 'multiclass'


class PropertiesCat(TransformerMixin, NoFitMixin, LengthMixin, Categorical):
    """
    Categorical properties:

    Class, Polarity, Charge, Asparagine or aspartic acid, Glutamine or glutamic acid,
    Leucine or Isoleucine, Hydrophobic, Aromatic, Aliphatic (non-aromatic), Small,
    Hydrophilic
    """
    _map = {
        'A': [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        'R': [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
        'N': [2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        'D': [3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        'C': [4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'E': [3, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        'Q': [2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        'H': [5, 2, 2, 0, 0, 0, 0, 1, 0, 0, 1],
        'I': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        'L': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        'K': [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
        'M': [4, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        'F': [5, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        'P': [6, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        'S': [7, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        'T': [7, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'W': [5, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        'Y': [5, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        'V': [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
    }
    _feature_lengths = [8, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2]

    length = sum(_feature_lengths)
    query = 'SELECT seqres FROM raw_data WHERE id=? ORDER BY resi'

    def __init__(self, onehot=True):
        """
        Categorical properties feature.

        Args:
            onehot: Transform the amino acid index to one-hot encoding.
        """
        self.onehot = onehot

    def transform(self, x):
        """
        If onehot=True, transforms the [n, 1]-shaped NumPy array containing the sequence to a [n, 20]-shaped
        NumPy array representing the one-hot encoded categorical features.

        Otherwise returns a [n, num_features]-shaped NumPy array converting the amino acid
        letters to feature values.

        Args:
            x: Sequence as [n, 1]-shaped NumPy array.

        Returns:
            [n, 20]-shaped NumPy array or [n, num_features]-shaped NumPy array.
        """
        if self.onehot:
            result = np.zeros((len(x), sum(self._feature_lengths)))
            for i, aa in enumerate(x[:, 0]):
                start = 0
                for feature_index, feature_length in enumerate(self._feature_lengths):
                    try:
                        onehot_index = start + self._map[aa][feature_index]
                        result[i, onehot_index] = 1.
                    except KeyError:
                        pass
                    start += feature_length
        else:
            result = np.zeros((len(x), len(self._feature_lengths)))
            for i, aa in enumerate(x[:, 0]):
                try:
                    result[i, :] = self._map[aa]
                except KeyError:
                    pass

        return result


class Sequence(TransformerMixin, NoFitMixin, LengthMixin, Categorical):
    """
    Sequence one-hot encoded.
    """
    _aas = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')
    _aa2index = {aa: index for index, aa in enumerate(_aas)}
    _index2aa = {index: aa for index, aa in enumerate(_aas)}

    length = len(_aas)
    query = 'SELECT seqres FROM raw_data WHERE id=? ORDER BY resi'

    def __init__(self, onehot=True):
        """
        Sequence feature.

        Args:
            onehot: Transform the amino acid index to one-hot encoding.
        """
        self.onehot = onehot

    def transform(self, x):
        """
        If onehot=True, transforms the [n, 1]-shaped NumPy array containing the sequence to a [n, 20]-shaped
        NumPy array representing the one-hot encoded sequence.

        Otherwise returns a [n, 1]-shaped NumPy array converting the amino acid
        letters to indices.

        Args:
            x: Sequence as [n, 1]-shaped NumPy array.

        Returns:
            [n, 20]-shaped NumPy array or [n, 1]-shaped NumPy array.
        """
        if self.onehot:
            result = np.zeros((len(x), len(self._aas)))
            for i, aa in enumerate(x[:, 0]):
                try:
                    result[i, self._aa2index[aa]] = 1.
                except KeyError:
                    pass
        else:
            result = np.zeros((len(x), 1))
            for i, aa in enumerate(x[:, 0]):
                try:
                    result[i, 0] = self._aa2index[aa]
                except KeyError:
                    pass

        return result

    def inverse_transform(self, x):
        """
        If onehot=True, transforms the [n, 20]-shaped NumPy array containing the one-hot encoded
        sequence to a [n, 1]-shaped NumPy array representing sequence.

        Otherwise returns a [n, 1]-shaped NumPy array converting the amino acid
        indices to letters.

        Args:
            x: One-hot encoded sequence as [n, 20]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.
        """
        if self.onehot:
            return np.array([[self._index2aa[r]] for r in x.argmax(axis=-1)])
        else:
            result = np.array((len(x), 1), dtype=str)
            for i, aa in enumerate(x[:, 0]):
                try:
                    result[i, 0] = self._index2aa[aa]
                except KeyError:
                    pass
            return result


class SecStruc(TransformerMixin, NoFitMixin, LengthMixin, Categorical):
    """
    Secondary structure,
    i.e.
    - [1, 0, 0] for helix.
    - [0, 1, 0] for sheet.
    - [0, 0, 1] for coil.

    or

    - [0] for helix.
    - [1] for sheet.
    - [2] for coil.
    """
    length = 3
    query = 'SELECT sec FROM raw_data where id=? ORDER BY resi'

    _sec2index = {'H': 0,
                  'G': 0,
                  'I': 0,
                  'E': 1,
                  'B': 1}
    _sec2onehot = {'H': [1., 0., 0.],
                   'G': [1., 0., 0.],
                   'I': [1., 0., 0.],
                   'E': [0., 1., 0.],
                   'B': [0., 1., 0.]}

    def __init__(self, onehot=True):
        """
        Init the transformer.

        Args:
            onehot: If True, one-hot encode the secondary structure.
        """
        self.onehot = onehot

    def _to_categorical_func(self, value, onehot):
        if onehot:
            return self._sec2onehot.get(value, [0., 0., 1.])
        else:
            return [self._sec2index.get(value, [2])]

    def transform(self, x):
        """
        Transforms the NumPy array [n, 1] to the [n, 3] categories.
        Args:
            x: [n, 1]-sized NumPy array (secondary structure).

        Returns:
            [n, 3] or [n,1]-sized NumPy array depending on onehot setting.
        """
        return np.array([self._to_categorical_func(v[0], self.onehot) for v in x])


class Topology(TransformerMixin, NoFitMixin, LengthMixin, Categorical):
    """
    ZCoordinates converted to three class topology but using reasonable TM segments,
    i.e.
    - [1, 0, 0, 0] for inside.
    - [0, 1, 0, 0] for TM.
    - [0, 0, 1, 0] for outside.
    - [0, 0, 0, 1] for a reentrant region.

    or

    - [0] for inside.
    - [1] for TM.
    - [2] for outside.
    - [3] for a reentrant region.
    """
    length = 4
    query = 'SELECT topo FROM raw_data where id=? ORDER BY resi'

    _topo2index = {'I': 0, 'T': 1, 'O': 2, 'R': 3}
    _topo2onehot = {'I': [1., 0., 0., 0.],
                    'T': [0., 1., 0., 0.],
                    'O': [0., 0., 1., 0.],
                    'R': [0., 0., 0., 1.]}

    def __init__(self, onehot=True):
        """
        Init the transformer.

        Args:
            onehot: If True, one-hot encode the topology.
        """
        self.onehot = onehot

    def _to_categorical_func(self, value, onehot):
        if onehot:
            return self._topo2onehot[value]
        else:
            return [self._topo2index[value]]

    def transform(self, x):
        """
        Transforms the NumPy array [n, 1] to the [n, 4] categories.
        Args:
            x: [n, 1]-sized NumPy array (topology).

        Returns:
            [n, 4] or [n,1]-sized NumPy array depending on onehot setting.
        """
        return np.array([self._to_categorical_func(v[0], self.onehot) for v in x])


class ZCoordinates(TransformerMixin, NoFitMixin, LengthMixin, Categorical):
    """
    ZCoordinates converted to three class topology, i.e.
    - [1, 0, 0] for inside.
    - [0, 1, 0] for TM.
    - [0, 0, 1] for outside.

    or

    - [0] for inside.
    - [1] for TM.
    - [2] for outside.
    """
    length = 3
    query = 'SELECT ca_z, thickness FROM raw_data JOIN proteins USING (id) where id=? ORDER BY resi'

    def __init__(self, onehot=True):
        """
        Init the transformer.

        Args:
            onehot: If True, one-hot encode the topology.
        """
        self.onehot = onehot

    @staticmethod
    def _to_categorical_func(value, thickness, onehot):
        if value < -thickness:
            if onehot:
                return 1., 0., 0.
            else:
                return [0.]
        elif value > thickness:
            if onehot:
                return 0., 0., 1.
            else:
                return [2.]
        else:
            if onehot:
                return 0., 1., 0.
            else:
                return [1.]

    def transform(self, x):
        """
        Transforms the NumPy array [n, 2] to the [n, 3] categories.
        Args:
            x: [n, 2]-sized NumPy array (z-coord and thickness tuples).

        Returns:
            [n, 3] or [n,1]-sized NumPy array depending on onehot setting.
        """
        return np.array([self._to_categorical_func(*v, self.onehot) for v in x])

