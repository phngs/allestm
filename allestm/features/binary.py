"""
Module containing binary input features as well as output targets.
"""
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, Binarizer

from allestm.features.base import LengthMixin, NoFitMixin


class Binary:
    """
    Using the sigmoid activation function and binary crossentropy loss.
    """
    keras_activation = 'sigmoid'
    keras_loss = 'binary_crossentropy'
    lightgbm_objective = 'binary'


class Bfactors(TransformerMixin, NoFitMixin, LengthMixin, Binary):
    """
    Scale bfactors according to z-score normalization and convert to binary according
    to threshold.
    """
    length = 1
    query = 'SELECT bfactor FROM raw_data WHERE id=? ORDER BY resi'

    def __init__(self, threshold=0.03, with_mean=True, with_std=True, copy=True):
        """
        Constructor.

        Args:
            threshold: Threshold used for binarization.
            with_mean: Substract mean for standard scaling.
            with_std: Divide by standard deviation for standard scaling.
            copy: Copy input.
        """
        self.threshold = threshold
        self.standard_scaler = StandardScaler(with_mean=with_mean, with_std=with_std, copy=copy)
        self.binarizer = Binarizer(threshold=threshold, copy=copy)

    def transform(self, x):
        """
        Transform the bfactor values by standard scaling and binarization.

        Args:
            x: [n, 1]-shaped NumPy array.

        Returns:
            [n, 1]-shaped NumPy array.
        """
        return self.binarizer.transform(self.standard_scaler.fit_transform(x))
