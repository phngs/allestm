"""
Module containing base classes for input features as well as output targets.
"""


class NoFitMixin:
    """
    Mixin class for not fittable transformers
    """
    def fit(self, x=None, y=None):
        """
        Does nothing.

        Args:
            x: [n, 1]-shaped NumPy array.
            y: [n, 1]-shaped NumPy array.

        Returns:
            Itself.
        """
        return self


class LengthMixin:
    """
    Implements the __len__ method returning the value of the
    class or instance variable length
    """
    def __len__(self):
        return self.length
