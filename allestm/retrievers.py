"""
Module containing data retrievers implemented as Transformer.
"""
import numpy as np
from sklearn.base import TransformerMixin

from mllib.features.base import NoFitMixin


class SQLRetriever(TransformerMixin, NoFitMixin):
    """
    Retrieve columns from a database according to a given query and parameters.
    """
    def __init__(self, conn, query):
        """
        Constructor.

        Args:
            conn: Connection object.
            query: The query including a ? which will be replaced by the id.
        """
        self.conn = conn
        self.query = query

    def transform(self, *params):
        """
        Fetch the data according to the query and the given parameters.

        Args:
            params: Parameters used in the where clause of the query (which have been given
            in the constructor with a ?.

        Returns:
            Numpy Array
        """
        return np.array(self.conn.cursor().execute(self.query, params).fetchall())
