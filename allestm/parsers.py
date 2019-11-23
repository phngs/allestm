"""Parse fasta and a3m files."""
import numpy as np
import pathlib
import logging


def parse_fasta(filename):
    """Parse a (single sequence) fasta file.

    Parameters
    ----------
    filename : string
        The fasta file.

    Returns
    -------
    np.array
        Sequence in an [n, 1] np.array.
    """
    seq = ''
    with open(filename, 'r') as fh:
        for line in fh:
            if line.startswith('>'):
                if len(seq) > 0:
                    logging.warning('FASTA file contains more than one sequence, will only use the first one and discard the others.')
                    break
            else:
                seq += line.strip().upper()

    return np.array([[x] for x in list(seq)])


def parse_a3m(filename):
    return np.array([[pathlib.Path(filename).read_text()]])
