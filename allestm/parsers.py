"""Parse fasta and a3m files."""
import numpy as np
import pathlib
import logging


def parse_a3m(filename):
    """Parse a a3m fasta file.

    Parameters
    ----------
    filename : string
        The a3m file.

    Returns
    -------
    tuple :
        (Sequence in an [n, 1] np.array, MSA in an [1, 1] np.array)
    """
    content = pathlib.Path(filename).read_text()

    seq = ''
    for line in content.split('\n'):
        if line.startswith('>'):
            if len(seq) > 0:
                break
        else:
            seq += line.strip().upper().replace('-', '')

    return (np.array([[x] for x in list(seq)]), np.array([[content]]))
