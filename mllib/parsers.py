"""
Parsers for several prediction tool outputs.
"""
import numpy as np

max_solvent_acc = {'A': 106.0, 'C': 135.0, 'D': 163.0,
                   'E': 194.0, 'F': 197.0, 'G': 84.0,
                   'H': 184.0, 'I': 169.0, 'K': 205.0,
                   'L': 164.0, 'M': 188.0, 'N': 157.0,
                   'P': 136.0, 'Q': 198.0, 'R': 248.0,
                   'S': 130.0, 'T': 142.0, 'V': 142.0,
                   'W': 227.0, 'Y': 222.0}


def psipred(infile, sequence):
    """Parses the PSIPRED .horiz output file.

    Parameters
    ----------
    infile : str
        PSIPRED .horiz file.
    sequence : SeqRecord
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.

    """
    aa2sec = {
        'H': [1, 0, 0],
        'E': [0, 1, 0],
        'C': [0, 0, 1]
    }
    result = []
    with open(infile, 'r') as fh:
        for line in fh:
            if line.startswith('Pred:'):
                spl = line.strip().split(' ')
                if len(spl) < 2:
                    continue
                for aa in spl[1]:
                    result.append(aa2sec[aa])

    return np.array([result])


def prof(infile, sequence):
    """Parses the prof .profRdb output file.

    Parameters
    ----------
    infile : str
        Prof .profRdb file.
    sequence : SeqRecord
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.

    """
    aa2sec = {
        'H': [1, 0, 0],
        'E': [0, 1, 0],
        'L': [0, 0, 1]
    }
    result = []
    with open(infile, 'r') as fh:
        for line in fh:
            if not line.startswith('#') and not line.startswith('No'):
                aa = line.strip().split()[3]
                result.append(aa2sec[aa])

    return np.array([result])


def anglor(infile, sequence):
    """
    Parses the ANGLOR output file.

    Args:
        infile: ANGLOR output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    return np.loadtxt(infile, usecols=1).clip(min=-180, max=180).reshape((1, -1, 1))


def anglor_phi(infile, sequence):
    """
    Parses the ANGLOR (phi) output file.

    Args:
        infile: ANGLOR output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    return anglor(infile, sequence)


def anglor_psi(infile, sequence):
    """
    Parses the ANGLOR (psi) output file.

    Args:
        infile: ANGLOR output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    return anglor(infile, sequence)


def memsat_svm(infile, sequence):
    """
    Parses the Memsat SVM output file.

    Args:
        infile: Memsat SVM output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    with open(infile, "r") as fh:
        for line in fh:
            if line.startswith("Signal peptide:"):
                sp = 0
                if not line.strip().endswith("Not detected."):
                    sp = line.split(":")[1].strip().split("-")[1]
            elif line.startswith("Topology"):
                tms = [[y[0]-1, y[1]] for y in [list(map(int, x.split("-")))
                                                for x in line.split(":")[1].strip().split(",")]]
            elif line.startswith("Re-entrant helices:"):
                reh = []
                if not line.strip().endswith("Not detected."):
                    reh = [[y[0]-1, y[1]] for y in [list(map(int, x.split("-")))
                                                    for x in line.split(":")[1].strip().split(",")]]
            elif line.startswith("N-terminal"):
                orient = line.split(":")[1].strip()

    if orient == "in":
        result = [[1, 0, 0, 0] for _ in range(len(sequence))]
        orient = "out"
    else:
        result = [[0, 0, 1, 0] for _ in range(len(sequence))]
        orient = "in"

    for tm in tms:
        for i in range(*tm):
            result[i] = [0, 1, 0, 0]
        for i in range(tm[1], len(result)):
            if orient == "in":
                result[i] = [1, 0, 0, 0]
            else:
                result[i] = [0, 0, 1, 0]
        if orient == "in":
            orient = "out"
        else:
            orient = "in"

    for r in reh:
        for i in range(*r):
            result[i] = [0, 0, 0, 1]

    return np.array([result])


def polyphobius(infile, sequence):
    """
    Parses the Polyphobius output file.

    Args:
        infile: Polyphobius output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    tms = []
    doms = []
    with open(infile, "r") as fh:
        for line in fh:
            if line.startswith("FT"):
                split = line.strip().split()
                if split[1] == "TOPO_DOM":
                    if split[4] == "CYTOPLASMIC.":
                        doms.append(["cyto", int(split[2]) - 1, int(split[3])])
                    else:
                        doms.append(
                            ["noncyto", int(split[2]) - 1, int(split[3])])
                elif split[1] == "TRANSMEM":
                    tms.append([int(split[2]) - 1, int(split[3])])

    if doms[0][0] == "cyto":
        result = [[1, 0, 0, 0] for _ in range(len(sequence))]
    else:
        result = [[0, 0, 1, 0] for _ in range(len(sequence))]

    for dom in doms:
        if dom[0] == "cyto":
            for i in range(*dom[1:]):
                result[i] = [1, 0, 0, 0]
        else:
            for i in range(*dom[1:]):
                result[i] = [0, 0, 1, 0]

    for tm in tms:
        for i in range(*tm):
            result[i] = [0, 1, 0, 0]

    return np.array([result])


def predyflexy(infile, sequence):
    """
    Parses the Predyflexy output file.

    Args:
        infile: Predyflexy output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    result = np.loadtxt(infile, usecols=10, skiprows=1).reshape((1, -1, 1))
    result[:, :10, 0] = 0
    result[:, -10:, 0] = 0
    return result


def profbval_strict(infile, sequence):
    """
    Parses the profbval (strict) output file.

    Args:
        infile: Profbval output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    result = np.zeros((1, len(sequence), 1))
    with open(infile, "r") as fh:
        it = 0
        for line in fh:
            if not line.startswith("number"):
                pred_str = line.strip().split()[5]
                if pred_str == "F":
                    result[0, it, 0] = 1
                it += 1

    return result


def profbval_bnorm(infile, sequence):
    """
    Parses the profbval (normalized bfactors) output file.

    Args:
        infile: Profbval output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    result = np.zeros((1, len(sequence), 1))
    with open(infile, "r") as fh:
        it = 0
        for line in fh:
            if not line.startswith("number"):
                result[0, it, 0] = float(line.strip().split()[3])
                it += 1

    return result


def spinex_phi(infile, sequence):
    """
    Parses the SpineX (phi) output file.

    Args:
        infile: SpineX output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    return np.loadtxt(infile, usecols=3, skiprows=1).reshape((1, -1, 1))


def spinex_psi(infile, sequence):
    """
    Parses the SpineX (psi) output file.

    Args:
        infile: SpineX output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    return np.loadtxt(infile, usecols=4, skiprows=1).reshape((1, -1, 1))


def spinex_rsa(infile, sequence):
    """
    Parses the SpineX (rsa) output file.

    Args:
        infile: SpineX output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    data = np.loadtxt(infile, usecols=10, skiprows=1).reshape((1, -1, 1))
    for i in range(len(sequence)):
        data[0, i, 0] /= max_solvent_acc[sequence[i].upper()]

    return data


def spinex_sec(infile, sequence):
    """
    Parses the SpineX (sec) output file.

    Args:
        infile: SpineX output file.
        sequence: SeqRecord object or any other object whichs __len__ method
        returns the length of the sequence.

    Returns:
        NumPy array.
    """
    return np.loadtxt(infile, usecols=[7, 5, 6], skiprows=1).reshape((1, -1, 3))
