# AllesTM - Predicting various structural features of transmembrane proteins.
AllesTM is an integrated tool to predict almost all structural features of
transmembrane proteins that can be extracted from atomic coordinate
data. It blends several machine learning algorithms: random forests and
gradient boosting machines, convolutional neural networks in their
original form as well as those enhanced by dilated convolutions and
residual connections, and, finally, long short-term memory architectures.

## Prerequisites

### Installing and running HHblits
AllesTM uses a multiple sequence alignment in a3m format as input which has
first to be created by HHblits, a database search tool.
HHblits and detailed installation instructions can be found on its [GitHub page](https://github.com/soedinglab/hh-suite).

Here is a short extract:
```
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
```

After installation, a database can be downloaded from
[here](http://gwdu111.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz). Make sure you check for the lastest version [here](https://uniclust.mmseqs.com/).

After extracting the database using `tar -xvf uniclust30_2018_08_hhsuite.tar.gz`, HHblits can be run with the following command:

```
hhblits -i infile.fasta -o output.hhr -oa3m msa.a3m -d PATH_TO_DB/uniclust30_2018_08/uniclust30_2018_08, -maxfilt 99999999
```

The `maxfilt` option is important to include all hits, the `msa.a3m` output
file will serve as input for AllesTM.


### Computing requirements
AllesTM uses more than 100 different trained models and some of them are quite
large. The tool was tested on a machine with 16 GB RAM which is the minimum for
it to work. If you experience issues and have a machine with 'only' 16 GB of
RAM, consider closing all other programs you have running.

During the first run, all models will be downloaded as they are not included in
the package. The download size in total
will be about 11 GB, therefore make sure that you have a fast internet
connection during that first run. Just to clarify, after the models are
downloaded and AllesTM finds them this download does not have to be repeated.

## Installation

### From PyPI
The easiest way to install AllesTM is by typing

```
pip install --user allestm
```

If python and pip is installed, this will get the necessary dependencies and AllesTM itself installed in your the user's home directory. See the section about model files while the --user option makes sense.

### From source
The package can easily be installed from the latest source here in the repo.

```
git clone https://github.com/phngs/allestm.git
cd allestm
pip install --user .
```

All dependencies like tensorflow and scikit-learn will be automatically
installed. If you want to use tensorflow with GPU support, make sure you
install it yourself (the speedup will be marginal, if noticable at all).


### Model files
The model files will be downloaded automatically (about 11 GB) if AllesTM does not find them. The `-m` parameter gives you the possibility to specify you own location for the model files, if not given, AllesTM will download them into the package directory. The `-m` flag can become handy if 

- The package was installed into a system directory and there is no possiblity
  to store the model files there, e.g. because of missing permissions.
- The model files should be stored in a location which is accessible over the
  network, so that AllesTM can be run on a cluster.

## Usage
To get information about the command line options call:

```
allestm -h
```

AllesTM can be run using the following command:

```
allestm msa.a3m output.json
```

During install, an example file is included. See the end of the output of `allestm -h` on how to call AllesTM with the example file.

## Output
The output is in JSON format and has the following levels:
1. Target 
2. Algorithm
3. (Fold)
4. Predictions

The fold level is omitted for the final predictions denoted by 'avg'. As 'avg' represents the final preditions, this is what you are most probably interested in. See the publication for a detailed explanation of the different targets and usage of algorithms. Here is a short description:

- continuous.PhiAngles
    - Torsion angle phi from -180 to +180 degrees.
- continuous.PsiAngles
    - Torsion angle psi from -180 to +180 degrees.
- continuous.Bfactors
    - Per protein z-normalized B-factors (not directly comparable to original B-factors)
- binary.Bfactors
    - Binary flexibility, 1 is a probablitly of 100% that the residue is flexible, 0 means not flexible.
- continuous.RsaComplex
    - Relative solvent accessiblity of the protein in its complex
- continuous.RsaChain
    - Relative solvent accessiblity of the protein as a monomere
- continuous.RsaDiff
    - Relative solvent accessiblity difference between the protein in complex and its monomeric form
- continuous.ZCoordinates
    - Distance in Angstroem from the membrane center, i.e. 0 is exactly between the two membrane boundaries (which are on average at -15 and +15)
- categorical.Topology
    - Prediction of membrane protein topology per residue in 4 states, each position in the array is the probability for that state:
    - [inside, transmembrane, outside, reentrant region]
    - e.g. [0.1, 0.6, 0.2, 0.1] means that the residue is most probably in the transmembrane region
- categorical.SecStruc
    - Three-state seconary structure with the three states [helix, sheet, coil]

```
{
    "continuous.PhiAngles":{
        "avg":[
            -34.597,
            -66.266,
            -63.31,
            ...
        ],
        "cnn":{
            "0":[
                18.339,
                -66.729,
                -63.547
            ],
            "1":[
                3.474,
                -65.085,
                -66.94
            ],
            ...
        },
        "dcnn":{
            ...
        },
        "lstm":{
            ...
        },
        "rf":{
            ...
        },
        "xgb":{
            ...
        },
        "blending":{
            ...
        }
    },
    "continuous.PsiAngles":{
        "avg":[
            8.346,
            -29.853,
            -38.515,
            ...
        ],
        ...
    },
    "continuous.Bfactors":{
        "avg":[
            1.852,
            1.488,
            1.473,
            ...
        ],
        ...
    },
    "binary.Bfactors":{
        "avg":[
            [
                0.169,
                0.831
            ],
            [
                0.199,
                0.801
            ],
            [
                0.198,
                0.802
            ],
            ...
        ],
        ...
    },
    "continuous.RsaComplex":{
        "avg":[
            0.75,
            0.485,
            0.641,
            ...
        ],
        ...
    },
    "continuous.RsaChain":{
        "avg":[
            0.78,
            0.536,
            0.7,
            ...
        ],
        ...
    },
    "continuous.RsaDiff":{
        "avg":[
            0.035,
            0.047,
            0.045,
            ...
        ],
        ...
    },
    "continuous.ZCoordinates":{
        "avg":[
            -14.872,
            -14.395,
            -14.356,
            ...
        ],
        ...
    },
    "categorical.Topology":{
        "avg":[
            [
                0.894,
                0.054,
                0.041,
                0.01
            ],
            [
                0.813,
                0.127,
                0.047,
                0.013
            ],
            [
                0.574,
                0.364,
                0.041,
                0.02
            ],
            ...
        ],
        ...
    },
    "categorical.SecStruc":{
        "avg":[
            [
                0.229,
                0.039,
                0.733
            ],
            [
                0.763,
                0.013,
                0.224
            ],
            [
                0.917,
                0.007,
                0.077
            ],
            ...
        ],
        ...
    }
}
```

## Citation
To be published.
