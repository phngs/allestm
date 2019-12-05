# allestm
Predicting various structural features of transmembrane proteins.

## Prerequisites

### Installing and running HHblits
Allestm uses a multiple sequence alignment in a3m format as input which has
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
file will serve as input for allestm.


### Computing requirements
Allestm uses more than 100 different trained models and some of them are quite
large. The tool was tested on a machine with 16 GB RAM which is the minimum for
it to work. If you experience issues and have a machine with 'only' 16 GB of
RAM, consider closing all other programs you have running.

During the first run, all models will be downloaded as they are not included in
the package. The download size in total
will be about 11 GB, therefore make sure that you have a fast internet
connection during that first run. Just to clarify, after the models are
downloaded and allestm finds them this download does not have to be repeated.

## Installation

### From PyPI
TODO

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
The model files will be downloaded automatically (about 11 GB) if allestm does not find them. The `-m` parameter gives you the possibility to specify you own location for the model files, if not given, allestm will download them into the package directory. The `-m` flag can become handy if 

- The package was installed into a system directory and there is no possiblity
  to store the model files there, e.g. because of missing permissions.
- The model files should be stored in a location which is accessible over the
  network, so that allestm can be run on a cluster.

## Usage
To get information about the command line options call:

```
allestm -h
```

Allestm can be run using the following command:

```
allestm msa.a3m output.json
```

## Output
TODO

## Citation
To be published.
