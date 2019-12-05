# allestm
Predicting various structural features of transmembrane proteins.

## Prerequisites

### Installing and running HHblits
Allestm uses a multiple sequence alignment in a3m format as input which has
first to be created by HHblits, a database search tool.
HHblits and detailed installation instructions can be found on its [https://github.com/soedinglab/hh-suite](GitHub page).

Here is a short extract:
```
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
```

After installation, a database can be downloaded from
[http://gwdu111.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz](here).


### Computing requirements
TODO

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


### Model files
TODO

## Usage
Allestm can be run using the following command:

```
allestm msa.a3m output.json
```

## Output
TODO

## Citation
