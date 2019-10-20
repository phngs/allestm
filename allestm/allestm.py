import sys
from argsparse import ArgumentParser

def parse_fasta(filename):
  pass

def parse_a3m(filename):
  pass

def parse_args(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-f', '--fasta', help='Single sequence .fasta file.')
    parser.add_argument('-a', '--a3m', help='MSA in a3m format')
    return parser.parse_args(argv)

# ["lstm_", "cnn_", "dcnn_", "rf_", "xgb_"]
# * 5 folds
# blending
# avg
# x targets
def main():
  args = parse_args(sys.argv[1:])

if __name__ == '__main__':
    main()