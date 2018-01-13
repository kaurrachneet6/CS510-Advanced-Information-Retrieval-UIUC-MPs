import metapy
import sys

from helpers import compute_map
from helpers import make_ranker
from helpers import create_param_table
from two_stage_smoothing import TwoStageSmoothing

querysets = ['short-keyword', 'long-keyword', 'short-verbose', 'long-verbose']

# Each entry in this dictionary is a (method-name, smoothing-parameter-list).
experiments = {
    'jelinek-mercer':  [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'dirichlet-prior': [100, 500, 1000, 2000, 4000, 8000, 10000],
    'two-stage':       [100, 500, 1000, 2000, 4000, 8000, 10000]
}

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: {} method-name dataset [query sets]".format(sys.argv[0]))
        sys.exit(1)

    # print progress output to stderr
    metapy.log_to_stderr()

    method = sys.argv[1]
    dataset = sys.argv[2]
    queries = querysets
    if len(sys.argv) > 3:
        queries = sys.argv[3:]
    create_param_table(method, experiments[method], dataset, queries)
