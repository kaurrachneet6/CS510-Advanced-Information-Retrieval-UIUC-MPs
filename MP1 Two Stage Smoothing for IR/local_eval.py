import metapy

from helpers import compute_map
from two_stage_smoothing import TwoStageSmoothing

if __name__ == '__main__':
    # print progress output to stderr
    metapy.log_to_stderr()

    idx = metapy.index.make_inverted_index('config.toml')
    ranker = TwoStageSmoothing()
    mean_ap = compute_map(idx, 'config.toml', ranker)

    print("Cranfield MAP: {}".format(mean_ap))
