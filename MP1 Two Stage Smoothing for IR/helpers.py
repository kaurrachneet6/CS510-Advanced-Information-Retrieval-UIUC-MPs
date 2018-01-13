import csv
import metapy
import pytoml
import sys
from tqdm import tqdm

from two_stage_smoothing import TwoStageSmoothing

"""
Use this function to compute the mean average precision of the given
ranker based on the queries and relevance judgments present in the
provided config file.
"""
def compute_map(idx, cfg_path, ranker):
    with open(cfg_path, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        raise RuntimeError("query-runner table needed in {}".format(cfg))

    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)
    ireval = metapy.index.IREval(cfg_path)

    top_k = idx.num_docs() # get the score for every document

    query = metapy.index.Document()
    with open(query_path) as query_file:
        for query_num, line in enumerate(tqdm(query_file.readlines())):
            query_id = query_num + query_start
            query.content(line.strip())
            ranked_list = ranker.score(idx, query, top_k)
            average_precision = ireval.avg_p(ranked_list, query_id, top_k)
            """
            Fill in the rest of this loop to:

            1. Set up the query to have the content we just read from the
               file.

            2. Obtain the ranked list for the top_k documents for this
               query according to the ranker that was passed in (see
               help(metapy.index.Ranker) for more info).

            3. Using this ranked list, compute the average precision by
               calling a function on the IREval object (see
               help(metapy.index.IREval) for more info).
            """

    """
    At the end of the above loop, return the mean average precision across
    all of the queries (see help(metapy.index.IREval) for more info).
    """
    return ireval.map()


"""
Use this function to create a ranker of the given method name with the
specified parameter.
"""
def make_ranker(method, param):
    if method == 'jelinek-mercer':
        #return None # REPLACE ME (param == lambda)
        return metapy.index.JelinekMercer(param)
    if method == 'dirichlet-prior':
        #return None # REPLACE ME (param == mu)
        return metapy.index.DirichletPrior(param)
    if method == 'two-stage':
        #return None # REPLACE ME (param == mu)
        return TwoStageSmoothing(0.7, param)

    # this should only be reached if the function is used incorrectly
    raise ValueError("invalid method name")


"""
This function should create a CSV file containing all of the information
needed to create one figure.

method: A string with value 'jelinek-mercer', 'dirichlet-prior', or
'two-stage'.

parameter_list: A list of parameter values to produce MAP results for

dataset: The name of the dataset to test on

querysets: A list of the query types (short, long) X (keyword, verbose)
"""
def create_param_table(method, parameter_list, dataset, querysets):
    filename = "{}-sensitivity-{}.csv".format(method, dataset)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['queryset', 'parameter', 'map'])

        for queryset in querysets:
            cfg = "/data/{}/cfgs/{}.toml".format(dataset, queryset)

            print("Benchmarking {} on {} with {} queries...".format(method,
                dataset, queryset), file=sys.stderr)
            idx = metapy.index.make_inverted_index(cfg)
            for parameter in parameter_list:
                ranker=make_ranker(method,parameter)
                MAP=compute_map(idx, cfg, ranker)
                writer.writerow([queryset, parameter, MAP])

            """
            Fill in the rest of this loop to

            * For each parameter value:

                1. Create the appropriate ranker based on the parameter and
                   the specified method (hint: this should just be a call to the
                   make_ranker() function above).

                2. Compute the mean average precision for your ranker on
                   the queryset (hint: this should just be a call to the
                   compute_map() function you filled in earlier).

                3. Write the results to the CSV file (hint: you can see the
                   expected columns near the beginning of this function).
            """

