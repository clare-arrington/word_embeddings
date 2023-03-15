#%%
from dotenv import dotenv_values
from base_wv import make_wvs
from collections import defaultdict
import json, itertools

FILE_PATHS = {
        "target_file"   : "masking_results/DATASET_NAME/C_GROUP/target_sense_labels.pkl",
        "sense_file"    : "masking_results/DATASET_NAME/C_GROUP/CORPUS_NAME_sense_sentences.pkl"
    }

## Generate respective paths for each form of sense clustering
def get_paths(wv_config, data_path, sense_cluster_types, dataset_name):

    cluster_paths = defaultdict(dict)
    ## We want all formats per corpus to be processed at the same time
    for corpus_name, cluster_type in itertools.product(wv_config['corpora'], sense_cluster_types): 
        if cluster_type == 'together':
            c_group = 'together'
        else:
            c_group = corpus_name

        ## Fill the paths with the proper variables
        paths = {}
        for path_name, path in FILE_PATHS.items():
            path = path.replace('DATASET_NAME', dataset_name)
            path = path.replace('C_GROUP', c_group)
            path = path.replace('CORPUS_NAME', corpus_name)
            path = path.replace('CLUSTER_TYPE', cluster_type)
            paths[path_name] = data_path + path

        cluster_paths[corpus_name][cluster_type] = paths
    return cluster_paths

#%%
if __name__ == "__main__":
    data_path = dotenv_values(".env")['data_path']
    sense_cluster_types = ['individual', 'together']
    dataset_name = 'semeval'

    with open(f"configs/{dataset_name}.json", "r") as read_file:
        wv_config = json.load(read_file)

    paths = get_paths(wv_config, data_path, sense_cluster_types, dataset_name)

    make_wvs(paths, wv_config, data_path, dataset_name)

    print('All done!')

# %%
