#%%
import itertools
from base_wv import main

## Import the wv_config you want to use
# from config_files.semeval_config import file_paths, wv_config
# from config_files.us_uk_config import file_paths, wv_config
from config_files.arxiv_config import file_paths, wv_config

def make_wv(corpus_name, vector_type, wv_config, file_paths):
    print(f"\n\n==== Going to make a {vector_type} word vector from {corpus_name} data ====\n")

    if vector_type == "normal":
        load_data = False
        save_data = True
    elif vector_type == "sense":
        load_data = True
        save_data = False
    else:
        print("Unknown vector type defined")
        return

    ## TODO: find a better way to define path templates and fill them later
    ## Bc eval is unsafe 
    paths = {}
    for path_name, path in file_paths.items():
        path = eval(f"f'{path}'")
        paths[path_name] = path

    main(
        vector_type, 
        wv_config['min_count'], wv_config['vector_size'], 
        wv_config['corpora_targets'][corpus_name], 
        load_data, save_data, 
        wv_config['data_path'], paths)

#%%
def batch_run(wv_config, file_paths):
    vector_types = ("normal", "sense")
    t = itertools.product(wv_config['corpora_targets'].keys(), vector_types)
    
    for corpus_name, vector_type in t:
        make_wv(corpus_name, vector_type, wv_config, file_paths)

if __name__ == "__main__":

    print(f"==== Starting batch run for {wv_config['dataset_name']} ====")

    batch_run(wv_config, file_paths)

#%%
