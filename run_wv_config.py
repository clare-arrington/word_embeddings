#%%
from dotenv import dotenv_values
from base_wv import main
import json

def make_wv(c_group, corpus_name, wv_config, file_paths, data_path):
    print(f"\n\n======== Pulling {c_group} : {corpus_name} data ========\n")

    ## TODO: find a better way to define path templates and eval them later
    ## Bc eval is unsafe 
    dataset = wv_config['dataset_name']
    paths = {}
    for path_name, path in file_paths.items():
        path = eval(f"f'{path}'")
        paths[path_name] = data_path + path

    main(paths, wv_config)

#%%
if __name__ == "__main__":

    dataset_name = 'spanish'
    data_path = dotenv_values(".env")['data_path']

    with open(f"config_files/{dataset_name}.json", "r") as read_file:
        config = json.load(read_file)

    corpora = config['wv_config']['corpora']
    for c_group in corpora:
        for corpus_name in corpora[c_group]:
            make_wv(c_group, corpus_name, config['wv_config'],
                    config['file_paths'], data_path)

    print('All done!')

# %%
