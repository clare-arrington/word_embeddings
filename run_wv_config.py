#%%
from dotenv import dotenv_values
from base_wv import main
import json

FILE_PATHS = {
        "target_file"       : "masking_results/{dataset_name}/{c_group}/target_sense_labels.pkl",
        "sense_file"        : "masking_results/{dataset_name}/{c_group}/{corpus_name}_sense_sentences.pkl",
        
        "sent_file"        : "corpus_data/{dataset_name}/subset/{corpus_name}_indexed_sentences.pkl",
        
        "stored_file"       : "word_vectors/{dataset_name}/extra_data/{corpus_name}_sents.pkl",
        "export_file"       : "word_vectors/{dataset_name}/{cluster_type}/VECTOR_TYPE_{corpus_name}.vec"
    }

def make_wv(dataset_name, data_path, cluster_type):

    with open(f"configs/{dataset_name}.json", "r") as read_file:
        wv_config = json.load(read_file)
        
    for corpus_name in wv_config['corpora']:
        if cluster_type == 'shared':
            c_group = 'shared'
        else:
            c_group = corpus_name

        print(f"\n\n======== Pulling {corpus_name} data from {cluster_type} clustering ========\n")
        
        ## this fills the paths defined above with the proper variables
        paths = {}
        for path_name, path in FILE_PATHS.items():
            path = eval(f"f'{path}'")
            paths[path_name] = data_path + path

        main(paths, wv_config)

#%%
if __name__ == "__main__":

    dataset_name = 'semeval'
    data_path = dotenv_values(".env")['data_path']
    sense_cluster_types = ['individual', 'together']

    for cluster_type in sense_cluster_types:
        make_wv(dataset_name, data_path, cluster_type)

    print('All done!')

# %%
