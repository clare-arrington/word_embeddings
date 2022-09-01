#%%
dataset = 'time'
data_path = '/home/clare/Data'
target_path = f'{data_path}/corpus_data/time/targets.txt'

wv_config = {
    "dataset_name"  : dataset,
    "data_path"     : data_path,
    "min_count"     : 100,
    "vector_size"   : 300,
    "pattern"       : r'[a-z]+',
    "corpora_targets"       : {
        # "1800s"     : target_path,
        # "2000s"     : target_path
        "coca"      : target_path
    }
}

file_paths = {
        "sense_file"        : "masking_results/{dataset}/{corpus_name}/sense_sentences.pkl",

        "target_file"       : "corpus_data/{dataset}/subset/target_sentences.pkl",
        "stored_non_t_file" : "word_vectors/{dataset}/extra_data/{corpus_name}_normal_sents.pkl",

        "non_target_file"   : "corpus_data/{dataset}/subset/{corpus_name}_non_target.dat",
        "stored_t_file"     : "word_vectors/{dataset}/extra_data/{corpus_name}_target_sents.pkl",

        "export_file"       : "word_vectors/{dataset}/{vector_type}/{corpus_name}.vec"
    }
