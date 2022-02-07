#%%
import pandas as pd

## TODO: don't like this
def get_targets(data_path, vector_type, from_masking=False, corpus_name=None):
    if from_masking and corpus_name:
        target_data = f'{data_path}/masking_results/semeval/{corpus_name}/target_sense_labels.pkl'
        target_data = pd.read_pickle(target_data)
        targets = [t for t in target_data.target.unique()]
    else:
        with open(f'{data_path}/corpus_data/semeval/targets.txt') as fin:
            targets = fin.read().split()
            if vector_type == 'sense':
                targets = [target.split('_')[0] for target in targets]
    
    print(f'{len(targets)} targets loaded')
    print(targets[:5])
    return targets

data_path = "/home/clare/Data"
targets =  ['virus', 'bit', 'memory', 'long', 
            'float', 'web', 'worm', 'bug', 'structure',
            'cloud', 'ram', 'apple', 'cookie', 
            'spam',  'intelligence', 'artificial', 
            'time', 'work', 'action', 'goal', 'branch',
            'power', 'result', 'complex', 'root',
            'process', 'child', 'language', 'term',
            'rule', 'law', 'accuracy', 'mean', 
            'scale', 'variable', 'rest', 
            'normal', 'network', 'frame', 'constraint', 
            'subject', 'order', 'set', 'learn', 'machine',
            'problem', 'scale', 'large', 
            'model', 'based', 'theory', 'example', 
            'function', 'field', 'space', 'state', 
            'environment', 'compatible', 'case', 'natural', 
            'agent', 'utility', 'absolute', 'value', 
            'range', 'knowledge', 'symbol', 'true', 
            'class', 'object', 'fuzzy', 'global', 'local', 
            'search', 'traditional', 'noise', 'system']

wv_config = {
    "data_path"     : data_path,
    "dataset_name"  : "semeval",
    "min_count"     : 50,
    "vector_size"   : 300,
    # "pattern"       : r'[a-z]+_[a-z]{2}|[a-z]+',
    "pattern"       : r'[a-z]+',
    "corpora_targets"       : {
        "1800s" :  targets,
                #  get_targets(data_path, vector_type)
        "2000s" :  targets
            # "targets" : get_targets(data_path, vector_type)
    }
}

# AI version
file_paths = {
        "sense_file"        : "masking_results/semeval/{corpus_name}_ai/sense_sentences.pkl",

        "target_file"       : "corpus_data/semeval/subset_ai/target_sentences.pkl",
        "stored_non_t_file" : "word_vectors/semeval/extra_data/{corpus_name}_ai_normal_sents.pkl",

        "non_target_file"   : "corpus_data/semeval/subset_ai/{corpus_name}_non_target.dat",
        "stored_t_file"     : "word_vectors/semeval/extra_data/{corpus_name}_ai_target_sents.pkl",

        "export_file"       : "word_vectors/semeval/{vector_type}/{corpus_name}_ai.vec"
    }
#%%
