#%%
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
    "dataset_name"  : "arxiv",
    "min_count"     : 50,
    "vector_size"   : 300,
    "pattern"       : r'[a-z]+',
    "corpora_targets"       : {
        "ai"    :  targets
        # "phys"  :  targets
    }
}

file_paths = {
        "sense_file"        : "masking_results/arxiv/{corpus_name}/sense_sentences.pkl",

        "target_file"       : "corpus_data/arxiv/subset/target_sentences.pkl",
        "stored_non_t_file" : "word_vectors/arxiv/extra_data/{corpus_name}_normal_sents.pkl",

        "non_target_file"   : "corpus_data/arxiv/subset/{corpus_name}_non_target.dat",
        "stored_t_file"     : "word_vectors/arxiv/extra_data/{corpus_name}_target_sents.pkl",

        "export_file"       : "word_vectors/arxiv/{vector_type}/{corpus_name}.vec"
    }
