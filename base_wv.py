#%%
from nltk.corpus import stopwords as sw
from gensim.models import Word2Vec
from collections import Counter
from pathlib import Path
from typing import Dict, List
import pickle 
import tqdm
import re

## TODO: adding a log file would be nice
def modify_slice_paths(
    paths: Dict[str, str],
    data_path: str, 
    slice_num: int
    ):

    slice_path = f'/slice_{slice_num}'

    for path_name, path in paths.items():
        if 'wv' in path_name:
            paths[path_name] = f'{data_path}/{path}{slice_path}'
        else:
            paths[path_name] = f'{data_path}/{path}{slice_path}/'

    for path_name, path in paths.items():
        paths[path_name] = f'{data_path}/{path}'

    return paths

def get_data(file_paths):
    with open(file_paths['target_file'], 'rb') as f:
        target_senses = pickle.load(f)
        targets = list(target_senses.target.unique())
    
    with open(file_paths['sent_file'], 'rb') as f:
        indexed_sents = pickle.load(f)
    
    with open(file_paths['sense_file'], 'rb') as f:
        sense_sents = pickle.load(f)

    return target_senses, targets, indexed_sents, sense_sents

def clean_regular_sentences(sentences, stopwords, pattern=None):
    if pattern is not None:
        reg_pattern = re.compile(pattern)

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        sent = re.sub('"', '', sent).lower()
        new_sent = []

        if pattern is not None:
            words = re.findall(reg_pattern, sent)
        else:
            words = sent.split()

        for word in words:
            if word not in stopwords and len(word) > 2:
                new_sent.append(word)

        clean_sents.append(new_sent) 

    return clean_sents

def filter_sentences(sentences, stopwords, sense_words): 
    print(f'\nCleaning the sentences with senses')

    found_senses = []
    filtered_sents = []
    for sent in tqdm.tqdm(sentences):
        new_sent = []
        for word in sent:
            word = word.lower()

            ## We check for this case first to record if it is a sense term 
            first_word, *etc = word.split('.')
            if first_word in sense_words and len(etc) == 1: 
                found_senses.append(word)
                new_sent.append(word) 

            elif word not in stopwords and len(word) > 2:
                new_sent.append(word)

        filtered_sents.append(new_sent)

    return filtered_sents, found_senses
  
def print_checks(targets, model):
    targets = [target.split('_')[0] for target in targets]
    not_removed = []
    included = []
    for target in targets:
        if target in model.wv.index_to_key:
            not_removed.append(target)
    print(f'{len(not_removed)} target bases not removed')
    print(', '.join(not_removed))

    for target in targets:
        if target in model.wv.index_to_key:
            included.append(target)
    print(f'\n{len(included)} targets included')
    print(', '.join(included))

#%%
def main (
    file_paths,
    wv_config,
    max_sents = None,
    verify_senses = False
    ):    

    target_senses, targets, indexed_sents, sense_sents = get_data(file_paths)

    ids_w_senses = list(target_senses.sent_idx.unique())
    sents_w_senses = indexed_sents[indexed_sents.index.isin(ids_w_senses)]
    sents_wo_senses = indexed_sents[~indexed_sents.index.isin(ids_w_senses)]

    stopwords = sw.words(wv_config['language'])
    stopwords.remove('no')
    
    ## Regardless of vector type, treat sentences without senses the same
    reg_sents = list(sents_wo_senses.sent)
    main_sentences = clean_regular_sentences(reg_sents, stopwords)

    # t = set([word for words in sentences for word in words])
    # for target in targets:
    #     if target in t:
    #         print(target)

    for vector_type in wv_config['vector_types']:
        print(f"\n\n==== Going to make a {vector_type} word vector ====\n")
        export_location = re.sub('VECTOR_TYPE', vector_type, file_paths['export_file'])
        print(f"Model will be saved to {export_location}")

        sentences = main_sentences

        if vector_type == 'sense':
            ## Shouldn't clean sentences because they've already been parsed
            clean_sents, found_senses = filter_sentences(
                sense_sents.sense_sent, stopwords, targets)
            print(f'{len(found_senses):,} sense occurences found')

            print(f'\n{len(Counter(found_senses)):,} senses found')
            sense_freqs = Counter(found_senses).most_common()
            print(f'5 most common targets : {sense_freqs[:5]}')
            print(f'5 least common targets : {sense_freqs[-5:]}')

        elif vector_type == 'normal':
            reg_sents = list(sents_w_senses.sent)
            clean_sents = clean_regular_sentences(reg_sents, stopwords, wv_config['pattern'])
            ## listcomp w/ intersect; should see a target 
            few_sents = [word for sent in clean_sents[:5] for word in sent]
            print(f'Target(s) present in first few sentences:', set(few_sents).intersection(set(targets)))

        sentences.extend(clean_sents)
        print(f'\n{len(sentences):,} total sentences prepped for model')

        print('\nMaking model...')
        model = Word2Vec(sentences, vector_size=wv_config['vector_size'], 
                         min_count=wv_config['min_count'], window=10)

        print('Saving model...')
        Path(export_location).parent.mkdir(parents=True, exist_ok=True)
        model.save(export_location)
        print('Model saved!')    

        print(f'Model length: {len(model.wv.index_to_key):,}')

        ##### 
        ## Few checks for making sure senses were accounted for correctly
        if verify_senses:
            print_checks(targets, model)
        
# %%
