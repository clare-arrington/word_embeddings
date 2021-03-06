#%%
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from collections import Counter
from pathlib import Path
from typing import Dict, List
import pandas as pd
import pickle 
import random
import tqdm
import re

## TODO: adding a log file would be nice
## TODO: transition all input data over to pkl files

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

def load_data_sentences(path, subset=None):
    if '.dat' or '.pkl' in path:
        with open(path, 'rb') as f:
            sentences = pickle.load(f)
    elif '.txt' in path:
        with open(path, 'r') as f:
            sentences = f.read().splitlines()

    if subset:
        num_samples = min(len(sentences), subset)
        if num_samples is not None:
            sentences = random.sample(sentences, num_samples)
    else: 
        num_samples = len(sentences)

    # print(f'{num_samples} sentences\n')
    return sentences

def clean_regular_sentences(sentences, pattern):
    print(f'\nCleaning data by applying regex pattern : {pattern}')

    reg_pattern = re.compile(pattern)
    stops = stopwords.words('english')

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        sent = re.sub('"', '', sent).lower()
        new_sent = []

        for word in re.findall(reg_pattern, sent):
            if word not in stops and len(word) > 2:
                new_sent.append(word)

        clean_sents.append(new_sent) 

    return clean_sents

def filter_sentences(sentences, sense_words): 
    # sense_pattern = r'[a-z]+\.\d' 
    stops = stopwords.words('english')
    print(f'\nFiltering out stop words and small words')

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

            elif word not in stops and len(word) > 2:
                new_sent.append(word)

        filtered_sents.append(new_sent)

    return filtered_sents, found_senses
  
def get_normal_data(
    non_target_file, stored_non_t_file, 
    pattern,
    max_sents=None,
    load_data=False, save_data=False):

    if load_data:
        print(f'\nLoading already parsed data from {stored_non_t_file}')
        with open(stored_non_t_file, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences):,} normal sentences loaded')
    else:    
        print(f'\nLoading normal data from {non_target_file}')
        normal_sents = load_data_sentences(non_target_file, subset=max_sents)
        sentences = clean_regular_sentences(normal_sents, pattern)

        if save_data:
            Path(stored_non_t_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'\nSaving normal data to {stored_non_t_file}')
            with open(stored_non_t_file, 'wb') as pf:
                pickle.dump(sentences, pf)

    return sentences

def get_sense_data(sense_file, targets):
    if '.csv' in sense_file:
        sents = pd.read_csv(sense_file, index_col='sent_id')
        sense_sents = sents.sense_sentence.apply(eval)
    elif '.pkl' in sense_file:
        sents = pd.read_pickle(sense_file)
        sense_sents = sents.sense_sentence

    ## Shouldn't clean sentences first because they've already been parsed before
    clean_sents, found_senses = filter_sentences(sense_sents, targets)
    print(f'{len(found_senses):,} sense occurences found')

    return clean_sents, found_senses

## Get sentences with target words that weren't sense labeled
def get_target_data(target_file, stored_t_file, pattern, load_data, save_data):
    if load_data:
        print(f'\nLoading already parsed data from {stored_t_file}')
        with open(stored_t_file, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences):,} target sentences loaded')
    else:    
        print(f'\nLoading target data from {target_file}')
        if '.csv' in target_file:
            data = pd.read_csv(target_file)
        elif '.pkl' in target_file:
            data = pd.read_pickle(target_file)
            ## TODO: change processed sentence in news
        
        sentences = list(data.sentence)
        sentences = clean_regular_sentences(sentences, pattern)
        print(f'{len(sentences):,} target sentences loaded')
        print('First cleaned sentence for sanity check:')
        print(sentences[0])

        if save_data:
            Path(stored_t_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'\nSaving target data to {stored_t_file}')
            with open(stored_t_file, 'wb') as pf:
                pickle.dump(sentences, pf)

    return sentences

#%%
def main (
    vector_type: str, 
    min_count : int, vector_size : int,
    targets : List[str], 

    load_data : bool, save_data : bool, 
    data_path: str, file_paths: Dict[str, str],

    max_sents : int = None,
    pattern: str = r'[a-z]+',
    verify_senses: bool = False
    ):    

    for path_name, path in file_paths.items():
        file_paths[path_name] = f'{data_path}/{path}'

    print(f"Model will be saved to {file_paths['export_file']}")

    sentences = get_normal_data(
        file_paths['non_target_file'], file_paths['stored_non_t_file'], 
        pattern, max_sents, load_data, save_data )

    # t = set([word for words in sentences for word in words])
    # for target in targets:
    #     if target in t:
    #         print(target)

    if vector_type == 'sense':
        clean_sents, found_senses = get_sense_data( file_paths['sense_file'], targets )
        
        print(f'\n{len(Counter(found_senses)):,} senses found')
        sense_freqs = Counter(found_senses).most_common()
        print(f'5 most common targets : {sense_freqs[:5]}')
        print(f'5 least common targets : {sense_freqs[-5:]}')

    elif vector_type == 'normal':
        clean_sents = get_target_data(
            file_paths['target_file'], file_paths['stored_t_file'], 
            pattern, load_data, save_data )
        ## listcomp w/ intersect; should see a target 
        few_sents = [word for sent in clean_sents[:5] for word in sent]
        print(f'Target(s) present in first few sentences:', set(few_sents).intersection(set(targets)))

    sentences.extend(clean_sents)
    print(f'\n{len(sentences):,} total sentences prepped for model')

    print('\nMaking model...')
    model = Word2Vec(sentences, vector_size=vector_size, min_count=min_count, window=10)

    print('Saving model...')
    Path(file_paths['export_file']).parent.mkdir(parents=True, exist_ok=True)
    model.save(file_paths['export_file'])
    print('Model saved!')    

    print(f'Model length: {len(model.wv.index_to_key):,}')

    ##### 
    ## Few checks for making sure senses were accounted for correctly
    if verify_senses:
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

# %%
