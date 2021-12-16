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
## Could also save config as JSON or something for reference
## Transition over all input data to pkl files

## Paths get set to default values. To change just override after it's set.
def make_config(
    dataset: str, 
    corpus_name: str, 
    run: str, 
    min_count: int, 
    vector_size: int,
    targets: List[str], 
    load_data: bool, 
    save_data: bool, 
    data_path: str, 
    paths: Dict[str, str] = {},
    slice_num: int = None,
    pattern: str = r'[a-z]+\.\d|[a-z]+',
    ):

    if slice_num is not None:
        slice_path = f'/slice_{slice_num}'

        paths = {
            'corpus_path'     : f'corpus_data/{dataset}/subset/{corpus_name}',
            'target_path'     : f'corpus_data/{dataset}/subset',
            'extra_data_path' : f'word_vectors/{dataset}/extra_data/{corpus_name}',
            'wv_path'         : f'word_vectors/{dataset}/{run}/{corpus_name}',
            'masking_path'    : f'masking_results/{dataset}/{corpus_name}',
        }

        for path_name, path in paths.items():
            if 'wv' in path_name:
                paths[path_name] = f'{data_path}/{path}{slice_path}'
            else:
                paths[path_name] = f'{data_path}/{path}{slice_path}/'
    else:
        paths.update({
            # 'corpus_path'     : f'corpus_data/{dataset}/subset/{corpus_name}{separator}',
            # 'target_path'     : f'corpus_data/{dataset}/subset/',
            'extra_data_path' : f'word_vectors/{dataset}/extra_data/{corpus_name}_',
            'wv_path'         : f'word_vectors/{dataset}/{run}/{corpus_name}',
            'masking_path'    : f'masking_results/{dataset}/{corpus_name}/',
        })

        for path_name, path in paths.items():
            paths[path_name] = f'{data_path}/{path}'
    
    config = {
        "dataset": dataset, 
        "corpus_name" : corpus_name,
        "run": run, 
        "min_count" : min_count, 
        "vector_size" : vector_size,
        "num_sents" : None,
        "pattern" : pattern,
        "targets" : targets,

        "load_data" : load_data,
        "save_data" : save_data,

        "non_target_file" : paths['corpus_path'] + "non_target.pkl",
        "stored_non_t_file" : paths['extra_data_path'] + "sents.pkl",
        
        "target_file" : paths['target_path'] + "target_sentences.pkl",
        "stored_t_file" : paths['extra_data_path'] + "target_sents.pkl",

        "export_file" : paths['wv_path'] + ".vec",
        "sense_file" :  paths['masking_path'] + "sense_sentences.pkl",
        }

    return config

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

    print(f'{num_samples} sentences\n')
    return sentences

# Reg pattern matches three things: word.#, word_pos, word
def clean_sentences(sentences, pattern):
    print('\nCleaning data')

    reg_pattern = re.compile(pattern)

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        sent = re.sub('"', '', sent).lower()
        cleaned = re.findall(reg_pattern, sent)
        clean_sents.append(cleaned) 

    return clean_sents

def filter_sentences(sentences, sense_words=[]):  
    stops = stopwords.words('english')
    print('\nFiltering data')

    found_senses = []
    filtered_sents = []
    for sent in tqdm.tqdm(sentences):
        new_sent = []
        for word in sent:
            word = word.lower()

            first_word, *etc = word.split('.')
            if first_word in sense_words and len(etc) == 1: 
                found_senses.append(word)
                new_sent.append(word) 

            ## If the target word isn't in either format, 
            ## but we specified it's a target, exclude it.
            ## That's b/c we have both labeled and unlabeled which is bad 
            elif word in sense_words:
                found_senses.append(word)
                new_sent.append(word)

            elif word not in stops and len(word) > 2:
                new_sent.append(word)
        filtered_sents.append(new_sent)
    return filtered_sents, found_senses
  
def get_normal_data(
    non_target_file, stored_non_t_file, 
    pattern,
    num_sents=None,
    load_data=False, save_data=False):

    if load_data:
        print(f'\nLoading already parsed data from {stored_non_t_file}')
        with open(stored_non_t_file, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences):,} normal sentences loaded')
    else:    
        print(f'Loading new data from {non_target_file}')
        normal_sents = load_data_sentences(non_target_file, subset=num_sents)
        clean_sents = clean_sentences(normal_sents, pattern)
        sentences, _ = filter_sentences(clean_sents) 
        ## above shouldn't need targets b/c no normal sent should have a target word in it anyway

        if save_data:
            Path(stored_non_t_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'\nSaving new data to {stored_non_t_file}')
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
        print(f'Loading target data from {target_file}')
        if '.csv' in target_file:
            data = pd.read_csv(target_file)
        elif '.pkl' in target_file:
            data = pd.read_pickle(target_file)
            ## TODO: change processed sentence in news
        
        sentences = list(data.sentence)
        sentences = clean_sentences(sentences, pattern)
        ## TODO: also filter?
        print(f'{len(sentences):,} target sentences loaded')
        print(sentences[0])

        if save_data:
            Path(stored_t_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'\nSaving target data to {stored_t_file}')
            with open(stored_t_file, 'wb') as pf:
                pickle.dump(sentences, pf)

    return sentences

def save_model(export_file, sentences, min_count, vector_size):
    print('Starting to make model')
    Path(export_file).parent.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(sentences, vector_size=vector_size, min_count=min_count, window=10)
    print('Starting to save model')
    model.save(export_file)
    print('Model saved!')
    return model

#%%
def main(config, verify_senses=False):    
    print(f"Model will be saved to {config['export_file']}")

    sentences = get_normal_data(
        config['non_target_file'], 
        config['stored_non_t_file'], 
        config['pattern'], 
        config['num_sents'],
        config['load_data'], 
        config['save_data']
        )

    # t = set([word for words in sentences for word in words])
    # for target in config['targets']:
    #     if target in t:
    #         print(target)

    if config['run'] == 'sense':
        clean_sents, found_senses = get_sense_data(
            config['sense_file'], config['targets'])
        
        print(f'{len(Counter(found_senses)):,} senses found')
        print('5 most common targets')
        print(Counter(found_senses).most_common(5))

    elif config['run'] == 'new':
        clean_sents = get_target_data(
            config['target_file'], config['stored_t_file'], 
            config['pattern'],
            config['load_data'], config['save_data'])
        # clean_sents = get_target_data(
        #     config['target_file'], config['stored_t_file'], 
        #     False, True)
        ## listcomp w/ intersect; should see a target 
        few_sents = [word for sent in clean_sents[:5] for word in sent]
        print(f'Target(s) present in first few sentences:', set(few_sents).intersection(set(config['targets'])))

    sentences.extend(clean_sents)
    print(f'\n{len(sentences):,} total sentences prepped for model')

    model = save_model(
        config['export_file'], 
        sentences, 
        config['min_count'], 
        config['vector_size'])
    print(f'Model length: {len(model.wv.index_to_key):,}')

    ##### 
    ## Few checks for making sure senses were accounted for correctly
    if verify_senses:
        targets = [target.split('_')[0] for target in config['targets']]
        not_removed = []
        included = []
        for target in targets:
            if target in model.wv.index_to_key:
                not_removed.append(target)
        print(f'{len(not_removed)} target bases not removed')
        print(', '.join(not_removed))

        for target in config['targets']:
            if target in model.wv.index_to_key:
                included.append(target)
        print(f'\n{len(included)} targets included')
        print(', '.join(included))

# %%
