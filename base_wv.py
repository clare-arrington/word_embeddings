#%%
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from collections import Counter
from pathlib import Path
from typing import List
import pandas as pd
import pickle 
import random
import glob
import tqdm
import re

## TODO: adding a log file would be nice
## Could also save config as JSON or something for reference

## Paths get set to default values. To change just override after it's set.

def make_config(   
    dataset: str, corpus_name: str, run: str, 
    min_count: int, vector_size: int,
    targets: List[str], 
    load_data: bool, save_data: bool, 
    data_path: str 
    ):
    config = {
        "dataset": dataset, 
        "corpus_name" : corpus_name,
        "run": run, 
        "min_count" : min_count, 
        "vector_size" : vector_size,
        "num_sents" : None,
        "targets" : targets,

        "load_data" : load_data,
        "save_data" : save_data,

        "non_target_file" : f'{data_path}/corpus_data/{dataset}/subset/{corpus_name}_non_target.dat',
        "stored_non_t_file" : f'{data_path}/word_vectors/{dataset}/extra_data/{corpus_name}_sents.dat',
        
        "target_file" : f'{data_path}/corpus_data/{dataset}/subset/target_sentences.csv',
        "stored_t_file" : f'{data_path}/word_vectors/{dataset}/extra_data/{corpus_name}_target_sents.dat',

        "export_file" : f'{data_path}/word_vectors/{dataset}/{run}/{corpus_name}.vec',
        "sense_file" :  f'{data_path}/masking_results/{dataset}/{corpus_name}/sense_sentences.csv',
        }

    return config

def load_data_sentences(path, subset=None):
    if '.dat' in path:
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
def clean_sentences(sentences, pattern=None):
    print('\nCleaning data')

    reg_pattern = re.compile(r'[a-z]+_[a-z]{2}|[a-z]+\.\d|[a-z]+')

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        sent = re.sub('"', '', sent)
        cleaned = re.findall(reg_pattern, sent)
        clean_sents.append(cleaned) 

    return clean_sents

def filter_sentences(sentences, sense_words=[]):  
    stops = stopwords.words('english')

    found_senses = []
    filtered_sents = []
    for sent in tqdm.tqdm(sentences):
        new_sent = []

        for word in sent:

            first_word, *etc = word.split('.')
            if first_word in sense_words and len(etc) == 1: 
                found_senses.append(word)
                new_sent.append(word) 

            ## If the target word isn't in either format, 
            ## but we specified it's a target, exclude it.
            ## That's b/c we have both labeled and unlabeled which is bad 
            # if word in sense_words:
            #     found_senses.append(word)
            #     new_sent.append(word)
                # continue

            elif word not in stops:
                new_sent.append(word)

        filtered_sents.append(new_sent)

    return filtered_sents, found_senses
  
def get_normal_data(
    non_target_file, stored_non_t_file, 
    targets, num_sents=None,
    load_data=False, save_data=False):

    if load_data:
        print(f'\nLoading already parsed data from {stored_non_t_file}')
        with open(stored_non_t_file, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences)} normal sentences loaded')
    else:    
        print(f'Loading new data from {non_target_file}')
        normal_sents = load_data_sentences(non_target_file, subset=num_sents)
    
        sentences = clean_sentences(normal_sents, targets) 
        ## above shouldn't need targets b/c no normal sent should have a target word in it anyway
        ## verify this first before removing ig

        if save_data:
            Path(stored_non_t_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'\nSaving new data to {stored_non_t_file}')
            with open(stored_non_t_file, 'wb') as pf:
                pickle.dump(sentences, pf)

    return sentences

def get_sense_data(sense_file, targets, corpus, target_file):
    sents = pd.read_csv(sense_file, index_col='sent_id')

    if 'all' in sense_file:
        all_sents = pd.read_csv(target_file, index_col='sent_id')
        sents = sents.join(all_sents)
        sents = sents[sents.corpus == corpus]

    sense_sents = sents.sense_sentence.apply(eval)
    clean_sents, found_senses = filter_sentences(sense_sents, targets)
    print(f'{len(found_senses)} sense occurences found')

    return clean_sents, found_senses

## TODO: Some values are string already and some aren't?
## Get sentences with target words that weren't sense labeled
def get_target_data(target_file, stored_t_file, load_data, save_data):
    if load_data:
        print(f'\nLoading already parsed data from {stored_t_file}')
        with open(stored_t_file, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences)} target sentences loaded')

    else:    
        print(f'Loading target data from {target_file}')
        data = pd.read_csv(target_file)
        # data.formatted_sentence = data.formatted_sentence.apply(eval)
        print(f'{len(data)} target sentences loaded')

        sentences = list(data.sentence)
        sentences = clean_sentences(sentences)

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

def pull_full_data(data_path):

    sentences = []
    # pattern = re.compile(r'[a-z]+_[a-z]{2}|[a-z]+')

    with open(data_path) as fin:
        for line in tqdm.tqdm(fin.readlines()):
            line = line.lower().strip()

            # words = re.findall(pattern, line)
            # line = ' '.join(words)
            sentences.append(line)

    return sentences

def main(config):    
    print(f"Model will be saved to {config['export_file']}")

    sentences = get_normal_data(
        config['non_target_file'], 
        config['stored_non_t_file'], 
        config['targets'], 
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
            config['sense_path'], 
            config['targets'],
            config['corpus_name'], 
            config['target_path'])
        
        print(f'{len(Counter(found_senses))} senses found')
        print('5 most common targets')
        print(Counter(found_senses).most_common(5))
    
    elif config['run'] == 'new':
        # clean_sents = get_target_data(
        #     config['target_file'], config['stored_t_file'], 
        #     config['load_data'], config['save_data'])
        clean_sents = get_target_data(
            config['target_file'], config['stored_t_file'], 
            False, True)
        ## listcomp w/ intersect 
        print(f'Proof of target presence:', set(clean_sents[0]).intersection(set(config['targets'])))

    sentences.extend(clean_sents)
    print(f'\n{len(sentences)} total sentences prepped for model')

    model = save_model(
        config['export_file'], 
        sentences, 
        config['min_count'], 
        config['vector_size'])
    print(f'Model length: {len(model.wv.index_to_key)}')

    ##### 
    # Few checks for making sure senses were accounted for correctly
    # TODO: put behind a verify boolean
    # targets = [target.split('_')[0] for target in config['targets']]
    # not_removed = []
    # included = []
    # for target in targets:
    #     if target in model.wv.index_to_key:
    #         not_removed.append(target)
    # print(f'{len(not_removed)} target bases not removed')
    # print(', '.join(not_removed))

    # for target in config['targets']:
    #     if target in model.wv.index_to_key:
    #         included.append(target)
    # print(f'\n{len(included)} targets included')
    # print(', '.join(included))

# %%
