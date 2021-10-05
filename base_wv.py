#%%
from semeval_config import config as semeval_config
from usuk_config import config as usuk_config

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from collections import Counter

import pandas as pd
import pathlib
import pickle
import random
import glob
import tqdm
import re

def load_data_sentences(path, subset=None):
    if '.dat' in path:
        with open(path, 'rb') as f:
            sentences = pickle.load(f)
    elif '.txt' in path:
        with open(path, 'r') as f:
            sentences = f.read().splitlines()

    if subset:
        num_samples = min(len(sentences), subset)
        sentences = random.sample(sentences, num_samples)
    else: 
        num_samples = len(sentences)

    print(f'{num_samples} sentences\n')
    return sentences

def load_sense_sentences(path):
    paths = glob.glob(f'{path}*.dat')
    print(f'Pulling sense sentences for {len(paths)} targets')
    targets = []
    sentences = []
    for p in tqdm.tqdm(paths):
        with open(p, 'rb') as f:
            sense_sentences = pickle.load(f)
            target = p[len(path):].split('_')[0]
            targets.append(target)

            #print(f'{target} : {len(sense_sentences)} sentences')
            sentences.extend(sense_sentences)

    print(f'\n{len(sentences)} sentences pulled')
    return sentences, targets

## Get sentences with target words that weren't sense labeled
def load_plain_target_sents(target_path, subset_path):
    data = pd.read_csv(target_path)
    data.formatted_sentence = data.formatted_sentence.apply(eval)
    print(f'{len(data)} target sentences loaded')

    sentence_ids = []
    for row in glob.glob(f'{subset_path}/*.dat'):
        target, _ = row[len(subset_path)+1:].split('_')
        
        with open(row, 'rb') as fin:
            clusters = pickle.load(fin)
            for id, cluster in clusters.items():
                sentence_ids.extend(cluster)

    target_sents = data[data.word_index.isin(sentence_ids)]
    target_sents = list(target_sents.sentence)
    print(f'\n{len(target_sents)} target sentences selected')

    return target_sents 

def clean_sentences(sentences, sense_words=[]):  
    print('\nCleaning data')
  
    ## TODO: if semeval we need this other as well, add a 
    # Reg pattern matches three things: word.#, word_pos, word
    # reg_pattern = re.compile(r'[a-z]+\.\d|[a-z]+_[a-z]{2}|[a-z]+')
    if len(sense_words) > 0:
        reg_pattern = re.compile(r'[a-z]+\.\d|[a-z]+')
    else:
        reg_pattern = re.compile(r'[a-z]+')

    stops = stopwords.words('english')

    found_senses = []
    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        clean_sent = []
        cleaned = re.findall(reg_pattern, sent)

        for word in cleaned:

            ## 'target' won't pass but 'target.0' should
            first_word, *etc = word.split('.')
            if first_word in sense_words and len(etc) == 1: 
                found_senses.append(word)
                clean_sent.append(word) 

            ## For the SemEval format: word_pos 
            elif '_' in word:
                target, pos = word.split('_')
                found_senses.append(target)
                clean_sent.append(target)

            ## If the target word isn't in either format, 
            ## but we specified it's a target, exclude it.
            ## That's b/c we have both labeled and unlabeled which is bad 
            elif word in sense_words:
                continue

            ## Basic checks
            elif (word not in stops) and (len(word) > 2):
                clean_sent.append(word)

        clean_sents.append(clean_sent)

    return clean_sents, found_senses
  
def get_normal_data(
    non_target_path, sampled_non_target_path, 
    targets, num_sents=1000000,
    load_data=False, save_data=False):

    if load_data:
        print(f'\nLoading already sampled data from {sampled_non_target_path}')
        with open(sampled_non_target_path, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences)} normal sentences loaded')
    else:    
        print(f'Loading new data from {non_target_path}')
        normal_sents = load_data_sentences(non_target_path, subset=num_sents)
    
        sentences, _ = clean_sentences(normal_sents, targets) 
        ## above shouldn't need targets b/c no normal sent should have a target word in it anyway
        ## verify this first before removing

        if save_data:
            print(f'\nSaving new data to {sampled_non_target_path}')
            with open(sampled_non_target_path, 'wb') as pf:
                pickle.dump(sentences, pf)

    return sentences


## TODO: The two below can prob be merged with the load functions
def get_sense_data(sense_path, targets):
    sense_sents, _ = load_sense_sentences(sense_path)
    clean_sents, found_senses = clean_sentences(sense_sents, targets)
    print(f'{len(found_senses)} senses found')

    return clean_sents, found_senses

## We don't pass in targets to so they aren't excluded
## Also don't get info on them in here though :/
def get_target_data(target_path, subset_path):
    target_sents = load_plain_target_sents(target_path, subset_path)
    clean_sents, _ = clean_sentences(target_sents)

    return clean_sents

def save_model(export_path, sentences, corpus_name, min_count):
    pathlib.Path(export_path).mkdir(parents=True, exist_ok=True)
    model = Word2Vec(sentences, vector_size=300, min_count=min_count, window=10)
    model.save(f'{export_path}/{corpus_name}.vec')
    print('Model saved!')

#%%
def main(config):
    ## TODO: I should add save and load data to the config I suppose
    
    print(f"Model will be saved to {config['export_path']}")

    sentences = get_normal_data(
        config['non_target_path'], 
        config['sampled_non_target_path'], 
        config['targets'], 
        load_data=True
        )

    if config['run'] == 'sense':
        clean_sents, found_senses = get_sense_data(config['sense_path'], config['targets'])
        print(Counter(found_senses).most_common(2))
    
    elif config['run'] == 'new':
        ## TODO: Num isn't exactly the same as senses; why?
        clean_sents = get_target_data(config['target_path'], config['subset_path'])
        ## listcomp w/ intersect 
        print(set(clean_sents[0]).intersection(set(config['targets'])))

    sentences.extend(clean_sents)
    print(f'\n{len(sentences)} total sentences prepped for model')

    save_model(config['export_path'], sentences, config['corpus_name'], config['min_count'])

# %%
main(semeval_config)
# %%
