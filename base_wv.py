#%%
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from collections import Counter
from pathlib import Path
import pandas as pd
import pickle
import random
import glob
import tqdm
import re

def make_config(
    dataset, corpus_name, run, min_count, 
    vector_size, targets, load_data, save_data,
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

        "non_target_file" : f'/home/clare/Data/corpus_data/{dataset}/subset/{corpus_name}_non_target.dat',
        "sampled_non_target_file" : f'/home/clare/Data/word_vectors/{dataset}/extra_data/{corpus_name}_sents.dat',
        "export_file" : f'/home/clare/Data/word_vectors/{dataset}/{run}/{corpus_name}.vec',
        
        "sense_path" :  f'/home/clare/Data/masking_results/{dataset}/{corpus_name}/sense_sentences.csv',
        "target_path" : f'/home/clare/Data/corpus_data/{dataset}/subset/target_sentencess.csv',
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

## Get sentences with target words that weren't sense labeled
def load_plain_target_sents(target_path, subset_path):
    data = pd.read_csv(target_path)
    data.formatted_sentence = data.formatted_sentence.apply(eval)
    print(f'{len(data)} target sentences loaded')

    if subset_path is None:
        target_sents = list(data.sentence)
    else:
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

# Reg pattern matches three things: word.#, word_pos, word
def clean_sentences(sentences, pattern=None):
    print('\nCleaning data')

    reg_pattern = re.compile(r'[a-z]+_[a-z]{2}|[a-z]+\.\d|[a-z]+')

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
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

            ## For the SemEval format: word_pos 
            ## Necessary?
            # elif '_' in word:
            #     print(word)
            #     target, pos = word.split('_')
            #     found_senses.append(target)
            #     clean_sent.append(target)

            elif word not in stops:
                new_sent.append(word)

        filtered_sents.append(new_sent)

    return filtered_sents, found_senses
  
def get_normal_data(
    non_target_file, sampled_non_target_file, 
    targets, num_sents=None,
    load_data=False, save_data=False):

    if load_data:
        print(f'\nLoading already sampled data from {sampled_non_target_file}')
        with open(sampled_non_target_file, 'rb') as pf:
            sentences = pickle.load(pf)
            print(f'\n{len(sentences)} normal sentences loaded')
    else:    
        print(f'Loading new data from {non_target_file}')
        normal_sents = load_data_sentences(non_target_file, subset=num_sents)
    
        sentences = clean_sentences(normal_sents, targets) 
        ## above shouldn't need targets b/c no normal sent should have a target word in it anyway
        ## verify this first before removing ig

        if save_data:
            Path(sampled_non_target_file).parent.mkdir(parents=True, exist_ok=True)
            print(f'\nSaving new data to {sampled_non_target_file}')
            with open(sampled_non_target_file, 'wb') as pf:
                pickle.dump(sentences, pf)

    return sentences

def get_sense_data(sense_path, targets, corpus, target_path):
    sents = pd.read_csv(sense_path, index_col='sent_id')

    if 'all' in sense_path:
        all_sents = pd.read_csv(target_path, index_col='sent_id')
        sents = sents.join(all_sents)
        sents = sents[sents.corpus == corpus]

    sense_sents = sents.sense_sentence.apply(eval)
    clean_sents, found_senses = filter_sentences(sense_sents, targets)
    print(f'{len(found_senses)} senses found')

    return clean_sents, found_senses

## We don't pass in targets to so they aren't excluded
## Also don't get info on them in here though :/
def get_target_data(target_path, subset_path):
    target_sents = load_plain_target_sents(target_path, subset_path)
    clean_sents = clean_sentences(target_sents)

    return clean_sents

def save_model(export_file, sentences, min_count, vector_size):
    Path(export_file).parent.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(sentences, vector_size=vector_size, min_count=min_count, window=10)
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

#%%
def full_file(config):
    print(f"Model will be saved to {config['export_file']}")

    ## TODO: some assumption on sent?
    sentences = pull_full_data(f'/home/clare/Data/corpus_data/{config["dataset"]}/subset/{config["corpus_name"]}.txt')
    # print(sentences[:3])

    sentences = clean_sentences(sentences)
    # print(sentences[:3])

    print(f'\n{len(sentences)} total sentences prepped for model')

    model = save_model(config['export_file'], sentences, config['min_count'])
    print(f'Model length: {len(model.wv.index_to_key)}\n')

def main(config):    
    print(f"Model will be saved to {config['export_file']}")

    sentences = get_normal_data(
        config['non_target_file'], 
        config['sampled_non_target_file'], 
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
            config['sense_path'], config['targets'],
            config['corpus_name'], config['target_path'])
        
        print('5 most common targets found')
        print(Counter(found_senses).most_common(5))
    
    elif config['run'] == 'new':
        ## TODO: fix this?
        clean_sents = get_target_data(config['target_path'], config['subset_path'])
        ## listcomp w/ intersect 
        print(set(clean_sents[0]).intersection(set(config['targets'])))

    sentences.extend(clean_sents)
    print(f'\n{len(sentences)} total sentences prepped for model')

    model = save_model(config['export_file'], sentences, config['min_count'], config['vector_size'])
    print(f'Model length: {len(model.wv.index_to_key)}')

    ##### 
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
