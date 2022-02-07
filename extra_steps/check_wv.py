#%%
from gensim.models import Word2Vec
from collections import defaultdict
import numpy as np

dataset = 'semeval_68'
corpus = 'ccoha2'
run = 'new'
path = f'/home/clare/Data/word_vectors/{dataset}/{run}/{corpus}.vec'
model = Word2Vec.load(path)

# with open(f'/home/clare/Data/corpus_data/{dataset}/targets.txt') as fin:
#     targets = []
#     for target in fin.read().strip().split('\n'):
#         word, label = target.split('\t')
#         target, pos = word.split('_')
#         targets.append(target)

with open(f'/home/clare/Data/corpus_data/{dataset}/targets.txt') as fin:
    targets = fin.read().strip().split('\n')

# vocab = list(model.wv.index_to_key)
# vectors=model.wv.get_normed_vectors()

#TODO: how to know how many of each?
senses = defaultdict(list)
for word in list(model.wv.index_to_key):
    if '.' in word:
        target, num = word.split('.')
        senses[target].append(num)

#%%

def print_closest(model, target, num_print=5):
    sims = model.wv.most_similar(target, topn=num_print)
    for word, dist in sims:
        print(f'{word:>15}: {dist:.2f}')

sense_counts = defaultdict(int)
for target, labels in senses.items():
    count = len(labels)
    sense_counts[count] += 1

sense_counts
#%%
for target in sorted(senses):
    labels = senses[target]
    avg = np.zeros(300)
    print(f'========= {target.capitalize()} =========')
    for num in sorted(labels):
        print(f'\nSense {num}')
        sense = f'{target}.{num}'
        avg += model.wv[f'{target}.{num}']
        print_closest(model, sense)
        
    print('\nAverage')
    avg /= len(labels)
    print_closest(model, avg)

    print('\n=====================================\n')

#%%
## TODO: could get size of each sense
## Also this interesting function below
# len(model.wv.closer_than('head.0', 'head.1'))

# %%
