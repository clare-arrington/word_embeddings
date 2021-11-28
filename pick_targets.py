#%%
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy

path = f'/home/clare/Data/word_vectors/semeval/sense/ccoha1.vec'
model = Word2Vec.load(path)
vocab = list(model.wv.index_to_key)

targets = ['little', 'much', 'hand', 'long', 
            'look', 'nul', 'shall', 'first', 
            'good', 'place', 'two', 'life', 
            'old', 'never', 'without', 'yet', 
            'many', 'heart', 'might', 'thing', 
            'leave', 'seem', 'love', 'power', 
            'feel', 'though', 'far', 'country', 
            'way', 'mind', 'tell', 'work', 
            'still', 'hear', 'call', 'people', 
            'form', 'house', 'friend', 'young', 
            'stand', 'speak', 'last', 'world', 
            'ever', 'get', 'present', 'whole', 
            'right', 'pass', 'high', 'let', 
            'god', 'become', 'child', 'bring', 
            'another', 'father', 'light', 
            'among', 'law', 'mean', 'fall', 
            'turn', 'name', 'nothing', 'whose', 
            'moment', 'general', 'side', 
            'nature', 'away', 'hope', 'use', 
            'subject', 'spirit', 'thy', 
            'character', 'however', 'three', 
            'large', 'keep', 'soon', 'return', 
            'live', 'night', 'hold', 'government', 
            'back', 'person', 'case', 'put', 
            'lay', 'believe', 'hour', 'point', 
            'foot', 'woman', 'sir', 'true', 
            'water', 'cause', 'mother', 'less',
            'always', 'receive', 'course', 
            'home', 'better', 'half', 'order',
            'death', 'arm', 'manner', 'small',
            'within', 'almost', 'follow', 'lady',
            'open', 'voice', 'public', 'meet',
            'party', 'truth', 'want', 'fact', 
            'soul', 'poor', 'object']

normal_len = []
normal_count = []
target_len = []
target_count = []
sense_len = []
sense_count = []
for word in vocab:
    count = model.wv.get_vecattr(word, 'count')
    vec_len = numpy.linalg.norm(model.wv[word])

    if '.' in word:
        w = word.split('.')[0]
        if w in targets:
            target_count.append(count)
            target_len.append(vec_len)
        else:
            sense_count.append(count)
            sense_len.append(vec_len)
    else:
        normal_count.append(count)
        normal_len.append(vec_len)
    
    # if count in range(2500, 7500) and int(vec_len) in range(7, 15):

        # targets.append(word)
        
# print(len(counts))

#%%
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(normal_count, normal_len, c='darkgrey')
plt.scatter(target_count, target_len, c='firebrick', label='target')
plt.scatter(sense_count, sense_len, c='teal', label='sense')
plt.xlabel('Term Frequency')
plt.ylabel('Term Vector Length')
plt.ylim(-.5,18.5)
plt.xlim(-1000, 22000)
plt.legend()
plt.show()
# plt.savefig('test.png')

# print(targets)

# %%
