from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy

path = f'/home/clare/Data/word_vectors/semeval/sense/ccoha1.vec'
model = Word2Vec.load(path)
vocab = list(model.wv.index_to_key)

lens = []
counts = []
targets = []
for word in vocab:
    count = model.wv.get_vecattr(word, 'count')
    vec_len = numpy.linalg.norm(model.wv[word])
    
    if count in range(2500, 7500) and int(vec_len) in range(7, 15):
        counts.append(count)
        lens.append(vec_len)
        targets.append(word)
        
print(len(counts))
    
plt.scatter(counts, lens)
plt.savefig('test.png')

print(targets)
