#%%
from base_wv import main, make_config
import itertools

def make_wv(corpus_name, vector_type, dataset="news", data_path="/data/arrinj"):
    min_count = 45
    vector_size = 300

    if vector_type == 'normal':
        load_data = False
        save_data = True
    else:
        load_data = True
        save_data = False

    with open(f'{data_path}/corpus_data/{dataset}/targets.txt') as fin:
        targets = fin.read().split()

    paths = {
            'corpus_path': f'corpus_data/{dataset}/subset/{corpus_name}/',
            'target_path': f'corpus_data/{dataset}/subset/{corpus_name}/'
        }

    wv_config = make_config(
            dataset, corpus_name, vector_type, min_count, 
            vector_size, targets, load_data, save_data, 
            data_path, paths)

    main(wv_config)
    print('\n\n')
    
corpora = ['conspiracy', 'mainstream'] # , 
vector_types = ['normal', 'sense']

for corpus_name, vector_type in itertools.product(corpora, vector_types):
    print(f'\n\n\n======= Making {corpus_name} {vector_type} wordvector =======')
    make_wv(corpus_name, vector_type)

print('Done!')

#%%
