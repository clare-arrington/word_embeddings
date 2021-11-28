#%%
from base_wv import main, make_config

dataset = "news"
run = "sense"
data_path = '/data/arrinj'

min_count = 25
vector_size = 300
load_data = False
save_data = True

with open(f'{data_path}/corpus_data/{dataset}/targets.txt') as fin:
    targets = fin.read().split()

# corpus_name = "alternative"
# slice_num = 0

for corpus_name in ['alternative', 'mainstream']:
    for slice_num in range(0, 6):

        config = make_config(
                dataset, corpus_name, run, min_count, 
                vector_size, targets, load_data, save_data, data_path, slice_num)
        
        main(config)

        print('\n\n')

print('Done!')

#%%
