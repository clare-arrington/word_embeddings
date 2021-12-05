#%%
from base_wv import main, make_config
import pandas as pd

def get_targets(data_path, corpus_name, run, from_masking=False):
    if from_masking:
        target_data = f'{data_path}/masking_results/semeval/{corpus_name}/target_sense_labels.pkl'
        target_data = pd.read_pickle(target_data)
        targets = [t for t in target_data.target.unique()]
    else:
        with open(f'{data_path}/corpus_data/semeval/targets.txt') as fin:
            targets = fin.read().split()
            if run == 'sense':
                targets = [target.split('_')[0] for target in targets]
    
    print(f'{len(targets)} targets loaded')
    print(targets[:5])
    return targets

def make_wv(corpus_name, run, load_data, save_data, dataset="semeval", data_path="/home/clare/Data"):
    min_count = 50
    vector_size = 300
    pattern = r'[a-z]+_[a-z]{2}|[a-z]+\.\d|[a-z]+'

    targets = get_targets(data_path, corpus_name, run)

    config = make_config(
            dataset, corpus_name, run, min_count, 
            vector_size, targets, 
            load_data, save_data,
            data_path, pattern=pattern)

    main(config)
    
make_wv("1800s", "new", False, True)
make_wv("2000s", "new", False, True)

make_wv("1800s", "sense", True, False)
make_wv("2000s", "sense", True, False)

print('\nDone!')
# %%
