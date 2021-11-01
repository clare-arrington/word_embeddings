#%%
from base_wv import full_file, main, make_config

for corpus_name in ["ccoha1_extra"]:
    corpus_name = 'ccoha2'
    dataset = "semeval"
    run = "sense"

    min_count = 20 
    vector_size = 300
    load_data = False
    save_data = True

    with open('/home/clare/Data/corpus_data/semeval/targets.txt') as fin:
        targets = fin.read().split()
        if run == 'sense':
            targets = [target.split('_')[0] for target in targets]

    # print(f'{len(targets)} targets loaded')

    config = make_config(
        dataset, corpus_name, run, min_count, 
        vector_size, targets, load_data, save_data)

    main(config)
    # full_file(config)

print('\nDone!')
# %%
