#%%
from base_wv import full_file, main, make_config

for corpus_name in ["ccoha1_extra"]:
#    corpus_name = 'ccoha2'
    dataset = "semeval"
    run = "sense"

    min_count = 20 
    vector_size = 300
    load_data = False
    save_data = True

    target_data = f'/home/clare/Data/masking_results/semeval/{corpus_name}/target_sense_labels.csv'
    target_data = pd.read_csv(target_data, usecols=['target'])
    targets = [t for t in target_data.target.unique()]
    # print(f'{len(targets)} targets loaded')

    config = make_config(
        dataset, corpus_name, run, min_count, 
        vector_size, targets, load_data, save_data)

    main(config)
    # full_file(config)

print('\nDone!')
# %%
