#%%
from base_wv import main

for corpus_name in ["ccoha1", "ccoha2"]:
    dataset = "semeval"
    run = "sense"

    min_count = 20
    vector_size = 300
    load_data = True
    save_data = False

    with open('/home/clare/Data/corpus_data/semeval/targets.txt') as fin:
        targets = fin.read().split()
        if run == 'sense':
            targets = [target.split('_')[0] for target in targets]

    # print(f'{len(targets)} targets loaded')

    ## TODO: I could pass the below params into some make config func
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

        "non_target_file" : f'/home/clare/Data/corpus_data/semeval/subset/{corpus_name}_non_target.dat',
        "sampled_non_target_file" : f'/home/clare/Data/word_vectors/semeval/extra_data/{corpus_name}_sents.dat',
        "export_file" : f'/home/clare/Data/word_vectors/{dataset}/both_{run}/{corpus_name}_both.vec',
        
        "sense_path" :  f'/home/clare/Data/masking_results/{dataset}/all_1/sense_sentences.csv',
        "target_path" : f'/home/clare/Data/corpus_data/{dataset}/subset/target_sentences.csv',
        }

    main(config)
    # full_file(config)

print('\nDone!')
# %%