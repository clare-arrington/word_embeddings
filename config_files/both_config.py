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

print('\nDone!')
# %%
