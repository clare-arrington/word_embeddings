def get_us_uk_targets(path, get_us=False, get_uk=False):
    targets = []
    ## Get dissimilar
    with open(f'{path}/dissimilar.txt') as fin:
        dis = fin.read().split()
        targets.extend(dis)

    ## Get similar
    with open(f'{path}/similar.txt') as fin:
        sim = fin.read().strip()
        for pair in sim.split('\n'):
            uk_word, us_word = pair.split()
            if get_us:
                targets.append(us_word)
            elif get_uk:
                targets.append(uk_word)

    return targets

dataset = "us_uk"
corpus_name = "bnc"
run = "new"

min_count = 50
num_sents = 1000000

input_path = '/home/clare/Data/corpus_data/us_uk'

if corpus_name == 'bnc':
    targets = get_us_uk_targets(f'{input_path}/truth', get_uk=True)
elif corpus_name == 'coca':
    targets = get_us_uk_targets(f'{input_path}/truth', get_us=True)

config = {
    "dataset": dataset, 
    "corpus_name" : corpus_name,
    "run": run, 
    "min_count" : min_count, 
    "num_sents" : num_sents,
    "targets" : targets,

    "non_target_path" : f'/home/clare/Data/corpus_data/{dataset}/subset/{corpus_name}_non_target.dat',
    "sampled_non_target_path" : f'/home/clare/Data/word_vectors/{dataset}/extra_data/{corpus_name}_sents_{num_sents}.dat',
    
    "export_path" : f'/home/clare/Data/word_vectors/{dataset}/{run}',
    "sense_path" :  f'/home/clare/Data/masking_results/{dataset}/{corpus_name}/sentences/',
    "target_path" : f'/home/clare/Data/corpus_data/{dataset}/subset/{corpus_name}_target_sents.csv',
    "subset_path" : f'/home/clare/Data/masking_results/{dataset}/{corpus_name}/clusters'
    }
