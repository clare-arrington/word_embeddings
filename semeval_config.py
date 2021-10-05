#%%
dataset = "semeval"
corpus_name = "ccoha2"
run = "sense"

min_count = 20
num_sents = 1000000

with open('/home/clare/Data/corpus_data/semeval/targets.txt') as fin:
    targets = fin.read().split()
    targets = [target.split('_')[0] for target in targets]

# print(f'{len(targets)} targets loaded')

## TODO: I could pass the below params into some make config thing
config = {
    "dataset": dataset, 
    "corpus_name" : corpus_name,
    "run": run, 
    "min_count" : min_count, 
    "num_sents" : num_sents,
    "targets" : targets,

    "non_target_path" : f'/home/clare/Data/corpus_data/semeval/subset/{corpus_name}_non_target.dat',
    "sampled_non_target_path" : f'/home/clare/Data/word_vectors/semeval/extra_data/{corpus_name}_sents_{num_sents}.dat',
    
    "export_path" : f'/home/clare/Data/word_vectors/{dataset}/{run}',
    "sense_path" :  f'/home/clare/Data/masking_results/{dataset}/{corpus_name}/sentences/',
    "target_path" : f'/home/clare/Data/corpus_data/{dataset}/subset/{corpus_name}_target_sents.csv',
    "subset_path" : f'/home/clare/Data/masking_results/{dataset}/{corpus_name}/clusters'
    }
