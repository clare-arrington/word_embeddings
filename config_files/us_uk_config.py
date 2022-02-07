#%%
def get_us_uk_targets(path, get_us=False, get_uk=False):
    targets = []
    ## Get dissimilar
    with open(f"{path}/dissimilar.txt") as fin:
        dis = fin.read().split()
        targets.extend(dis)

    ## Get similar
    with open(f"{path}/similar.txt") as fin:
        sim = fin.read().strip()
        for pair in sim.split("\n"):
            uk_word, us_word = pair.split()
            if get_us:
                targets.append(us_word)
            elif get_uk:
                targets.append(uk_word)

    ## Get spelling
    with open(f"{path}/spelling.txt") as fin:
        sp = fin.read().strip()
        for pair in sp.split("\n"):
            uk_word, us_word = pair.split()
            if get_us:
                targets.append(us_word)
            elif get_uk:
                targets.append(uk_word)

    return targets

## TODO: can I do something about the min issue
## I should just filter it out myself for other words I think
## Min_word_count from paper: 100 BNC, 300 COCA

data_path = "/data/arrinj"

config = {
    "data_path"     : data_path,
    "min_count"     : 100,
    "vector_size"   : 100,
    "corpora"       : {
        "bnc" : {
            "targets" : get_us_uk_targets(f"{data_path}/corpus_data/us_uk/truth", get_uk=True)
        },
        "coca" : {
            "targets" : get_us_uk_targets(f"{data_path}/corpus_data/us_uk/truth", get_us=True)
        }
    }
}

file_paths = {
        "export_file"       : "word_vectors/us_uk/{vector_type}/{corpus_name}.vec",
        "sense_file"        : "masking_results/us_uk/{corpus_name}/sense_sentences.pkl",

        "target_file"       : "corpus_data/us_uk/subset/target_sentences.pkl",
        "stored_non_t_file" : "word_vectors/us_uk/extra_data/{corpus_name}_normal_sents.pkl",

        "non_target_file"   : "corpus_data/us_uk/subset/{corpus_name}_non_target.dat",
        "stored_t_file"     : "word_vectors/us_uk/extra_data/{corpus_name}_target_sents.pkl"
    }
#%%
