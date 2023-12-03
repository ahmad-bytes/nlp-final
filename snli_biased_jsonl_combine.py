import json
from glob import glob
import os

#walk_dir = "C:/Users/bilal/Dropbox/MS/NLP/Final/biased_set/en_v1.1/competency_set/train/**"
#out_file  = "C:/Users/bilal/Dropbox/MS/NLP/Final/biased_set/en_v1.1/competency_set/train/biased_train.jsonl"

walk_dir = "C:/Users/bilal/Dropbox/MS/NLP/Final/biased_set/en_v1.1/bias-controlled-set_v1.1/train/**"
out_file  = "C:/Users/bilal/Dropbox/MS/NLP/Final/biased_set/en_v1.1/bias-controlled-set_v1.1/train/biased_train.jsonl"

def main():
    files = [f for f in glob(walk_dir, recursive=True) if os.path.isfile(f)]
    with open(out_file, "w") as ofile:
        for file in files:
            with open(file, 'r') as json_file:
                json_list = list(json_file)
                for json_str in json_list:
                    ofile.write(json_str)


if __name__ == "__main__":
    main()

