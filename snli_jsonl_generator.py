import json
import pandas as pd

file_path = f"C:/Users/bilal/Dropbox/MS/NLP/Final/biased_set/en_v1.1/bias-eval-set_v1.1/all-words/4-all_v1.1.json"
output_file_path = "C:/Users/bilal/Dropbox/MS/NLP/Final/biased_set/en_v1.1/bias-eval-set_v1.1/all-words/eval.jsonl"

# {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
label_names = ["entailment", "neutral", "contradiction"]

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


with open(file_path, 'r') as json_file:
    json_list = list(json_file)
    results = []
    for json_str in json_list:
        result = json.loads(json_str)

        result['premise'] = result.pop('sentence1')
        result['hypothesis'] = result.pop('sentence2')
        result['label'] = result.pop('label')
        result['label'] = 1

        results.append(result)
    dump_jsonl(results, output_file_path)