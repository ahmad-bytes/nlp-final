import json
import pandas as pd

file_path = "squad_eval\eval_predictions.jsonl"
output_file_path = "squad_eval\eval_predictions.csv"

# with open(file_name, 'r') as json_file:
#     json_list = list(json_file)
#
# for json_str in json_list:
#     result = json.loads(json_str)
#     print(f"result: {result}")
#     print(isinstance(result, dict))
#


jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
jsonObj.to_csv(output_file_path)

#"C:\Users\bilal\Dropbox\MS\NLP\fp-dataset-artifacts\snli_eval\biased_eval.jsonl"