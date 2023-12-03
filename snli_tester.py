from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch


# Selecting the pre-trained model and tokenizer
model_name = "C:\\Users\\bilal\\DropBox\\MS\\NLP\\fp-dataset-artifacts\\snli_eval\\trained_model"
#model_name = "C:\\Users\\bilal\\DropBox\\MS\\NLP\\fp-dataset-artifacts\\snli_eval\\retrained_modelv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

premise = "the investment banker rides a pink motorcycle that has bull horns on the handle bars"
hypothesis = "the womman rides a pink motorcycle that has bull horns on the handle bars"

#premise = "the nanny sitting on a bench with a statue and holding its hand"
#hypothesis = "the woman sitting on a bench with a statue and holding its hand"

#the man sitting on a bench with a statue and holding its hand

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")

output = model(input["input_ids"])  # device = "cuda:0" or "cpu"

prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)