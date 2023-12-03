from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch


# Selecting the pre-trained model and tokenizer
model_name = "C:\\Users\\bilal\\--\\MS\\NLP\\fp-dataset-artifacts\\squad_eval\\trained_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=True)


questions = ["Which NFL team represented the AFC at Super Bowl 50?"]
context = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion
 Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third
 Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area
 at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary"
 with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl
 game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo
 could prominently feature the Arabic numerals 50.."""

for question in questions:
    inputs = tokenizer(question, context, return_tensors="pt")

    outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

    print(predict_answer_tokens)
    answer = tokenizer.decode(predict_answer_tokens)

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")