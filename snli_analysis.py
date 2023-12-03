import mpmath
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import json
import numpy as np
import math

def get_female_occ_list():
    female_stereo = [
        "caretaker",
        "dancer",
        "hairdresser",
        "housekeeper",
        "interior designer",
        "librarian",
        "nanny",
        "nurse",
        "receptionist",
        "registered nurse",
        "secretary",
        "stylist",
        "teacher"
    ]
    return female_stereo


def get_male_occ_list():
    male_stereo = [
        "ambassador",
        "archaeologist",
        "architect",
        "assassin",
        "astronaut",
        "athlete",
        "athletic director",
        "ballplayer",
        "banker",
        "bodyguard",
        "boss",
        "boxer",
        "butcher",
        "captain",
        "carpenter",
        "chancellor",
        "coach",
        "colonel",
        "commander",
        "commissioner",
        "conductor",
        "constable",
        "cop",
        "custodian",
        "dean",
        "dentist",
        "deputy",
        "director",
        "disc jockey",
        "doctor",
        "drummer",
        "economics professor",
        "electrician",
        "farmer",
        "fighter pilot",
        "firefighter",
        "gangster",
        "industrialist",
        "investment banker",
        "janitor",
        "judge",
        "laborer",
        "lawmaker",
        "lieutenant",
        "lifeguard",
        "magician",
        "magistrate",
        "major leaguer",
        "manager",
        "marshal",
        "mathematician",
        "mechanic",
        "minister",
        "mobster",
        "neurologist",
        "neurosurgeon",
        "officer",
        "parliamentarian",
        "pastor",
        "philosopher",
        "physicist",
        "plumber",
        "preacher",
        "president",
        "prisoner",
        "programmer",
        "ranger",
        "sailor",
        "scholar",
        "senator",
        "sergeant",
        "sheriff deputy",
        "skipper",
        "soldier",
        "sportswriter",
        "superintendent",
        "surgeon",
        "taxi driver",
        "technician",
        "trader",
        "trucker",
        "tycoon",
        "vice chancellor",
        "warden",
        "warrior",
        "welder",
        "wrestler"
    ]
    return male_stereo


def get_neutral_occ_list():
    neutral = [
        "accountant",
        "adjunct professor",
        "administrator",
        "adventurer",
        "advocate",
        "aide",
        "alderman",
        "analyst",
        "anthropologist",
        "archbishop",
        "artist",
        "artiste",
        "assistant professor",
        "associate dean",
        "associate professor",
        "astronomer",
        "author",
        "baker",
        "ballerina",
        "barber",
        "barrister",
        "bartender",
        "biologist",
        "bishop",
        "bookkeeper",
        "broadcaster",
        "broker",
        "bureaucrat",
        "businessman",
        "butler",
        "cameraman",
        "campaigner",
        "cardiologist",
        "cartoonist",
        "cellist",
        "chef",
        "chemist",
        "choreographer",
        "cinematographer",
        "civil servant",
        "cleric",
        "clerk",
        "collector",
        "columnist",
        "comedian",
        "commentator",
        "composer",
        "consultant",
        "correspondent",
        "councilor",
        "counselor",
        "critic",
        "curator",
        "dermatologist",
        "detective",
        "diplomat",
        "doctoral student",
        "economist",
        "editor",
        "educator",
        "employee",
        "entertainer",
        "entrepreneur",
        "environmentalist",
        "envoy",
        "epidemiologist",
        "evangelist",
        "fashion designer",
        "filmmaker",
        "financier",
        "fisherman",
        "footballer",
        "foreman",
        "freelance writer",
        "gardener",
        "geologist",
        "goalkeeper",
        "graphic designer",
        "guidance counselor",
        "guitarist",
        "handyman",
        "historian",
        "hitman",
        "illustrator",
        "infielder",
        "inspector",
        "instructor",
        "inventor",
        "investigator",
        "jeweler",
        "journalist",
        "jurist",
        "landlord",
        "lawyer",
        "lecturer",
        "lyricist",
        "marksman",
        "mediator",
        "medic",
        "midfielder",
        "missionary",
        "musician",
        "narrator",
        "naturalist",
        "negotiator",
        "novelist",
        "organist",
        "painter",
        "paralegal",
        "parishioner",
        "pathologist",
        "patrolman",
        "pediatrician",
        "performer",
        "pharmacist",
        "photographer",
        "photojournalist",
        "pianist",
        "planner",
        "plastic surgeon",
        "playwright",
        "poet",
        "politician",
        "pollster",
        "priest",
        "principal",
        "professor",
        "professor emeritus",
        "promoter",
        "proprietor",
        "prosecutor",
        "protester",
        "provost",
        "psychiatrist",
        "psychologist",
        "publicist",
        "radiologist",
        "realtor",
        "researcher",
        "restaurateur",
        "saint",
        "salesman",
        "saxophonist",
        "scientist",
        "screenwriter",
        "sculptor",
        "servant",
        "serviceman",
        "shopkeeper",
        "singer",
        "singer songwriter",
        "socialite",
        "sociologist",
        "solicitor",
        "solicitor general",
        "soloist",
        "stockbroker",
        "strategist",
        "student",
        "substitute",
        "surveyor",
        "swimmer",
        "therapist",
        "treasurer",
        "trooper",
        "trumpeter",
        "tutor",
        "undersecretary",
        "understudy",
        "violinist",
        "writer"
    ]
    return neutral


def get_model_bin(is_trained=True):
    if not is_trained:
        return "C:\\Users\\bilal\\DropBox\\MS\\NLP\\fp-dataset-artifacts\\snli_eval\\trained_model"
    else:
        return "C:\\Users\\bilal\\DropBox\\MS\\NLP\\fp-dataset-artifacts\\snli_eval\\retrained_modelv2"


def get_model(is_trained=True):
    model_name = get_model_bin(is_trained)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return model, tokenizer


def predict_prob(model, tokenizer, premise, hypothesis):
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"])  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    return prediction

def entailment_predict_prob(model, tokenizer, premise, hypothesis):
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"])  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    #return entailment probabilities
    return prediction[0]


def predict_label(model, tokenizer, premise, hypothesis):
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"])  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    l = np.array(prediction)
    return np.argmax(l)


#0 for neutral, 1 for man and 2 for woman
def get_occupation_label(premise):
    found = [w for w in get_female_occ_list() if w in premise]
    if found:
        return 2
    found = [w for w in get_male_occ_list() if w in premise]
    if found:
        return 1
    return 0


def false_positve_eval(file):
    fp_accuracy_by_gender = {0:0, 1:0}
    counts_by_gender = {0:0, 1:0}

    model, tokenizer = get_model()
    with open(file, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            this_label = result['label']
            this_hypothesis = result['hypothesis']
            this_premise = result['premise']
            #gender_occupation_label = get_occupation_label(this_premise)

            #force entailment
            this_label = 0

            #assume male label to initialize
            gender_occupation_label = 0
            if 'woman' in this_hypothesis:
                gender_occupation_label = 1

            counts_by_gender[gender_occupation_label] = counts_by_gender[gender_occupation_label] + 1
            predicted_label = predict_label(model, tokenizer, this_premise, this_hypothesis)
            if predicted_label != this_label:
                fp_accuracy_by_gender[gender_occupation_label] = fp_accuracy_by_gender[gender_occupation_label] + 1

            # replace man by woman or woman by man and switch gender label
            if 'woman' in this_hypothesis:
                this_hypothesis = this_hypothesis.replace('woman', 'man')
                gender_occupation_label = 0
            else:
                this_hypothesis = this_hypothesis.replace('man', 'woman')
                gender_occupation_label = 1

            counts_by_gender[gender_occupation_label] = counts_by_gender[gender_occupation_label] + 1
            predicted_label = predict_label(model, tokenizer, this_premise, this_hypothesis)
            if predicted_label != this_label:
                fp_accuracy_by_gender[gender_occupation_label] = fp_accuracy_by_gender[gender_occupation_label] + 1

    return fp_accuracy_by_gender, counts_by_gender


def entailment_prob_eval(file):
    diff = 0
    counts = 0

    model, tokenizer = get_model()
    with open(file, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            result = json.loads(json_str)
            this_label = result['label']
            this_hypothesis = result['hypothesis']
            this_premise = result['premise']
            #force entailment
            this_label = 0

            #assume male label to initialize
            gender_occupation_label = 0
            if 'woman' in this_hypothesis:
                gender_occupation_label = 1

            counts = counts + 1

            entailment_prob1 = entailment_predict_prob(model, tokenizer, this_premise, this_hypothesis)
            # replace man by woman or woman by man and switch gender label
            if 'woman' in this_hypothesis:
                this_hypothesis = this_hypothesis.replace('woman', 'man')
            else:
                this_hypothesis = this_hypothesis.replace('man', 'woman')

            entailment_prob2 = entailment_predict_prob(model, tokenizer, this_premise, this_hypothesis)

            diff = diff + abs(entailment_prob1 - entailment_prob2)

    return diff/counts

def main():
    # model, tokenizer = get_model()
    # premise = "the investment banker holds up a cell phone in a crowd"
    # hypothesis = "the woman holds up a cell phone in a crowd"
    # prediction = predict_prob(model, tokenizer, premise, hypothesis)
    # print(prediction)

    # accuracy_by_gender, counts_by_gender = false_positve_eval('biased_eval_neutral.jsonl')
    # print(accuracy_by_gender)
    # print(counts_by_gender)

    diff = entailment_prob_eval('biased_eval_neutral.jsonl')
    print(diff)

if __name__ == "__main__":
    main()
