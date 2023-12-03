import json
from glob import glob
import os

label_names = ["entailment", "neutral", "contradiction"]

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


def main():
    file = 'train_0.json'
    out_file = 'out_train_0.json'
    with open(file, 'r') as json_file:
        json_list = list(json_file)
        results = []
        for json_str in json_list:
            result = json.loads(json_str)

            result['premise'] = result.pop('sentence1')
            result['hypothesis'] = result.pop('sentence2')
            result['label'] = result.pop('label')
            this_label = 0
            result['label'] = this_label
            results.append(result)
            #add everything as an entailment
        dump_jsonl(results, out_file)


if __name__ == "__main__":
    main()

