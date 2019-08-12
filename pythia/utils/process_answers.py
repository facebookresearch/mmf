# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os
import re


class ProcessAnswer:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }

        self.manual_map = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]
        self.period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile("(?<=\d)(\,)+(?=\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.comma_strip, inText) is not None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.period_strip.sub("", outText, re.UNICODE)
        return outText

    def process_digit_article(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manual_map.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def process_answer(self, answer):
        answer = self.word_tokenize(answer)
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.process_punctuation(answer)
        answer = self.process_digit_article(answer)
        return answer

    def get_score(self, occurences):
        if occurences == 0:
            return 0
        elif occurences == 1:
            return 0.3
        elif occurences == 2:
            return 0.6
        elif occurences == 3:
            return 0.9
        else:
            return 1

    def multiple_replace(self, text, wordDict):
        for key in wordDict:
            text = text.replace(key, wordDict[key])
        return text

    def filter_answers(self, answers_dset, min_occurence):
        """This will change the answer to preprocessed version
        """
        occurence = {}
        answer_list = []
        for ans_entry in answers_dset:
            gtruth = ans_entry["multiple_choice_answer"]
            gtruth = self.process_answer(gtruth)
            if gtruth not in occurence:
                occurence[gtruth] = set()
            occurence[gtruth].add(ans_entry["question_id"])
        for answer in occurence.keys():
            if len(occurence[answer]) >= min_occurence:
                answer_list.append(answer)

        print(
            "Num of answers that appear >= %d times: %d"
            % (min_occurence, len(answer_list))
        )
        return answer_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="input train annotationjson file",
    )
    parser.add_argument(
        "--val_annotation_file",
        type=str,
        required=False,
        help="input val annotation json file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="output directory, default is current directory",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=0,
        help="the minimum times of answer occurrence \
                              to be included in vocabulary, default 0",
    )
    args = parser.parse_args()

    train_annotation_file = args.annotation_file
    out_dir = args.out_dir
    min_freq = args.min_freq

    answer_file_name = "answers_vqa.txt"
    os.makedirs(out_dir, exist_ok=True)

    train_answers = json.load(open(train_annotation_file, "r"))["annotations"]
    answers = train_answers

    if args.val_annotation_file is not None:
        val_annotation_file = args.val_annotation_file
        val_answers = json.load(open(val_annotation_file, "r"))["annotations"]
        answers = train_answers + val_answers

    answer_processor = ProcessAnswer()
    answer_list = answer_processor.filter_answers(answers, min_freq)
    answer_list = [t.strip() for t in answer_list if len(t.strip()) > 0]
    answer_list.sort()

    if "<unk>" not in answer_list:
        answer_list = ["<unk>"] + answer_list

    answer_file = os.path.join(out_dir, answer_file_name)
    with open(answer_file, "w") as f:
        f.writelines([w + "\n" for w in answer_list])
