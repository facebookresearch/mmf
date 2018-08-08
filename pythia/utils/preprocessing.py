import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def text_tokenize(sentence):
    sentence = sentence.lower()
    sentence = (
        sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s'))
    tokens = SENTENCE_SPLIT_REGEX.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens
