from itertools import chain


def generate_ngrams(tokens, n=1):
    """Generate ngrams for particular 'n' from a list of tokens

    Parameters
    ----------
    tokens : List[str]
        List of tokens for which the ngram are to be generated
    n : int
        n for which ngrams are to be generated

    Returns
    -------
    List[str]
        List of ngrams generated

    """
    shifted_tokens = (tokens[i:] for i in range(n))
    tuple_ngrams = zip(*shifted_tokens)
    return (" ".join(i) for i in tuple_ngrams)


def generate_ngrams_range(tokens, ngram_range=(1, 3)):
    """Generates and returns a list of ngrams for all n present in ngram_range.

    Parameters
    ----------
    tokens : List[str]
        List of string tokens for which ngram are to be generated
    ngram_range : List[int]
        List of 'n' for which ngrams are to be generated. For e.g. if
        ngram_range = (1, 4) then it will returns 1grams, 2grams and 3grams

    Returns
    -------
    List[str]
        List of ngrams for each n in ngram_range.

    """
    assert len(ngram_range) == 2, "'ngram_range' should be a tuple" \
                                  " of two elements which is range of numbers"
    return chain(*(generate_ngrams(tokens, i) for i in range(*ngram_range)))
