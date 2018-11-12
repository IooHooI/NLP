import string

from tqdm.autonotebook import tqdm
import numpy as np


def iob3bio(tags):
    entity = False
    curr_tag = ''
    tags = [str(x).split('-')[0] if str(x) != 'O' else str(x) for x in tags]
    for i in tqdm(range(len(tags)), desc='IOB TO BIO: '):
        if tags[i] != 'O' and not entity:
            entity = True
            curr_tag = tags[i]
            tags[i] = 'B-{}'.format(tags[i])
        elif tags[i] != 'O' and tags[i] == curr_tag and entity:
            tags[i] = 'I-{}'.format(tags[i])
        elif tags[i] != 'O' and tags[i] != curr_tag and entity:
            curr_tag = tags[i]
            tags[i] = 'B-{}'.format(tags[i])
        else:
            entity = False
    return tags


def filtrations(df, with_dots=False):
    if with_dots:
        tqdm.pandas(desc="WITH DOTS: ")
        df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation.replace('.', ''))]
    else:
        tqdm.pandas(desc="WITHOUT DOTS: ")
        df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation)]

    mask = (df.ner_tag != '[]') & (df.ner_tag != '') & (df.lemma != '') & (df.token != '')

    df = df[mask]

    tqdm.pandas(desc="")

    return df


def crf_pre_processing(df):
    columns = [
        'token',
        'pos_tag',
        'lemma',
        'is_title',
        'contains_digits',
        'word_len'
    ]
    df[df.lemma == '.'] = '%'
    X, y = df[columns].values, df['ner_tag'].values
    X, y = np.split(X, np.argwhere(X[:, 0] == '%').flatten()), np.split(y, np.argwhere(y == '%').flatten())
    for i in range(1, max(len(X), len(y))):
        X[i] = X[i][1:]
        y[i] = y[i][1:]
    X = [[dict(zip(columns, word)) for word in sentence] for sentence in X]
    return X, y


def additional_features(df):
    tqdm.pandas(desc="IS TITLE: ")
    df['is_title'] = df.token.progress_apply(lambda x: int(str(x).istitle()))

    tqdm.pandas(desc="CONTAINS DIGITS: ")
    df['contains_digits'] = df.token.progress_apply(lambda x: int(not str(x).isalpha()))

    tqdm.pandas(desc="WORD LENGTH: ")
    df['word_len'] = df.token.progress_apply(lambda x: len(str(x)))

    tqdm.pandas(desc="")

    return df
