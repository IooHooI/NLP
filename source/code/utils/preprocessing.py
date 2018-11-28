import string
import numpy as np

from tqdm.autonotebook import tqdm
from nltk.corpus import stopwords


def iob3bio(tags):
    entity = False
    curr_tag = ''
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
    tags = ['-'.join(str(x).split('-')[0:2]) if str(x) != 'O' else str(x) for x in tags]
    return tags


def filtrations(df, with_dots=False):
    stopWords = set(stopwords.words('english'))

    if with_dots:
        tqdm.pandas(desc="WITH DOTS: ")

        df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation.replace('.', ''))]
    else:
        tqdm.pandas(desc="WITHOUT DOTS: ")

        df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation)]

    mask = (~df.lemma.isin(stopWords)) & (df.ner_tag != '[]') & (df.ner_tag != '') & (df.lemma != '') & (df.token != '')

    df = df[mask]

    tqdm.pandas(desc="")

    return df


def additional_features(df):
    tqdm.pandas(desc="IS TITLE: ")

    df['is_title'] = df.token.progress_apply(lambda x: int(str(x).istitle()))

    tqdm.pandas(desc="CONTAINS DIGITS: ")

    df['contains_digits'] = df.token.progress_apply(lambda x: int(not str(x).isalpha()))

    tqdm.pandas(desc="WORD LENGTH: ")

    df['word_len'] = df.token.progress_apply(lambda x: len(str(x)))

    tqdm.pandas(desc="SUFFIX: ")

    df['suffix'] = df.lemma.progress_apply(lambda x: str(x)[-3:])

    tqdm.pandas(desc="PREFIX: ")

    df['prefix'] = df.lemma.progress_apply(lambda x: str(x)[0:3])

    tqdm.pandas(desc="")

    df['prev_pos_tag'] = np.roll(df.pos_tag.values, 1)

    df['prev_is_title'] = np.roll(df.is_title.values, 1)

    df['prev_contains_digits'] = np.roll(df.contains_digits.values, 1)

    df['prev_word_len'] = np.roll(df.word_len.values, 1)

    df['prev_suffix'] = np.roll(df.suffix.values, 1)

    df['prev_prefix'] = np.roll(df.prefix.values, 1)

    df['next_pos_tag'] = np.roll(df.pos_tag.values, -1)

    df['next_is_title'] = np.roll(df.is_title.values, -1)

    df['next_contains_digits'] = np.roll(df.contains_digits.values, -1)

    df['next_word_len'] = np.roll(df.word_len.values, -1)

    df['next_suffix'] = np.roll(df.suffix.values, -1)

    df['next_prefix'] = np.roll(df.prefix.values, -1)

    return df
