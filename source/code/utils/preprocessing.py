import string

from tqdm.autonotebook import tqdm
from nltk.corpus import stopwords


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

    tqdm.pandas(desc="")

    return df
