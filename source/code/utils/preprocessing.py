import string

from nltk.corpus import stopwords
from tqdm.autonotebook import tqdm
import numpy as np


def filtrations(df, with_dots=False):
    def iob3bio(tags):
        entity = False
        curr_tag = ''
        for i in tqdm(range(len(tags)), desc='IOB to BIO: '):
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

    if with_dots:
        tqdm.pandas(desc="Punctuation without dots: ")
        df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation.replace('.', ''))]
    else:
        tqdm.pandas(desc="Punctuation with dots: ")
        df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation)]

    stop_words = set(stopwords.words('english'))

    df = df[~df.lemma.isin(stop_words)]

    df = df[df.word_net_sense_number != 'O']

    df = df[df.ner_tag != '[]']

    df = df[df.ner_tag != '']

    df = df[df.lemma != '']

    df = df[df.token != '']

    tqdm.pandas(desc="Target tags: ")

    df.ner_tag = df.ner_tag.progress_apply(lambda x: str(x).split('-')[0] if str(x) != 'O' else str(x))

    df.ner_tag = iob3bio(df.ner_tag.values)

    tqdm.pandas(desc="")

    return df


def crf_pre_processing(df):
    columns = [
        'token',
        'pos_tag',
        'lemma',
        'word_net_sense_number',
        'verb_net_roles',
        'semantic_relation',
        'animacy_tag',
        'super_tag',
        'semantic_relation_tagged',
        'animacy_tagged',
        'lambda_dsr_len',
        'word_sense_exists',
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
    tqdm.pandas(desc="Semantic relation: ")
    df['semantic_relation_tagged'] = df.semantic_relation.progress_apply(lambda x: int(str(x) != 'O'))

    tqdm.pandas(desc="Animacy tagged: ")
    df['animacy_tagged'] = df.animacy_tag.progress_apply(lambda x: int(str(x) != 'O'))

    tqdm.pandas(desc="Lambda-DSR len: ")
    df['lambda_dsr_len'] = df.lambda_dsr.progress_apply(lambda x: len(str(x)))

    tqdm.pandas(desc="Word sense: ")
    df['word_sense_exists'] = df.word_net_sense_number.progress_apply(lambda x: int(int(x) > 0))

    tqdm.pandas(desc="Is title: ")
    df['is_title'] = df.token.progress_apply(lambda x: int(str(x).istitle()))

    tqdm.pandas(desc="Contains digits: ")
    df['contains_digits'] = df.token.progress_apply(lambda x: int(not str(x).isalpha()))

    tqdm.pandas(desc="Word len: ")
    df['word_len'] = df.token.progress_apply(lambda x: len(str(x)))

    tqdm.pandas(desc="")

    return df
