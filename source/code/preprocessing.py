from tqdm.autonotebook import tqdm
from tqdm import trange
import numpy as np
import string

tqdm.pandas()


def filtrations(df):
    tqdm.pandas(desc="Punctuation: ")

    df = df[df.lemma.progress_apply(lambda lemma: str(lemma) not in string.punctuation)]
    df = df[df.ner_tag != '[]']

    tqdm.pandas(desc="Target tags: ")
    df.ner_tag = df.ner_tag.progress_apply(lambda x: str(x).split('-')[0] if str(x) != 'O' else str(x))
    tqdm.pandas(desc="")

    return df


def crf_filtration_and_pre_processing(df):
    sentences, tags = [], []
    curr_sent, curr_tags = [], []
    columns = [
        'token',
        'pos_tag',
        'lemma',
        'word_net_sense_number',
        'verb_net_roles',
        'semantic_relation',
        'animacy_tag',
        'super_tag',
        'ner_tagged',
        'semantic_relation_tagged',
        'animacy_tagged',
        'lambda_dsr_len',
        'word_sense_exists',
        'is_title',
        'contains_digits',
        'word_len'
    ]

    df = df[df.ner_tag != '[]']

    df.ner_tag = df.ner_tag.apply(lambda x: str(x).split('-')[0] if str(x) != 'O' else str(x))

    for i in trange(1, len(df)):
        if df.iloc[i]['token'] not in string.punctuation:
            curr_sent.append(dict(zip(columns, df.iloc[i][columns].values.tolist())))
            curr_tags.append(df.iloc[i]['ner_tag'])
        else:
            sentences.append(curr_sent)
            tags.append(curr_tags)
            curr_sent, curr_tags = [], []
    return sentences, tags


def additional_features(df):

    df = df[df.word_net_sense_number != 'O']
    df.word_net_sense_number = df.word_net_sense_number.astype(np.int64)

    tqdm.pandas(desc="NER tagged: ")
    df['ner_tagged'] = df.ner_tag.progress_apply(lambda x: int(str(x) != 'O'))

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