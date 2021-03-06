import os
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm


def filter_by_subcorpus(data_root_folder, subcorpus_name):
    data_folders = [
        os.path.join(data_root_folder, 'data', folder, subfolder)
        for folder in tqdm(os.listdir(os.path.join(data_root_folder, 'data')), desc="READ FOLDERS: ")
        for subfolder in os.listdir(os.path.join(data_root_folder, 'data', folder))
    ]
    mask = []
    for folder in tqdm(data_folders, desc="FILTER FOLDERS: "):
        with open(os.path.join(folder, 'en.met'), 'r', encoding='utf-8') as myfile:
            mask.append(subcorpus_name in myfile.read().replace('\n', ''))
    return np.array(data_folders)[mask]


def get_tagged_texts(folders, cache_folder):
    if not os.path.exists(os.path.join(cache_folder, 'en.tags.extracted.npy')):
        texts = []
        for folder in tqdm(folders, desc="READ FILES: "):
            with open(os.path.join(folder, 'en.tags'), 'r', encoding='utf-8') as myfile:
                texts.extend(myfile.readlines())
        np.save(os.path.join(cache_folder, 'en.tags.extracted.npy'), texts)
        return texts
    else:
        return np.load(os.path.join(cache_folder, 'en.tags.extracted.npy'))


def get_tagged_texts_as_pd(folders, cache_folder):
    if not os.path.exists(os.path.join(cache_folder, 'en.tags.pd.extracted.csv')):
        columns = [
            'token',
            'pos_tag',
            'lemma',
            'ner_tag',
            'word_net_sense_number',
            'verb_net_roles',
            'semantic_relation',
            'animacy_tag',
            'super_tag',
            'lambda_dsr'
        ]
        tagged_dfs = []

        for folder in tqdm(folders, desc="READ FILES: "):
            tagged_text_df = pd.read_csv(
                os.path.join(folder, 'en.tags'),
                delimiter='\t',
                encoding='utf-8',
                names=columns
            )
            tagged_dfs.append(tagged_text_df)
        df = pd.concat(tagged_dfs, ignore_index=True, sort=False)
        # cache the file
        df.to_csv(os.path.join(cache_folder, 'en.tags.pd.extracted.csv'), index=False)
        return df
    else:
        df = pd.read_csv(os.path.join(cache_folder, 'en.tags.pd.extracted.csv'))
        return df


def create_sub_folders(path):
    folders = path.split('/')
    sub_folder = ''
    for folder in folders:
        sub_folder += folder + '/'
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
