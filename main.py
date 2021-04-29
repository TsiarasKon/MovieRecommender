import pandas as pd
import numpy as np

imdb_data_folder = 'imdb_data/'
name_basics_file = 'name.basics.tsv'
tconst_files = [
    'title.basics.tsv',
    # 'title.crew.tsv',
    # 'title.episode.tsv',
    # 'title.principals.tsv',
    'title.ratings.tsv'
]

movielens_data_folder = 'movielens_data/'

pd.set_option('display.max_columns', None)


def data_loading():
    # load each file with tconst (title id) as index
    print('Loading IMDB data')
    all_dfs = []
    for file in tconst_files:
        df = pd.read_csv(imdb_data_folder + file, index_col='tconst',  # usecols=[], dtype={'col': 'UInt32'}
                         sep='\t', encoding='utf-8',
                         keep_default_na=False, na_values=['\\N'])
        all_dfs.append(df)

    # combine all into one big fat DataFrame
    print('concatenating...')
    full_df = pd.concat(all_dfs, axis=1)
    full_df = full_df[(full_df['startYear'] >= 1960) & (full_df['startYear'] <= 2020)]
    print('done')

    # fix NA and types afterwards as it is not supported for read_csv
    full_df['numVotes'] = full_df['numVotes'].fillna(0).astype(np.uint16)
    full_df['isAdult'] = full_df['isAdult'].astype(np.bool)
    full_df['startYear'] = full_df['startYear'].fillna(0).astype(np.uint16)
    full_df['endYear'] = full_df['endYear'].fillna(0).astype(np.uint16)

    print(full_df)


data_loading()
