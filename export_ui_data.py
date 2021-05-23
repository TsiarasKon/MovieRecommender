import json

from rdf import data_loading


def df_to_json():
    print("Converting df to json...")
    movies_df, _ = data_loading()
    movies_df.drop(columns=['titleType', 'originalTitle', 'endYear'], inplace=True)
    movies_df['genres'].apply(lambda g: g.split(',') if g else [])
    # movies_df.to_csv('movies.tsv', sep='\t', index_label='tconst')
    return movies_df.rename_axis('tconst').reset_index().to_json(orient='records')


if __name__ == '__main__':
    filename = "movies.json"
    json_data = df_to_json()
    print(f"Writing to '{filename}'...")
    with open(filename, 'w') as f:
        json.dump(json_data, f)
