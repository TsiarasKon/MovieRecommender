import flask
import numpy as np
from bidict import bidict
from flask import jsonify, request

from recommendation import recommend_for_single_user, use_wikidata
from utils import load_dict

app = flask.Flask('movie_recommender_backend')
app.config["DEBUG"] = True

# Should we not recommend movies that the user has already seen
ignore_seen = True


@app.route('/recommend', methods=['POST'])
def post_answer():
    input_json = request.get_json(force=True)
    if app.config["DEBUG"]:
        print('data from client:', input_json)

    # decode JSON input
    user_ratings = {}
    for rating in input_json['movieRating']:
        user_ratings[rating['tconst']] = rating['rating']
    topK = input_json['recommendationsNum']

    if app.config["DEBUG"]:
        print('topK =', topK)
        print('user_ratings =', user_ratings)

    recommendations = recommend_for_single_user(user_ratings, item_features, movie_pos, feature_lens, topK=topK, verbose=True, ignore_seen=ignore_seen)

    if app.config["DEBUG"]:
        print('Recommendations:\n', recommendations)

    return jsonify(recommendations)


if __name__ == '__main__':
    # load saved features
    print('Loading previously saved features', 'with wikidata' if use_wikidata else '')
    item_features = np.load(f'item_features{"_wikidata" if use_wikidata else ""}.npy')
    movie_pos: bidict = load_dict(f'movie_pos{"_wikidata" if use_wikidata else ""}')
    feature_lens: dict = load_dict(f'feature_lens{"_wikidata" if use_wikidata else ""}')
    print('Done.')

    # run app
    app.run()
