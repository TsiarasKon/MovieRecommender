# MovieRecommender

### Summary

In this project we build a **content-based recommender system** for movies. In content-based recommendation, it is paramount to find appropriate features for the items at hand (i.e. the movies), based on which we can calculate a reliable similarity metric between them. This is where knowledge bases and knowledge extraction comes in. We build an RDF graph containing all the data from the official IMDb dataset that we deemed most useful and, then, we enrich it with relevant info from other online RDF knowledge bases such as wikidata (https://www.wikidata.org/). Having done that, we are able to use SPARQL in order to query from said knowledge graph all the features we need for our task.

### Content-Based Recommendation

#### What you need to know

In order to calculate similarity between movies we represent each item/movie as a point in some high-dimensional Euclidean space by constructing a real vector from the features of each movie. We can then calculate the similarity between movies by using the **cosine similarity** of their respective feature vectors. 

There are two types of features:
* **Numerical features** (e.g. year): which take up one dimension of item vectors and are normalized in [0, 1].
* **Categorical features** (e.g. genres): which take up as many dimensions as there are possible values for them. We employ a multi-label binary encoding for these features where each element is 1 if the corresponding feature value is valid for the item and 0 otherwise (that is multiple values are possible).

Each feature may not be as important. Furthermore, numerical feature may be overshadowed by categorical features (which take up more "spots") when using this encoding. Therefore, we correspond to each feature a **weight** by which we multiply all its values in the feature vector. The higher the weight of a feature the more "say" this feature has for our final recommendations. These are, of course, hyperparameters of the process.

#### How it works

We can summarize how we have implemented our content-based recommender system with the following flow diagram.

![Flow Diagram of the recommendation process](MovieRecommender.png "Flow Diagram of the recommendation process")

The process can be described as follows:
1. We begin by extracting all the features to use from our enriched RDF graph and **constructing the item feature matrix**. We do this by creating each item vector individually out of the features of the corresponding movie as queried from the RDF graph. Here we need to treat numerical and categorical features appropriately (as previously described) as well as multiply them by their weights.<br>
    Extra care had to be given to reduce the amount of possible values for categorical features with too many possible values (e.g. we reduced the 84.000 actors to 13.000 of them that had made at least 3 movies).
   
Then, each time we get called to make recommendations for a user given ratings of his to movies we do the following:

2. We **create the user vector**, describing the user's **estimated preferences** in movies, by aggregating *appropriately* the item vectors in the item feature matrix that correspond to movies that the user has rated. For numerical features, this aggregation may simply be the sample mean or the median. For categorical features, however, we want to weight each value based on the difference between the user's rating and the average rating (e.g. either fixed to 2.5 out 5 or the average rating that the user has given in his ratings). When this is positive then it is desirable for a movie to have this categorical value in order to be more similar to the user's preferences, whilst when it is negative then it's not. <br>
   
    After much consideration, the formula we ended up using for item feature matrix <i>I</i> (with 0/1 encoding) is the following:<br>

    ![User vector creation for categorical features](UserVector.png "User vector creation for categorical features")
   
   Here, *temperature* is another hyperparameter of the process. Higher *temperature* results in more "extreme" estimations in that less ratings are needed to reach the maximum/minimum value for a feature value (e.g. the less % of comedies one would need to watch for their estimated preferences for comedies to be close to 1 times the weight of the genre feature). <br> Also, note that the weights of each feature are already incorporated in the item feature matrix and that we clip the results of this formula in the range of [-w, w] for each categorical feature with weight w.<br>

3. We modify the item feature matrix (a copy of it) to have an encoding of -1/1 instead of 0/1 for the categorical features. This makes more sense for our impending cosine similarity calculation between user and item vectors since now the item vector's values will conveniently be in the exact same range of [-w, w] for each feature with weight w as the user vector's values.<br>

4. Finally, we calculate the cosine similarity between the user vector and all the item vectors in the modified item feature matrix (with -1/1 encoding) and sort all movies by descending similarity. Our recommender can simply recommend to the user the top N most similar movies that weren't in his user ratings (i.e. that he hasn't watched yet).


### Movie features used

TODO: From **IMDb dataset** these...

TODO: From wikidata enrichment these...

TODO: 2nd enrichment???


### Performance Evaluation

When it comes to recommender system evaluation, the typical metrix to use is the RMSE metric between predicted and true user ratings of movies. However, our method does not predict a rating. It rather just sorts movies in a way that the most relevant should be first. Therefore, in order to evaluate the performance of our recommender we turned to metrics that are used primarily in Information Retrieval such as: **recall@N** and **Mean Average Precision MAP@N**, where "@N" signifies the **cut-off** for our recommendations (i.e. use the first N recommendations).

#### How they work

We deem a recommended movie/item as relevant if its true rating is above a certain threshold (e.g. 3.0, 3.5). 

We calculate **recall@N** as the percentage of relevant items in our top N recommendations (if there are less than N relevant items overall then we fix the percentage accordingly).

We calculate **MAP@N** by first calculating the precision **P(k)** at cut-off k, that is the 
percentage of relevant items in the top k recommendations, for each k from 1 to N if the k-th item is relevant. We then average all the **P(k)** that occurred to get the average precision **AP@N** for a user and take the average of multiple users to get finally get the **MAP@N** value. 

The **MAP@N** is more informative because it takes into account the order in which relevant items were recommended in-between irrelevant items.

A more detailed explaination may be found here: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html 

#### How we applied them

In order to evaluate our recommender system using said metrics we used the **movieLens dataset** (https://grouplens.org/datasets/movielens/), which contains 25.000.000 ratings of 162.000 users in 62.000 movies. For practical reason, only a fraction of these were used in our experiments.

We evaluated both metrics for some reasonable values of n (e.g. 15, 25, 50, etc) and for a bunch of different hyperparameter configurations in order to **tune our hyperparameters**. These primarily include the weights of each feature and the *temperature* hyperparameter, but also some other minor boolean flags for optional adjustments. Below are some experiments we wrote down for a cut-off of 25 and a threshold of 3.5.

| temperature (best of 1, 10, 25, 50 & 100) | w_rating | w_year | w_genres | w_actors | w_ directors | w_series | w_ subjects | w_distributors | RECALL @25 | MAP@25 |
|-------------------------------------------|----------|--------|----------|----------|--------------|----------|-------------|----------------|------------|--------|
| 50                                        | 1        | 1      | 1        | 1        | 1            | 1        | 1           | 1              | 0.8903     | 0.8499 |
| 50                                        | 1        | 0.25   | 1        | 1        | 1            | 1        | 1           | 0.1            | 0.9131     | 0.8879 |
| 50                                        | 1        | 0.25   | 1        | 3        | 2            | 1        | 1           | 0.1            | 0.9272     | 0.9122 |
| 50                                        | 1        | 0.25   | 1        | 3        | 2            | 1        | 2           | 0.1            | 0.9270     | 0.9113 |
| 50                                        | 1        | 0.25   | 1        | 3        | 2            | 1.5      | 2           | 0.1            | 0.9268     | 0.9108 |

For instance, we found that the distributors feature was not as important (based on this dataset at least) as lowering its weight improved our metrics significantly. That being said, if we are not planning on using this system for the same kind of users then it is reasonable to change these weights based on our expectations as well. For example, we believe that the series feature should have a big say (perhaps the biggest) on our recommendations as it is our belief that users who rated highly one or more movies that are part of a series (e.g. Star Wars) would probably also want to watch (and therefore would rate highly) other movies in the same series.  


### Application

TODO: Describe application

TODO: Show sample screenshots from our application


### Future work

Potential future work includes:
* Using **Locality Sensitive Hashing (LSH)** in order to speed up the cosine similarity calculation. With LSH we would not be calculating the similarity between the user vector and *all* the item vectors but only with some item vectors that are expected to probably be more similar to the user's vector.
* Adding more features that may be relevant.
