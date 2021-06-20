# MovieRecommender

#### Summary

In this project we build a **content-based recommender system** for movies. In content-based recommendation, it is paramount to find appropriate features for the items at hand (i.e. the movies), based on which we can calculate a reliable similarity metric between them. This is where knowledge bases and knowledge extraction comes in. We build an RDF graph containing all the data from the official IMDb dataset that we deemed useful and, then, we enrich it with relevant info from other online RDF knowledge bases such as wikidata (https://www.wikidata.org/). Having done that, we are able to use SPARQL in order to query from said knowledge graph all the features we need for our task.
