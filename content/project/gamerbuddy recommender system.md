+++

title = "GamerBuddy Video Game Recommender"

+++
## **1 INTRODUCTION**
We will be building a recommender system that recommends games based on the desciption of the game you are interested in. GamerBuddy recommender system is an extension of GamerBuddy [search engine](https://varunsrao.netlify.com/project/gamerbuddy-search-engine/) and [classifier](https://varunsrao.netlify.com/project/gamerbuddy-video-game-classifier/).


## **2 FEATURES**

The [recommender systems](https://towardsdatascience.com/how-to-build-from-scratch-a-content-based-movie-recommender-with-natural-language-processing-25ad400eb243) of most websites usually gets description of the products present in your search history and recommends games based on it. In short, it tries to find similarities between the product that you'r intersted in and other products that are present in the collection. 

Lets say you like the game Mortal Kombat and it is present in your search history. The recommender system takes description of Mortal Kombat and recommends a similar game.

Here we will be building a content based recommender system that takes description of a game of a particular genre and recommends a similar game of the same genre.

The datasets that is used by the recommender system is available [here](http://jmcauley.ucsd.edu/data/amazon/links.html). Video game meta data is used for building our search engine, GamerBuddy. Make sure you turn the files in the string to list of jsons.

## **3 IMPLEMENTATION**

The data set contains set of json string. Each json string in meta data file represents a product.
To build a search engine, the following aspects need to be implemented.
  
### a. Processing your [dataset](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy/gamerbuddy_dataset). 
     Turn the .zip files into list of json string using following code. Each json string contains details of a product.
  ```python
  
  # generate list of json strings.
  #op meta file=> [ {"asin" : "value", "title": "value" , "description": "value", "imUrl": "value"........ }, .....]
  
  def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
      yield json.dumps(eval(l))

  if __name__ == "__main__":
    path = input("Enter .zip path: \n")
    final_path = path.replace(".gz", "")
    f = open(final_path, 'w+')
    s = []
    for l in parse(path):
        s .append(l)
    f.write("[\n{}\n]".format(",\n".join(s)))
    f.close()

  ```    
  The list of jsons in meta file will next be converted into json of format shown below.
  
  ```python
  # product_details
  
  { 
  "asin1": { "description": "value",
             "title": "value",
             "imUrl": "value",
             .
             .
             .},
  "asin2": { "description": "value",
              "title": "value",
              "imUrl": "value",
              .
              .
              .},
    .
    .
    .
  }
  
  ```
  
  The above json can be converted to dictionary of the same form by using json.dumps(). 
  The format change can be performed by executing the following code.
  
  ```python
  
  def split_file(file_name):
  #file_name :: metadata file
    #for meta file
    product_dict = {}
    input_file = open(file_name+".json", "r+")
    output_file = open("product_details.json", "w+")
    input_json = json.loads(input_file.read())
    for product in input_json:
        product_dict[product["asin"]] = product
        del(product["asin"])
    output_file.write(json.dumps(product_dict))
    output_file.close() 

  
  ```
  
###  b. Generating a cosine dot product of given description and the description of the games in collection.
   We need to games whose description are similar to that of the given description we need to get the [cosine rank](https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/) of all the videogames. The steps to implement cosine ranking system is described in implmentation of [search engine](https://varunsrao.netlify.com/project/gamerbuddy-search-engine/). Using that we will build a class that will perform cosine ranking system. We will later create an object of that class to genrate the cosine products.  


Note that we all so need to get term frequency of terms for every videogame and also inverse document frequency of the term in the entire videogame collections. The steps to get term frequency and inverse document frequency is explained in construction of [search engine](https://varunsrao.netlify.com/project/gamerbuddy-search-engine/)


The class for generating cosine ranking system is given below
```python   
      
      from math import sqrt
import json

class RankGenerator(object):

    def __init__(self, tf_details, idf_details):
        self.tf_details = tf_details
        self.idf_details = idf_details
        self.product_list = None
        self.query_words_list = None

    def generate_ranks_for_products(self, query_words_list, product_list):
        product_tf_idf = self._generate_tf_idf_value_for_products(product_list, query_words_list)
        query_tf_idf = self._generate_tf_idf_value_for_query(query_words_list)
        sorted_products = sorted(self._generate_cosine_value(product_list, product_tf_idf, query_tf_idf).items(),
                                 key=lambda kv: kv[1], reverse=True)
        print("sorted result :: \n{}".format(str(sorted_products)))
        return [product_det[0] for product_det in sorted_products]

    def _generate_tf_idf_value_for_products(self, product_list, query_words_list):
        print("info :: generating tf-idf value for products...")
        product_tf_idf = self._generate_tf_idf_without_normalization(product_list, query_words_list)
        return self._nomalise_tf_idf(product_tf_idf)


    def _generate_tf_idf_without_normalization(self, product_list, query_words_list):
        print("info :: generate tf-id without normalization...")
        product_tf_idf = dict()
        for asin in product_list:
            product_tf_idf[asin] = {}
            for term in query_words_list:
                tf = self.tf_details[term].get(asin, 0)
                idf = self.idf_details[term]
                product_tf_idf[asin][term] = tf * idf
        # print("unnormalised")
        # print(product_tf_idf)
        return product_tf_idf

    def _nomalise_tf_idf(self, product_tf_idf):
        print("info :: normalise tf-idf...")
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            magnitude = sqrt(sum([temp_product[term] ** 2 for term in temp_product]))
            # print("magnitude {}".format(str(magnitude)))
            if magnitude == 0:
                continue
            for term in temp_product:
                temp_product[term] /= magnitude
        print(product_tf_idf)
        return product_tf_idf

    def _generate_tf_idf_value_for_query(self, query_words_list):
        print("info :: generating tf-idf value for query...")
        query_tf_idf = self.generate_unnormalized_tf_idf_for_query(query_words_list)
        return self.normalize_tf_idf_for_query(query_tf_idf)

    def generate_unnormalized_tf_idf_for_query(self, query_words_list):
        print("info :: generating unnormalised tf-idf value for query...")
        query_tf_idf = dict()
        total_len = 0
        for term in query_words_list:
            total_len += len(query_words_list[term])
        for term in query_words_list:
            tf = float(sum(query_words_list[term])) / total_len
            idf = float(self.idf_details[term])
            query_tf_idf[term] = tf * idf
        return query_tf_idf

    def normalize_tf_idf_for_query(self, query_tf_idf):
        print("info :: generating tf-idf for query...")
        magnitude = sqrt(sum([query_tf_idf[term] ** 2 for term in query_tf_idf]))
        for term in query_tf_idf:
            query_tf_idf[term] /= magnitude
        # print("query tf-idf {}".format(str(query_tf_idf)))
        return query_tf_idf

    def _generate_cosine_value(self, product_list, product_tf_idf, query_tf_idf):
        print("info :: generating cosine values")
        product_rank = dict()
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            for term in query_tf_idf:
                product_rank[product] = temp_product[term] * query_tf_idf[term]
            product_rank[product] *= product_list[product]
        print("info :: cosine result {}".format(str(product_rank)))
        print(json.dumps(sorted(product_rank.items(),
                                 key=lambda kv: kv[1], reverse=True), indent=4, sort_keys=True))
        return product_rank

    def generate_ranks_for_recommender_system(self, query_words_list, product_list):
        product_tf_idf = self._generate_tf_idf_value_for_products(product_list, query_words_list)
        query_tf_idf = self._generate_tf_idf_value_for_query(query_words_list)
        sorted_products = sorted(self._generate_cosine_value_for_recommender_system(product_tf_idf, query_tf_idf).items(),
                                 key=lambda kv: kv[1], reverse=True)
        print("sorted result :: \n{}".format(str(sorted_products)))
        return sorted_products


    def _generate_cosine_value_for_recommender_system(self, product_tf_idf, query_tf_idf):
        print("info :: generating cosine values")
        product_rank = dict()
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            for term in query_tf_idf:
                product_rank[product] = temp_product[term] * query_tf_idf[term]
        print("info :: cosine result {}".format(str(product_rank)))
        print(json.dumps(sorted(product_rank.items(),
                                 key=lambda kv: kv[1], reverse=True), indent=4, sort_keys=True))
        return product_rank
  ```
  

### c. Putting them all of them together.

Now, we need to use cosine ranking object to find similarities between the given description text of a video game and that of all the videogames in our collection. But finding the similarities with all the games in the collection is time consuming and redundant. So we will only find the similarities with the games of the same genre in the collection as that of the given videogame. 

to implement a recommender we will perform the follwoing steps.

  1. Get input description of the games and the genres (max of 3. ex: fighting, action, adventure) the game belongs to.
  2. Find all the games in the collection that belongs to the given genres
  3. Find the all unique terms and the number of times the terms in occur in the given input description.
  4. Find cosine similarities between the given descripions and that of video games of the same genre in the collection.
  5. Find the top 5 games  that are similar to the given video game
  
  
```python
import json

class GamerBuddyRecommender(object):

    def __init__(self, product_terms, genre_products, rank_gen):
        self.rank_gen = rank_gen
        self.genre_products_details = genre_products
        self.product_terms = product_terms
        pass

    def recommend_games(self, product, genres):
        recommended_games = list()
        for genre in genres:
            query_terms_list = self.product_terms[product].get("title", []) + self.product_terms[product].get(
                "description", [])
            query_terms_list = self._get_query_index(query_terms_list)
            product_list = self.genre_products_details[genre]
            recommended_games = recommended_games + self.rank_gen.generate_ranks_for_recommender_system(
                query_words_list=query_terms_list, product_list=product_list)
        recommended_games = [product_det[0] for product_det in recommended_games]
        return self._get_top_five_recommended_products(recommended_games)

    def _get_top_five_recommended_products(self, product_list):
        count = 0
        recommended_games = []
        product_len = len(product_list)
        while len(recommended_games) < 5 and count < product_len:
            if product_list[count] not in recommended_games:
                recommended_games.append(product_list[count])
                count += 1
        return recommended_games

    def _get_query_index(self, query_terms_list):
        query_words_index = {}
        for index, term in enumerate(query_terms_list, start=1):
            if term not in query_words_index:
                query_words_index[term] = [index]
            else:
                query_words_index[term].append(index)
        return query_words_index


```



## **SOURCE CODE**
The entire source code for this can be found [here](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy). 

## **REFERRENCE**
[Tf-Idf and Cosine similarity](https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/)










