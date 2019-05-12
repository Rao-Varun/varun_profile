+++

title = "GamerBuddy, a video game searchengine, genre classifier, and a videogame recommender"

+++
## **1 INTRODUCTION**
GamerBuddy, a project that implements a search engine that takes query and presents related games, a classifier that predicts genres, and recommends games to you based on your interested games.

## **2 FEATURES**

GamerBuddy performs the following tasks in the following order

  1. Take query from user.
  2. Search games related to query and present it to the users.
  2. Present description of the game, which user selected from the list of games. The classifier will instantly predict the genre based on the description of the game and present it to the user. 
  3. Recommend similar games to user on the description page. 
  
## **3 IMPLEMENTATION** 

The following steps need to be implemented

  
### a. Building a search engine object to return list of games.

We need a search engine that takes query from the user and gives us a list of games that are related to the query.  

The complete guide to building a gamerbuddy search engine is given [here](https://varunsrao.netlify.com/project/gamerbuddy-search-engine/). I have a given a complete details on how I've implemented it. Note that we need a class that does the job of the search engine. You can refer that link to implement your own search engine. 


We will implement GamerBuddy as a class. We will build our GamerBuddy class to create a search engine object and call the necessary functions of the search engine object to provide us the list of games that are related to the input query. Each of the games in a result list is represented as a dictionary, containing product id as key and value as another dictionary containing image, title and description of the game.

```python
class GamerBuddy(object):

    def __init__(self):
        self.product_details = self._get_dict_from_json_files("gamerbuddy_dataset/product_details.json") #contains product details like title, description, and img link of the game
        self.inverted_index = self._get_dict_from_json_files("input_generator/ii.json")
        self.tf_detail = self._get_dict_from_json_files("input_generator/term_frequency.json") #term frequency
        self.idf_details = self._get_dict_from_json_files("input_generator/inverse_document_frequency.json") #inverse document frequency
        self.product_terms = self._get_dict_from_json_files("input_generator/int_file.json") # dictionary containing product as key and terms in the product's title and description as values
        self.rank_generator = RankGenerator(tf_details=self.tf_detail, idf_details=self.idf_details)
        self.search_engine = GamerBuddySearchEngine(product_details=self.product_details,
                                                    inverted_index=self.inverted_index,
                                                    rank_generator=self.rank_generator)

    def get_gamer_buddy_search_engine(self):
          return self.search_engine

    def search_query(self, query):
        return self.search_engine.search_product_in_dataset(query)
        

    def _get_dict_from_json_files(self, json_file):
        print("info :: loading {}...".format(json_file))
        with open(json_file) as json_obj:
            return json.load(json_obj)


```

Let us also implement a function in our object that takes videogame id as input and returns a dictionary containg the detail of a games such as title, image and description of that game.

```python

    def get_product_details(self, product_id):
        product = {}
        for key in ["title", "description", "imUrl"]:
            product[key] = self.product_details[product_id].get(key, "{} not available".format(key))
        product["genre"] = self.classify_video_games(product_id)
        return product
        
```


  
### b. Building a genre classifier to classify games. 
Once we select a game from the list of games, we will build a genre classifier that takes description of the game as input and predict the genre of the game.

The complete guide on building our GamerBuddy classifier is given [here](https://varunsrao.netlify.com/project/gamerbuddy-video-game-classifier/). You can use this as a referrence to build your classifier class.

Now, we need to create a class that creates an object of the classifier class. We will also need to implement the function 
that calls the necessary function of the classifier object that can tell us which genre a given game belongs to.

```python

class GamerBuddy(object):

    def __init__(self):
        self.product_details = self._get_dict_from_json_files("gamerbuddy_dataset/product_details.json")
        self.inverted_index = self._get_dict_from_json_files("input_generator/ii.json")
        self.tf_detail = self._get_dict_from_json_files("input_generator/term_frequency.json")
        self.idf_details = self._get_dict_from_json_files("input_generator/inverse_document_frequency.json")
        self.product_terms = self._get_dict_from_json_files("input_generator/int_file.json")
        self.genre_products_details = self._get_dict_from_json_files("recommender_data/genre_products.json")
        self.rank_generator = RankGenerator(tf_details=self.tf_detail, idf_details=self.idf_details)
        self.search_engine = GamerBuddySearchEngine(product_details=self.product_details,
                                                    inverted_index=self.inverted_index,
                                                    rank_generator=self.rank_generator)
        self.classifier = GamerBuddyClassifier()
        
    def get_gamer_buddy_search_engine(self):
        return self.search_engine

    def search_query(self, query):
        return self.search_engine.search_product_in_dataset(query)

    def get_product_details(self, product_id):
        product = {}
        for key in ["title", "description", "imUrl"]:
            product[key] = self.product_details[product_id].get(key, "{} not available".format(key))
        product["genre"] = self.classify_video_games(product_id)
        return product

    def get_gamer_buddy_classifier(self):
        return self.classifier

    def classify_video_games(self, product_id):
        return self.classifier.predict_product_genre(
            self.product_terms[product_id].get("title", []) + self.product_terms[product_id].get("description", []))

    def _get_dict_from_json_files(self, json_file):
        print("info :: loading {}...".format(json_file))
        with open(json_file) as json_obj:
            return json.load(json_obj)


```
  
### c. Building a recommender object to recommend games.
We've come to the third aspect of GamerBuddy. We build a recommender object in this section. The complete guide to building a recommender is given [here](https://varunsrao.netlify.com/project/gamerbuddy-recommender-system/). Our recommender object is used take the description and the genre of that game as input and find games that are similar to it. 

We will use this object in our GamerBuddy class in the following way.


```python

import copy
import json
from classifier import GamerBuddyClassifier
from recommender import GamerBuddyRecommender
from search_engine import GamerBuddySearchEngine
from input_processor.rank_generator import RankGenerator


class GamerBuddy(object):

    def __init__(self):
        self.product_details = self._get_dict_from_json_files("gamerbuddy_dataset/product_details.json")
        self.inverted_index = self._get_dict_from_json_files("input_generator/ii.json")
        self.tf_detail = self._get_dict_from_json_files("input_generator/term_frequency.json")
        self.idf_details = self._get_dict_from_json_files("input_generator/inverse_document_frequency.json")
        self.product_terms = self._get_dict_from_json_files("input_generator/int_file.json")
        self.genre_products_details = self._get_dict_from_json_files("recommender_data/genre_products.json")
        self.rank_generator = RankGenerator(tf_details=self.tf_detail, idf_details=self.idf_details)
        self.search_engine = GamerBuddySearchEngine(product_details=self.product_details,
                                                    inverted_index=self.inverted_index,
                                                    rank_generator=self.rank_generator)
        self.classifier = GamerBuddyClassifier()
        self.recommender = GamerBuddyRecommender(product_terms=self.product_terms,
                                                 genre_products=self.genre_products_details,
                                                 rank_gen=self.rank_generator)

    def get_gamer_buddy_search_engine(self):
        return self.search_engine

    def search_query(self, query):
        return self.search_engine.search_product_in_dataset(query)

    def get_product_details(self, product_id):
        product = {}
        for key in ["title", "description", "imUrl"]:
            product[key] = self.product_details[product_id].get(key, "{} not available".format(key))
        product["genre"] = self.classify_video_games(product_id)
        return product

    def get_gamer_buddy_classifier(self):
        return self.classifier

    def classify_video_games(self, product_id):
        return self.classifier.predict_product_genre(
            self.product_terms[product_id].get("title", []) + self.product_terms[product_id].get("description", []))

    def get_gamer_buddy_recommender_system(self):
        return self.recommender

    def get_recommended_products(self, product_id, genre):
        recommended_poducts_id = self.recommender.recommend_games(product_id, genre)
        recommended_poducts = {}
        for product_id in recommended_poducts_id:
            recommended_poducts[product_id] = {}
            for key in ["title", "description", "imUrl"]:
                recommended_poducts[product_id][key] = self.product_details[product_id].get(key, "{} not available".format(key))
        return recommended_poducts

    def _get_dict_from_json_files(self, json_file):
        print("info :: loading {}...".format(json_file))
        with open(json_file) as json_obj:
            return json.load(json_obj)


```


### d. Creating a GamerBuddy Server.

We will use FLASK to write a script that will take requests from client and provide response back to it. The server script will have to handle the following scenarios. 
#### 1. Keeping GamerBuddy ready.
Before the server script takes any request, we will have to create an instance of our GamerBuddy class. The GamerBuddy object will create an instance of our search engine, classifier and recommender classes. 

```python

from flask import Flask, render_template, request
from gamer_buddy import GamerBuddy

app = Flask(__name__)

@app.before_first_request
def setup_gamer_buddy():
    print("app info :: Setting up gamer buddy...")
    global gamer_buddy
    gamer_buddy = GamerBuddy()

```

#### 2. Displaying the index page.
Once the server is ready with the GamerBuddy object, the server should display the index page of our search engine.

```python

from flask import Flask, render_template, request
from gamer_buddy import GamerBuddy

app = Flask(__name__)

@app.before_first_request
def setup_gamer_buddy():
    print("app info :: Setting up gamer buddy...")
    global gamer_buddy
    gamer_buddy = GamerBuddy()

@app.route("/")
def index(): #index page.
    print("app info :: building index page...")
    return render_template("index.html")

```

The html code for the index page is given below 

```html

<html>
  <head>
    <title>GamerBuddy</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="author" content="colorlib.com">
    <link href="https://fonts.googleapis.com/css?family=Poppins" rel="stylesheet" />
    <link href="{{ url_for('static',filename='css/main.css') }}" rel="stylesheet" />
  </head>
  <body>
    <div class="s01">
      <form name="query" action="display_result", method="POST">
        <fieldset>
          <legend id="logo" align="center">GamerBuddy</legend>
        </fieldset>
        <div class="inner-form">
          <div class="input-field first-wrap">
            <input id="search" name="search" type="text" placeholder="What are you looking for?" />
          </div>
          <div class="input-field third-wrap">
            <button class="btn-search" type="submit" value="Search"><i class="fa fa-search"></i></button>
          </div>
        </div>
      </form>
    </div>
  </body>
</html>


```

#### 3. Handling queries from user.
The server should now handle queries from the client. The python Flask code for this given below:



  
  2. for each term, find all the products that contain the term.
  3. find products that are in common for all query terms.
  4. filter products second time based on which key[desciption, title] the term exits. If a product contains all the query terms then those terms should exist in the same key. 
  5. get multiplier for all the matched product based on the order of the query terms that were found in the document.
  
```python

  #getting terms in query sentence
  def get_query_words(self, query):
      print("info :: splitting query sentence...")
      return [term for term in word_tokenize(query.lower(), language="english") if
              term not in stop_words]

  #getting process  
  def process_query(self, query):
    query_list = get_query_words(query)
    term_asin_list = get_asin_from_inverted_index()
    common_asin_list = get_common_asin_products(term_asin_list)
    asin_list = get_asin_containing_query(common_asin_list)
    return _get_multipliers_for_those_containing_exact_query(asin_list, query)

  #get products that contain the terms in its description or title. Each term has its own 
  def get_asin_from_inverted_index(self):
    term_asin_list = {}
    for term in query_list:
        term_asin_list[term] = inverted_index.get(term)
    return term_asin_list


  #filter products second time based on which key[desciption, title] all the term exits. 
  def get_common_asin_products(self, term_asin_list):
        print("info :: getting products(asin) containing all term words...")
        common_asin = set()
        for term in term_asin_list:
            if len(common_asin) == 0:
                common_asin = set(term_asin_list[term].keys())
            else:
                common_asin = common_asin & set(term_asin_list[term].keys())
        return list(common_asin)

  #get multiplier for all the matched product based on the order of the query terms that were found in the document.
  def _get_multipliers_for_those_containing_exact_query(asins, query):
    print("info :: getting asin list....")
    query = query.lower()
    asin_list = {}
    for asin in asins:
        asin_list[asin] = 1
        for key in asins[asin]:
            if self.product_details[asin].get(key) and query in self.product_details[asin][key].lower():
                asin_list[asin] *= 10 ** (2*self.product_details[asin].get(key).lower().count(query))
    return asin_list

```

For the final implementation of our search engine, we will be needing a an object of a class that does the processing of query for us. The class can be implementated in the following way


```python

import json
from os.path import dirname, abspath, join
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class ProcessQuery(object):

    def __init__(self, product_details, inverted_index):
        self.product_details = product_details
        self.inverted_index = inverted_index
        self.stop_words = list(set(stopwords.words("english")))

    def _load_json_from_file(self, json_file):
        print("info :: loading inverted_index...")
        project_loc = dirname(dirname(abspath(__file__)))
        json_file = join(join(project_loc, "gamerbuddy_inputs"), json_file)
        with open(json_file) as json_obj:
            return json.load(json_obj)

    def _get_query_words(self, query):
        print("info :: splitting query sentence...")
        return [term for term in word_tokenize(query.lower(), language="english") if
                term not in self.stop_words]

    def process_query(self, query): #main function
        print("info :: begin processing query...")
        self.query_list = self._get_query_words(query)
        print("info :: query list - {}".format(str(self.query_list)))
        term_asin_list = self._get_asin_from_inverted_index()
        common_asin_list = self._get_common_asin_products(term_asin_list)
        asin_list = self._get_asin_containing_query(common_asin_list)
        return self._get_multipliers_for_those_containing_exact_query(asin_list, query)
    #
    def _get_asin_from_inverted_index(self):
        print("info :: obtaining inverted index...")
        term_asin_list = {}
        for term in self.query_list:
            term_asin_list[term] = self.inverted_index.get(term)
        return term_asin_list

    def _get_common_asin_products(self, term_asin_list):
        print("info :: getting products(asin) containing all term words...")
        common_asin = set()
        for term in term_asin_list:
            print("term :: {}....".format(str(term)))
            if len(common_asin) == 0:
                common_asin = set(term_asin_list[term].keys())
            else:
                common_asin = common_asin & set(term_asin_list[term].keys())
        return list(common_asin)

    def _get_asin_containing_query(self, common_asin_list):
        print("info :: getting keys for products that common for all terms.....")
        common_keys_for_asin = {}
        for asin in common_asin_list:
            for term in set(self.query_list):
                if asin not in common_keys_for_asin:
                    common_keys_for_asin[asin] = set(self.inverted_index[term][asin].keys())
                else:
                    common_keys_for_asin[asin] = common_keys_for_asin[asin] & set(
                        self.inverted_index[term][asin].keys())
            common_keys_for_asin[asin] = list(common_keys_for_asin[asin])
        return common_keys_for_asin


    def _get_multipliers_for_those_containing_exact_query(self, asins, query):
        print("info :: getting asin list....")
        query = query.lower()
        asin_list = {}
        for asin in asins:
            asin_list[asin] = 1
            for key in asins[asin]:
                if self.product_details[asin].get(key) and query in self.product_details[asin][key].lower():
                    asin_list[asin] *= 10 ** (2*self.product_details[asin].get(key).lower().count(query))
                    print("info count :: asin {} :: {}".format(asin,
                                                               str(self.product_details[asin].get(key).count(query))))
            print("info :: asin {} :: {}".format(asin, str(asin_list[asin])))
        print("asin list :: {} \n".format(str(asin_list)))
        return asin_list

```



### e. Generate [cosine rank](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/rank_generator.py) between query term and product document. 

Consine Rank is used to rank the search result that we obtained in step d. We use [vector space model ](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf). Each term in the document collections represents an axis in that vector space model and a document is represented as a vector. Each vectors have their terms as its components and the value of each component  is given by that terms tf-idf. Similarly queries are represented as vectors. The tf of a term in query sentence is given by the number of times that term occur in the sentence and for idf, we refer the same idf table we created for our document collects(product list)

Cosine Ranking is computed in the following way. 

  1. Normalise document vectors
  2. Normalise query vectors
  3. Perform dot product of both vectors(cosine value).
  4. Multiply the multiplier of that product with its dot product.
  5. Sort the vectors in descending order. The higher the product value, the more relevant the document is to the query. 

##### 1. Code for normalised documents vectors is given below

~~~python
    from math import sqrt
    
    #product_list = search result occured in step d. product_list = { "asin1" : multiplier_1, "asin2" : multiplier_2, ....}
    #query_words_list = dictionary containing all the terms as keys and their respective frequency in the query sentence as their values

    def generate_tf_idf_value_for_products(product_list, query_words_list):
        product_tf_idf = self._generate_tf_idf_without_normalization(product_list, query_words_list)
        return self._nomalise_tf_idf(product_tf_idf)

    def generate_tf_idf_without_normalization(product_list, query_words_list):
        product_tf_idf = dict()
        for asin in product_list:
            product_tf_idf[asin] = {}
            for term in query_words_list:
                tf = tf_details[term][asin]
                idf = idf_details[term]
                product_tf_idf[asin][term] = tf * idf
        return product_tf_idf

    def nomalise_tf_idf(product_tf_idf):
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            magnitude = sqrt(sum([temp_product[term] ** 2 for term in temp_product]))
            for term in temp_product:
                temp_product[term] /= magnitude
        return product_tf_idf

~~~

##### 2. Code for normalised documents vectors is given below

~~~python

    def generate_tf_idf_value_for_query(query_words_list):
        query_tf_idf = generate_unnormalized_tf_idf_for_query(query_words_list)
        return normalize_tf_idf_for_query(query_tf_idf)

    def generate_unnormalized_tf_idf_for_query(query_words_list):
        query_tf_idf = dict()
        total_len = 0
        for term in query_words_list:
            total_len += len(query_words_list[term])
        for term in query_words_list:
            tf = float(sum(query_words_list[term])) / total_len
            idf = float(self.idf_details[term])
            query_tf_idf[term] = tf * idf
        return query_tf_idf

    def normalize_tf_idf_for_query(query_tf_idf):
        magnitude = sqrt(sum([query_tf_idf[term] ** 2 for term in query_tf_idf]))
        for term in query_tf_idf:
            query_tf_idf[term] /= magnitude
        return query_tf_idf

~~~


##### 3 & 4. Code for generating dot product of documents and query and multiply the multiplier of that product with its dot product.

~~~python

    def generate_cosine_value(product_tf_idf, query_tf_idf):
        product_rank = dict()
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            for term in query_tf_idf:
                product_rank[product] = temp_product[term] * query_tf_idf[term]
            product_rank[product] *= product_list[product]
        return product_rank

~~~

For building our search engine we need to build an object of a class that implements all the cosine rank generating system. I have implemented it in the following way.

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
        return product_tf_idf

    def _nomalise_tf_idf(self, product_tf_idf):
        print("info :: normalise tf-idf...")
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            magnitude = sqrt(sum([temp_product[term] ** 2 for term in temp_product]))
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

## **FLOW OF SEARCH ENGINE**
The search engine we built will work in the following way:

1. Get the input query from the user 
2. Process the query and get list of documents that contain the query terms
3. Perform cosine ranking system and find which documents are similar to the query.
4. Return the result of cosine ranking system to the user

The implementation of search engine class is given below:

```python

from collections import OrderedDict

from input_processor.process_query import ProcessQuery
from input_processor.rank_generator import RankGenerator
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class GamerBuddySearchEngine(object):

    def __init__(self, product_details, inverted_index, rank_generator):
        self.product_details = product_details
        self.inverted_index = inverted_index
        self.stop_words = list(set(stopwords.words("english")))
        self.query_processor = ProcessQuery(product_details=self.product_details, inverted_index=self.inverted_index)
        self.rank_generator = rank_generator

    def _get_product_details(self, json_file):
        print("info :: loading {}...".format(json_file))
        with open(json_file) as json_obj:
            return json.load(json_obj)

    def search_product_in_dataset(self, query):
        query_term_list = self._get_query_terms(query)
        product_list = self.query_processor.process_query(query)
        print("query words present in the following document {}".format(str(product_list)))
        query_index = self._get_query_index(query_term_list)
        result = self.rank_generator.generate_ranks_for_products(query_index, product_list)
        # print(result)
        return self._get_required_product_details(result)

    def _get_query_index(self, query_terms_list):
        query_words_index = {}
        for index, term in enumerate(query_terms_list, start=1):
            if term not in query_words_index:
                query_words_index[term] = [index]
            else:
                query_words_index[term].append(index)
        return query_words_index

    def _get_query_terms(self, query):
        print(query)
        return [term for term in word_tokenize(query.lower(), language="english") if
                term not in self.stop_words]

    def _get_required_product_details(self, result):
        product_result = OrderedDict()
        for product in result:
            product_result[product] = dict()
            for key in ["title", "description", "imUrl"]:
                product_result[product][key] = self.product_details[product].get(key, "{} not available".format(key))
        return product_result


if __name__ == "__main__":
    product_details = open("product_details.json", "r+").read()
    inverted_index = open("inverted_index.json", "r+").read()
    tf_details = open("tf_details.json", "r+").read()
    idf_details = open("idf_details.json", "r+").read()
    rank_generator = RankGenerator(tf_details=tf_details, idf_details=idf_details)
    se = GamerBuddySearchEngine(product_details=product_details, inverted_index=inverted_index,
                                rank_generator=rank_generator)
    while (True):
        query = input("Enter query \n")
        if query == "exit":
            exit()
        result = se.search_product_in_dataset(query)
        print(result)


```


## **SOURCE CODE**
The entire source code for this can be found [here](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy). 

## **REFERRENCE**

[How to build a search engine](http://aakashjapi.com/fuckin-search-engines-how-do-they-work/)

[Amazon product data](http://jmcauley.ucsd.edu/data/amazon/links.html)

[inverted index](https://en.wikipedia.org/wiki/Inverted_index)

[tf-idf and cosine similariry](https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/)

[Scoring, term weighting and the vector space model](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf)






