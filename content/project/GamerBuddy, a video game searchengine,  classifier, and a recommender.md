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

The index screen for the one i built looks like this


![index page](/img/index.png)




#### 3. Handling queries from user.
The server should now handle queries from the client. Everytime a client writes a query and clicks the submit button, the query is sent to the server with the usl /display_result as a POST request. The server extracts the query from the POST request and then calls necessary function of GamerBuddy to provide a list of games that match the query. The result is then displayed to the client. The python Flask code to handle this is given below:

```python 

from flask import Flask, render_template, request
from gamer_buddy import GamerBuddy

app = Flask(__name__)


@app.route("/")
def index():
    print("app info :: building index page...")
    return render_template("index.html")


@app.before_first_request
def setup_gamer_buddy():
    print("app info :: Setting up gamer buddy...")
    global gamer_buddy
    gamer_buddy = GamerBuddy()


@app.route("/display_result", methods=["POST"])
def result_page():
    print("app info :: fetching query display")
    result = request.form["search"]
    product_details = gamer_buddy.search_query(result)
    return render_template("display_result.html", product_details=product_details)


```

The html code to display the result is given below.

```html

<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
    <meta name="author" content="colorlib.com">
    <link href="https://fonts.googleapis.com/css?family=Poppins" rel="stylesheet"/>
    <link href="{{ url_for('static',filename='css/search_result.css') }}" rel="stylesheet"/>
    <title>GamerBuddy</title>
</head>
<body>
<div id="head">
    <p class="logo"><a href="/"><strong>GamerBuddy</strong></a></p>
    <form class="example" action="display_result" , method="POST">
        <input type="text" placeholder="Search.." name="search">
        <button type="submit"><i class="fa fa-search"></i></button>
    </form>
</div>
<div>

    {% for key1,product in product_details.items() %}
        <div class="search_result">
            <p class="img">
                <a href="get_product?id={{ key1 }}" class = "product_link">
                {% if product["imUrl"] == "imUrl not available" %}
                    <img src="{{ url_for('static', 'images/missing_image.jpg') }}" alt="missing image">
                {% else %}
                    <img src="{{ product["imUrl"] }}" alt="missing image">
                {% endif %}
                </a>
            </p>
            <div class="result_box">
                <p class="description">{{ product["title"] }}</p>
            </div>
        </div>

    {% endfor %}

</div>
</body>
</html>

```

The page that I built to displays result looks like this

![Result list screen](/img/result_list.png)


#### 4. Display a game from the result list.

A client can choose any of the games from the result list, presented by the server. When a client clicks a game, a GET request is sent to the user along with the id of the video game. We need to display the complete description, genre, and the image of the game. Along with this, we will also display recommended games to our clients.

```python
from flask import Flask, render_template, request
from gamer_buddy import GamerBuddy

app = Flask(__name__)


@app.route("/")
def index():
    print("app info :: building index page...")
    return render_template("index.html")


@app.before_first_request
def setup_gamer_buddy():
    print("app info :: Setting up gamer buddy...")
    global gamer_buddy
    gamer_buddy = GamerBuddy()


@app.route("/display_result", methods=["POST"])
def result_page():
    print("app info :: fetching query display")
    result = request.form["search"]
    product_details = gamer_buddy.search_query(result)
    return render_template("display_result.html", product_details=product_details)


@app.route("/get_product", methods=["GET"])
def get_product_details():
    result = request.args.get("id")
    print("app info :: fetching {} details".format(result))
    product_details = gamer_buddy.get_product_details(result)
    print("app info :: product {} details\n{} ".format(result, str(product_details)))
    print("app info :: fetching recommender result...")
    recommender_result = gamer_buddy.get_recommended_products(product_id=result,
                                                              genre=product_details.get("genre"))
    print("app info :: recommended products \n{}".format(str(recommender_result)))
    return render_template("display_product.html", product_details={result: product_details}, recommender_result=recommender_result)


if __name__ == '__main__':
    app.run(debug=True)



        
```

The html code to display the video game details is given below

```html

<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
    <meta name="author" content="colorlib.com">
    <link href="https://fonts.googleapis.com/css?family=Poppins" rel="stylesheet"/>
    <link href="{{ url_for('static',filename='css/search_result.css') }}" rel="stylesheet"/>
    <title>GamerBuddy</title>
</head>
<body>
<div id="head">
    <p class="logo"><a href="/"><strong>GamerBuddy</strong></a></p>
    <form class="example" action="display_result" , method="POST">
        <input type="text" placeholder="Search.." name="search">
        <button type="submit"><i class="fa fa-search"></i></button>
    </form>
</div>
<div>

    {% for key1,product in product_details.items() %}
        <div class="search_result">
            <p class="img">
                <a href="get_product?id={{ key1 }}" class="product_link">
                    {% if product["imUrl"] == "imUrl not available" %}
                        <img src="{{ url_for('static', 'images/missing_image.jpg') }}" alt="missing image">
                    {% else %}
                        <img src="{{ product["imUrl"] }}" alt="missing image">
                    {% endif %}
                </a>
            </p>
            <div class="result_box">
                <p class="description"><strong>{{ product["title"] }}</strong></p>
                </br>
                <p class="description">
                    <strong>Genre:</strong> {{ product["genre"][0] }}, {{ product["genre"][1] }}, {{ product["genre"][2] }}
                </p>
                </br>
                <p class="description"><strong>Description</strong></br>{{ product["description"] }}</p>
            </div>
        </div>
    {% endfor %}
    <div class="search_result"><h2>  Games you may like</h2></div>
    {% for key1,product in recommender_result.items() %}
        <div class="search_result">
            <p class="img">
                <a href="get_product?id={{ key1 }}" class="product_link">
                    {% if product["imUrl"] == "imUrl not available" %}
                        <img src="{{ url_for('static', 'images/missing_image.jpg') }}" alt="missing image">
                    {% else %}
                        <img src="{{ product["imUrl"] }}" alt="missing image">
                    {% endif %}
                </a>
            </p>
            <div class="result_box">
                <p class="description"><strong>{{ product["title"] }}</strong></p>
            </div>
        </div>
    {% endfor %}
</div>
</body>
</html>

```

The page that I Built to display the product details looks like this

![Product description screen](/img/product_description.png)



## **Deploying GamerBuddy**
I have my GamerBuddy app running on PythonAnywhere. 
You can place your code as it is on PythonAnywhere and create a new virtual environment.
You can find my script running on [http://varun2828.pythonanywhere.com/](http://varun2828.pythonanywhere.com/)

## **SOURCE CODE**
The entire source code for this can be found [here](https://github.com/Rao-Varun/gamerbuddy). 

## **REFERRENCE**

[How to build a search engine](http://aakashjapi.com/fuckin-search-engines-how-do-they-work/)

[Amazon product data](http://jmcauley.ucsd.edu/data/amazon/links.html)

[inverted index](https://en.wikipedia.org/wiki/Inverted_index)

[tf-idf and cosine similariry](https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/)

[Scoring, term weighting and the vector space model](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf)






