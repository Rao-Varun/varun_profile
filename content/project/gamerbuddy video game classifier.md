+++

title = "GamerBuddy Video Game Classifier"

+++
## **1 INTRODUCTION**
Gamer Buddy Video Game Classfier extension to the [search engine](https://varunsrao.netlify.com/project/gamer-buddy-search-engine/) that we had built earlier.  Our video classifier classifies video games to its respective genres it belongs to based on its description.



## **2 FEATURES**

The classifier we are going to build here uses [Naive Bayes algorithm](https://towardsdatascience.com/introduction-to-naive-bayes-classification-4cffabb1ae54). 
The probability of a game depemds on 2 factors.
 
  1. Prior (probability of a game being in a specific genre from the given set of genres.)
  2. Likelihood (Likelihood is the conditional probability of a term occurring in a video game given that the videogame belongs to a particular genre.)
 
 The probability of a game belonging to a genre is given by the product of prior and likelihood. The product is found for all the genre. The genre with the largest product is then considered as the genre of the product.
 
There are not a lot of datasets that are available on the internet to begin with. Our dataset needs title, description and its genre. The datasets that is used for the classfier is same as the one we used for search engine and is available [here](http://jmcauley.ucsd.edu/data/amazon/links.html).
Video game meta data is used for building our classifier. But there's a problem here. This dataset does not contain genre. It only contains description and title of games. The solution to this problem is given in the later sections.
Before we begin implimenting it, make sure you turn the files in the string to list of jsons.  

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
  
###  b. Fixing our dataset
  
As I said before, our dataset doesn't have genre in it. We will fix this by getting another dataset from [Kaggle](https://www.kaggle.com/egrinstein/20-years-of-games#ign.csv)
We will find all the games that are common to both the datasets and get the genre from the one which we got from Kaggle.


```python   
      
      import csv
import json
import requests as r

product_genre = {}


def get_genre_for_products():
    file_json_ob = open("product_details.json", "r+")
    file_csv_ob = open("ign.csv", "r+")
    json_product = json.loads(file_json_ob.read())
    csv_products = csv.reader(file_csv_ob, delimiter=",")
    title = {}
    pg = open("product_genre.json", "r+")
    product_genre = json.loads(pg.read())
    count = 1
    csv_products.__next__()
    for csv_product in csv_products:
        for asin in json_product:
            if len(csv_product[2].split()) > 1 and (
                    csv_product[2] in str(json_product[asin].get("description")) or csv_product[2] in str(
                    json_product[asin].get("title"))):
                print("{} {} :: {} :: {}".format(str(count), csv_product[2], asin, csv_product[6]))
                if asin not in product_genre:
                    product_genre[asin] = [csv_product[6]]
                    count += 1
                if csv_product[6] not in product_genre[asin]:
                    product_genre[asin].append(csv_product[6])
                    count += 1
    open("product_genre_new.json", "w+").write(json.dumps(product_genre, indent=4, sort_keys=True))

if __name__ == "__main__":
    get_genre_for_products()
    
  ```
  We need to separate our dataset for training and test.
  
### c. Building the multipliers resource for the classifier. 

In this step we will build prior and likelihood so that it is readily available when we are trying to classify our document. 

1. Prior for a genre here is given by 
```python
Prior = (Number of video games that belongs to that genre) / (Total number of video games that are available in our dataset)
```

2. Likelihood here is given by 
```python
P(Word/genre) = (Number of occurrence of the term in all the videogame description from a genre+1) / (All the words in every videogame description from a category + Total number of unique words in all the videogame descriptions)
```
Notice that we add value to number of occurrence of the term in all the videogame description from a genre and 

To impliment this we we'll have to do the following steps.

#### 1. Categorize products to genre.
This step is to keep track of which product belongs to which genre.

#### 2. Categorize terms to genre.
Take all the terms in the descriptions of products of all genres and keep track of which genres they belong to.

#### 3. Build genre term index. 
This will keep a record how many times a term occurs in all the videogames of a given genre.

#### 4. Get likelihood of terms for a genre.

#### 5. Get prior for a genre.

#### 6. Get denominator value for each genre if the term doesn't exist
Denominator value is given by (All the words in every videogame description from a category + Total number of unique words in all the videogames descriptions)


Save the prior, likelihoods of terms , and denominator in different files as we need this later.


The code for building our prior, likelihood and denominator value is given below.

```python

import json
from itertools import product


class GamerBuddyClassifier(object):

    def __init__(self):
        self.product_genre = json.loads(open("product_genre_new.json", "r+").read())
        self.product_terms = json.loads(
            open("int_file.json", "r+").read()) #file containing dictionary of products and the list of terms that are in its descriptions and title.
        self.genre_product_details = {}
        self.genre_terms_details = {}
        self.genre_term_index = {}
        self.unique_terms = set()
        self.genre_denom_val = {}
        

    def categorise_terms(self):
        self._categorize_products_to_genres()
        # exit()
        self._categorize_terms_to_genres()
        self._build_genre_term_index()
        open("genre_term_index.json", "w+").write(json.dumps(self.genre_term_index, indent=4, sort_keys=True))
        self._get_prob_values()
        self._get_prior()

    def _categorize_products_to_genres(self):
        print("_categorize_products_to_genres")
        self.prior = {}
        for product in self.product_genre:
            for genre in self.product_genre[product]:
                if genre not in self.genre_product_details:
                    self.genre_product_details[genre] = [product]
                else:
                    self.genre_product_details[genre].append(product)
        for genre in self.genre_product_details:
            # if len(self.genre_product_details[genre])>100:
            print("{} :: {}".format(genre, len(self.genre_product_details[genre])))
        open("genre_products.json", "w+").write(json.dumps(self.genre_product_details, indent=4, sort_keys=True))

    def _categorize_terms_to_genres(self):
        print("_categorize_terms_to_genres")
        for genre in self.genre_product_details:
            if len(self.genre_product_details[genre]) > 100:
                self.genre_terms_details[genre] = []
                self.prior[genre] = len(self.genre_product_details[genre])
                for product in self.genre_product_details[genre]:
                    for key in self.product_terms[product]:
                        self.genre_terms_details[genre] = self.genre_terms_details[genre] + self.product_terms[product][
                            key]
                        self.unique_terms = self.unique_terms | set(self.product_terms[product][key])
        self.unique_terms_len = len(self.unique_terms)
        print(json.dumps(self.genre_terms_details, indent=4, sort_keys=True))

    def _build_genre_term_index(self):
        print("_build_genre_term_index")
        for genre in self.genre_terms_details:
            self.genre_term_index[genre] = {}
            for term in self.genre_terms_details[genre]:
                if term not in self.genre_term_index[genre]:
                    self.genre_term_index[genre][term] = 1
                else:
                    self.genre_term_index[genre][term] += 1
        print(json.dumps(self.genre_term_index, indent=4, sort_keys=True))

    def _get_prob_values(self):
        self.genre_term_prob_values = {}
        for genre in self.genre_term_index:
            terms_sum = sum(self.genre_term_index[genre].values())
            self.genre_denom_val[genre] = terms_sum + self.unique_terms_len
            self.genre_term_prob_values[genre] = {}
            for term in self.genre_term_index[genre]:
                print(" {} :: {}  :: {}".format(str(self.genre_term_index[genre][term] + 1), str(terms_sum),
                                                self.unique_terms_len))
                self.genre_term_prob_values[genre][term] = float(self.genre_term_index[genre][term] + 1) / float(
                    terms_sum + self.unique_terms_len)
        open("genre_probability.json", "w+").write(json.dumps(self.genre_term_prob_values, indent=4, sort_keys=True))
        open("genre_denom_values.json", "w+").write(json.dumps(self.genre_denom_val, indent=4, sort_keys=True))

    def _get_prior(self):
        all_docs = sum(self.prior.values())
        for genre in self.prior:
            self.prior[genre] = float(self.prior[genre]) / all_docs
        open("genre_prior.json", "w+").write(json.dumps(self.prior, indent=4, sort_keys=True))

```



### d. Build the classifier. 

This is the part where we finally build our classifier. 
To immplement the classfier we perform the following steps.
#### 1. split the descriptions into list of terms.
We have already implemented this while we were building our dataset for our [search engine](https://varunsrao.netlify.com/project/gamerbuddy-search-engine/).
The implimentation of this is given below.
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = list(set(stopwords.words("english")))
terms_in_file = {}

def generate_all_terms_in_metadata(metadata_dict):
      for product in metadata_dict:
          terms_in_file[product] = _get_terms_in_json(metadata_dict[product])
      open("int_file", "w+").write(json.dumps(terms_in_file)


def _get_terms_in_json(json_dict, key_list):
      json_to_terms = {}
      for key in json_dict:
          json_to_terms[key] = [term for term in word_tokenize(json_dict[key].lower(), language="english") if
                                term not in stop_words]
      return json_to_terms

```
#### 2. For each genre, find log of prior and likelihood of each term in the description and sum them.

We get the prior and and likelihood of all terms from the dictionaries we had built for each in step C. Finally, take log of each of  these values and get their sum.
We take the log of these values because the product of prior of genre and likelihood of terms is smaller than the minimum float value in python.
If the term doesn't exist in genre list then the likelihood of that term is the default value of that genre. i.e 1/ (All the words in every videogame description from a category + Total number of unique words in all the videogames descriptions)

(All the words in every videogame description from a category + Total number of unique words in all the videogames descriptions) is given by denom_value dictionary which we had built in step c

#### 3. Sort the genre based on the calculated sum in descending order.
The one with largest sum is the genre of the game. 


The code for implementating the classifier is given below
```python

  class GamerBuddyClassifier(object):

    def __init__(self):
        self.prior = json.loads(open("classifier_data/genre_prior.json", "r+").read())
        self.term_prob = json.loads(open("classifier_data/genre_probability.json", "r+").read())
        self.denom_val = json.loads(open("classifier_data/genre_denom_values.json", "r+").read())
        self.stop_words = list(set(stopwords.words("english")))

    def predict_product_genre(self, product_terms):
        print("info :: predicting genres of classifier")
        terms = product_terms
        self.prob = {}
        for genre in self.term_prob:
            self.prob[genre] = log(self.prior[genre])
            for term in terms:
                self.prob[genre] += log(self.term_prob[genre].get(term, float(1) / float(self.denom_val[genre])))
            self.prob[genre] = self.prob[genre]
        sorted_genres = sorted(self.prob.items(), reverse=True, key=lambda kv: (kv[1], kv[0]))
        print("info :: top 5 genre result {}".format(str(sorted_genres[:5])))
        return [sorted_genres[0][0], sorted_genres[1][0], sorted_genres[2][0]]


if __name__ == "__main__":
    GamerBuddyClassifier().predict_product_genre(input("Enter description\n"))

```



## **SOURCE CODE**
The entire source code for this can be found [here](https://github.com/Rao-Varun/gamerbuddy). 

## **REFERRENCE**


[Introduction to Naive Bayes Classification](https://towardsdatascience.com/introduction-to-naive-bayes-classification-4cffabb1ae54)


[DOCUMENT CLASSIFICATION USING MULTINOMIAL NAIVE BAYES CLASSIFIER](https://www.3pillarglobal.com/insights/document-classification-using-multinomial-naive-bayes-classifier)









