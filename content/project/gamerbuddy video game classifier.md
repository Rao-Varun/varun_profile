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

To impliment this we weill have to do the following steps.

#### 1. Categorize products to genre.
This step is to keep track of which product belongs to which genre.

#### 2. Categorize terms to genre.
Take all the terms in the descriptions of products of all genres and keep track of which genres they belong to.

#### 3. Build genre term index. 
This will keep a record how many times a term occurs in all the videogames of a given genre.

#### 4. Get likelihood of terms for a genre.

#### 5. Get prior for a genre.

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

Consider a document with terms [versatile, profit, leaf, loss, stock]. tf of profit in the document is 1/5 ie 0.2 .

The code for calculating tf is given below

```python
    
    term_tf = {}
    product_length = {}
    product_details = {dictionary containing all the product details obatained from json, which was calculated in the first step a }
    inverse_index = { dictionary containing all terms and its positions in documents. Obtained from step b}
    
    def generate_tf_for_terms_in_all_products():
        for term in inverse_index:
            term_tf[term] = dict()
            for product in inverse_index[term]:
                term_count = get_term_count_in_a_product(term, product)
                all_term_count = get_all_term_count_in_product(product)
                generate_tf_value_of_a_term_for_a_product(term, product, term_count, all_term_count)

    def _get_term_count_in_a_product(term, product):
        inv_ind_dict = inverse_index[term][product]
        term_count = sum([len(inv_ind_dict[key]) for key in inv_ind_dict])
        return term_count

    def _get_all_term_count_in_product(product):
        if product in product_length:
            return product_length[product]
        product_details = product_details[product]
        result = sum([len(product_details[key]) for key in product_details])
        product_length[product] = result
        return result

    def _generate_tf_value_of_a_term_for_a_product(term, product, term_count, all_term_count):
        term_tf[term][product] = float(term_count) / float(all_term_count)


```

**Inverse document frequency** of a term gives us the importance of a term in the entire document collection. Inverse document frequency is calculated in the following way

```python

          inverse docuemt frequency of term t = 1 + log(total number of documents in the collection / number of documents that have t)

```
The following code can be executed to obtained idf

```python

total_number_of_product = len(product_details) #product details, a dictionary that contains the details of all products.
term_idf = {}

def generate_idf_for_terms():
        total_len = len(inverse_index)
        for term in inverse_index:
            term_document_count = len(inverse_index[term].keys())
            term_idf[term] = 1 + log(total_number_of_product / float(term_document_count))

```


We calculate tf and idf for tf-idf. tf-idf of a term in a document is the product of Term Frequency(tf) of that term in that document and inverse document frequency(idf). We will discuss this later in generating cosine product.

```python

          tf-idf = tf * idf

```



### d. Process [queries](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/process_query.py) and get products that contain the query terms.

Processing queries basically means to find the document that contain all the terms which are present in the queries. 
So we first find list of documents for each term that contain it and then we will find all the documents that are common for all the terms.

The implementation of processing query for GamerBuddy includes following step:
  1. get all terms in in query sentence.
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


## **DEPLOYING SEARCH ENGINE**
Search engine can deployed using flask and [ngok](https://ngrok.com/).
An implemented search engine is available [here](http://55b4c05d.ngrok.io).


## **SOURCE CODE**
The entire source code for this can be found [here](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy). 

## **REFERRENCE**

[How to build a search engine](http://aakashjapi.com/fuckin-search-engines-how-do-they-work/)








