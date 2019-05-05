+++

title = "GamerBuddy Search Engine"

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
   es in the collection to get the [cosine rank](https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/). The steps to implement cosine ranking system is described in implmentation of [search engine](https://varunsrao.netlify.com/project/gamerbuddy-search-engine/). Using that we will build a class that will perform cosine ranking system. We will build an object that genrates the cosine products.  


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
  
  
### c. Building [tf idf](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py) for all the terms in inverse index.


**Term frequency** of a term in a document is the ratio of number of times the term occur in that document to the total number of terms in the document. 

```python
          term_frequency(tf) of term t in document d  =  number of times t occurs in d / total number of terms in d
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








