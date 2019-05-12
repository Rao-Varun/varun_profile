+++

title = "GamerBuddy Search Engine"

+++
## **1 INTRODUCTION**
So we are gonna build a video game search engines that takes query string from user and provides a list of games that are related to the query. I named mine as GamerBuddy.


## **2 FEATURES**

The Search Engine system gives a list of games based on the following attributes of users favorite games:

  1. Title.
  2. Description.
  
So that means your datasets must contain the above mentioned attributes and the queries are matched with the strings present in the above mentioned attributes of the dataset. The datasets that is used by the search engine is available [here](http://jmcauley.ucsd.edu/data/amazon/links.html). Video game meta data is used for building our search engine, GamerBuddy. Make sure you turn the files in the string to list of jsons.  

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
  
###  b. Building a dictionary containing [inverted index](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py).
  
The basic idea of search engine is to see which document consists all the terms that are present in our query. We use the datastructure called [inverted index](https://en.wikipedia.org/wiki/Inverted_index). This helps us find the documents and the positions of a term in which it occurs. We make sure we avoid [stopwords](https://en.wikipedia.org/wiki/Stop_words) in our collection of terms. This decreases the size of our inverted index. You can further decrease your inverted index size by using [lemmetization and stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html), which I have not used in my search engine. The inverted index for our document is of the following structure:

```python   
      
      { 
        "word1" : { "asin_value1" :{"description": [ 10, 40, 50, ...], "title": [15, 40, 52.....]},
                    "asin_value1" :{"description": [ 10, 40, 50, ...], "title": [15, 40, 52.....]}, 
                  .
                  .
                  .},
        "word2": {..................},
        
          .
          .
          .
        
      }
      
  ```       
  But before that we first build a list of terms for each product. That is
  ```python
      
      {
      "asin_value": { 
                      "description": ["word1", "word2", .......],
                      "title": ["word1", "word2", .......]
                    },
      "asin_value": { 
                      "description": ["word1", "word2", .......],
                      "title": ["word1", "word2", .......]
                    },
          .
          .
          .
      }
      
  ```
  To build the a list of terms for each product("asin") we execute the following code.
  
  ```python
  
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize
  
  stop_words = list(set(stopwords.words("english")))
  terms_in_file = {}
  
  def generate_all_terms_in_metadata(metadata_dict):
        for product in metadata_dict:
            terms_in_file[product] = _get_terms_in_json(metadata_dict[product])
  
  
  def _get_terms_in_json(json_dict, key_list):
        json_to_terms = {}
        for key in json_dict:
            json_to_terms[key] = [term for term in word_tokenize(json_dict[key].lower(), language="english") if
                                  term not in stop_words]
        return json_to_terms
        
  ```
  
  Now to convert a list a terms in documents to inverted index we execute the following code.
  
  ```python
   
   inverted_index = {}
   
   def generate_position_of_terms_in_input_json():
        for count, asin in enumerate(terms_in_file, start=1):
                product_term_collections[asin] = _find_all_term_position_for_a_product(terms_in_file[asin])
                _update_term_positions_dictionary(asin, product_term_collections)
                
   def  _find_all_term_position_for_a_product(product_term_collection):
        term_position = {}
        for key in product_term_collection:
            for index, term in enumerate(product_term_collection[key]):
                if not term in term_position:
                    term_position[term] = {}
                if not key in term_position[term]:
                    term_position[term][key] = []
                term_position[term][key].append(index)
        return term_position 
   
   def _update_term_positions_dictionary(asin, term_collection):
        for term in term_collection:
            if not term in term_position:
                inverted_index[term] = {}
            if not asin in self.term_position[term]:
                inverted_index[term][asin] = {}
            for key in term_collection[term]:
                inverted_index[term][asin][key] = term_collection[term][key]

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






