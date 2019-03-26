+++

title = "GamerBuddy Search Engine"

+++
## **1 INTRODUCTION**
So we are gonna build a video game search engines that takes query string from user and provides a list of games that are related to the query. I named mine as GamerBuddy.


## **2 FEATURES**

The Search Engine system gives a list of games based on the following attributes of users favorite games:

  1. Title.
  2. Description.
  3. Reviews of those games.
  
So that means your datasets must contain the above mentioned attributes and the queries are matched with the strings present in the above mentioned attributes of the dataset. The datasets that is used by the search engine is available [here](http://jmcauley.ucsd.edu/data/amazon/links.html). Both video game meta data and video game reviews are used for building our search engine GamerBuddy. Make sure you turn the files in the string to list of jsons.  

## **3 IMPLEMENTATION**

The data set contains set of json string. Each json string in meta data represents a product and a json in review file represents a review of a product in meta file.
To build a search engine, the following aspects need to be implemented.
  
### a. Processing your [dataset](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy/gamerbuddy_dataset). 
     Turn the .zip files into list of json string using following code. Each json string contains details of a product(meta file) or its review(review file).
  ```python
  
  # generate list of json strings.
  #op meta file=> [ {"asin" : "value", "title": "value" , "description": "value", "imUrl": "value"........ }, .....]
  #op review file=>[ {"reviewText: "value", "reviewerID": "value", "asin": "value" ...}....]
  
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
  The list of jsons in review file will be converted into
  ```python
  # product_reviews
  
  {
  "asin1" : {
            { "reviewerId" : "value", "reviewText": "value", ...},
            { "reviewerId" : "value", "reviewText": "value", ...},
              .
              .
              .},
  "asin2" : {
            { "reviewerId" : "value", "reviewText": "value", ...},
            .
            .
            },
      .
      .
      .
   }
            
  ```
  The format change can be performed by executing the following code.
  
  ```python
  
  def split_file(file_name):
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

  def split_file_review(file_name):
    #for review file
    review_dict = {}
    input_file = open(file_name + ".json", "r+")
    output_file = open("product_reviews.json", "w+")
    meta_file = open("product_details.json", "r+")
    input_json = json.loads(input_file.read())
    meta_json = json.loads(meta_file.read())
    products = meta_json.keys()
    for review in input_json:
        if review["asin"] in products:
            if review["asin"] not in review_dict:
                review_dict[review["asin"]] = [review]
            else:
                review_dict[review["asin"]].append(review)
        del (review["asin"])
    output_file.write(json.dumps(review_dict))
    output_file.close()
  
  ```
  
###  b. Building a dictionary containing [inverted index](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py).
  
The basic idea of search engine is to see which document consists all the terms that are present in our query. We use the datastructure called [inverted index](https://en.wikipedia.org/wiki/Inverted_index). This helps us find the documents and the positions of a term in which it occurs. We make sure we avoid [stopwords](https://en.wikipedia.org/wiki/Stop_words) in our collection of terms. This decreases the size of our inverted index. You can further decrease your inverted index size by using [lemmetization and stemming](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html), which I have not used in my search engine. The inverted index for our document is of the following structure:

```python   
      
      { 
        "word1" : { "asin_value1" :{"description": [ 10, 40, 50, ...], "title": [15, 40, 52.....], "reviewerId1_value": [13, 78, 42],                                       ....},
                    "asin_value1" :{"description": [ 10, 40, 50, ...], "title": [15, 40, 52.....], "reviewerId34_value": [13, 78,                                           42]....} "
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
      "asin_value": { "description": ["word1", "word2", .......], "title": ["word1", "word2", .......], "reviewid_value": ["word1", "word2", .......]},
      "asin_value": { "description": ["word1", "word2", .......], "title": ["word1", "word2", .......], "reviewid_value": ["word1", "word2", .......]},
      .
      .
      .
      }
      
  ```
  To build the generate a list for each product("asin") we execute the following code.
  
  ```python
  
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize
  
  stop_words = list(set(stopwords.words("english")))
  terms_in_file = {}
  
  def generate_all_terms_in_metadata(metadata_dict):
        for product in metadata_dict:
            terms_in_file[product] = get_terms_in_json(metadata_dict[product], ["title", "description"])
  
  def generate_all_terms_in_review_data(self, review_dict):
        for asin in review_dict:
            for review in review_dict[asin]:
                terms_in_file[asin][review["reviewerID"]] = get_terms_in_json(review, ["reviewText"])[
                    "reviewText"]
  
  
  def get_terms_in_json(json_dict, key_list):
        json_to_terms = {}
        for key in key_list:
            if key not in json_dict:
                continue
            json_to_terms[key] = [term for term in word_tokenize(json_dict[key].lower(), language="english") if
                                  term not in stop_words]
        return json_to_terms
        
  ```
  
  Now to convert a list a terms in documents we execute the following code.
  
  ```python
   
   inverted_index = {}
   
   def generate_position_of_terms_in_input_json():
        for count, asin in enumerate(terms_in_file, start=1):
                product_term_collections[asin] = find_all_term_position_for_a_product(terms_in_file[asin])
                update_term_positions_dictionary(asin, product_term_collections)
                
   def  find_all_term_position_for_a_product(product_term_collection):
        term_position = {}
        for key in product_term_collection:
            for index, term in enumerate(product_term_collection[key]):
                if not term in term_position:
                    term_position[term] = {}
                if not key in term_position[term]:
                    term_position[term][key] = []
                term_position[term][key].append(index)
        return term_position 
   
   def update_term_positions_dictionary(asin, term_collection):
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

    def get_term_count_in_a_product(term, product):
        inv_ind_dict = inverse_index[term][product]
        term_count = sum([len(inv_ind_dict[key]) for key in inv_ind_dict])
        return term_count

    def get_all_term_count_in_product(product):
        if product in product_length:
            return product_length[product]
        product_details = product_details[product]
        result = sum([len(product_details[key]) for key in product_details])
        product_length[product] = result
        return result

    def generate_tf_value_of_a_term_for_a_product(term, product, term_count, all_term_count):
        term_tf[term][product] = float(term_count) / float(all_term_count)


```

**Inverse document frequency** of a term is the importance of a term in the entire document collection. Inverse document frequency is calculated in the following way

```python

          inverse docuemt frequency of term t = 1 + log(total number of documents in the collection / number of documents that have t)

```
The following code can be executed to obtained idf

```python

term_idf = {}

def generate_idf_for_terms():
        total_len = len(inverse_index)
        for term in inverse_index:
            term_document_count = len(inverse_index[term].keys())
            term_idf[term] = 1 + log(self.total_number_of_product / float(term_document_count))

```


We calculate tf and idf for tf-idf. tf-idf is the product of Term Frequency(tf) and inverse document frequency(idf). We will discuss this later in generating cosine product.

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
  4. filter products second time based on which key[desciption, title, reviews] the term exits. If a product contains all the query terms then those terms should exist in the same key. 
  
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
    return get_asin_containing_query(common_asin_list)

  #get products that contain the terms in its description, title, or reviews. Each term has its own 
  def get_asin_from_inverted_index(self):
    term_asin_list = {}
    for term in query_list:
        term_asin_list[term] = inverted_index.get(term)
    return term_asin_list


  #filter products second time based on which key[desciption, title, reviews] the term exits. 
  def get_common_asin_products(self, term_asin_list):
        print("info :: getting products(asin) containing all term words...")
        common_asin = set()
        for term in term_asin_list:
            if len(common_asin) == 0:
                common_asin = set(term_asin_list[term].keys())
            else:
                common_asin = common_asin & set(term_asin_list[term].keys())
        return list(common_asin)

  

```




### e. Generate [cosine rank](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/rank_generator.py) between query term and product document. 

Consine Rank is used to rank the search result that we obtained in step d. We use [vector space model ](https://nlp.stanford.edu/IR-book/pdf/06vect.pdf). Each term in the document collections represents an axis in that vector space model and a document is represented as a vector. Each vectors have their terms as its components and the value of each component  is given by that terms tf-idf. Similarly queries are represented as vectors. The tf of a term in query sentence is given by the number of times that term occur in the sentence and for idf, we refer the same idf table we created for our document collects(product list)

Cosine Ranking is computed in the following way. 

  1. Normalise document vectors
  2. Normalise query vectors
  3. Perform dot product of both vectors(cosine value).
  4. Sort the vectors in descending order. The higher the product value, the more relevant the document is to the query. 

##### 1. Code for normalised documents vectors is given below

~~~python
    from math import sqrt
    
    #product_list = search result occured in step d.
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

##### 3. Code for generating dot product of documents and query

~~~python

    def generate_cosine_value(product_tf_idf, query_tf_idf):
        product_rank = dict()
        for product in product_tf_idf:
            temp_product = product_tf_idf[product]
            for term in query_tf_idf:
                product_rank[product] = temp_product[term] * query_tf_idf[term]
        return product_rank

~~~


## **DEPLOYING SEARCH ENGINE**
Search engine can deployed using flask and [ngok](https://ngrok.com/).
An implemented search engine is available [here](http://55b4c05d.ngrok.io).


## **SOURCE CODE**
The entire source code for this can be found [here](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy). 

##**REFERRENCE**

[How to build a search engine](http://aakashjapi.com/fuckin-search-engines-how-do-they-work/)








