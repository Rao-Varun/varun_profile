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
  
### Processing your [dataset](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy/gamerbuddy_dataset). 
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
  The list of jsons in meta file will next be converted into { "asin1": { "description": "value", "title": "value", "imUrl": "value",...}, "asin2": { "description": "value", "title": "value", "imUrl": "value",...}....}
  
  The list of jsons in review file will be converted into {"asin1" : {{ "reviewerId" : "value", "reviewText": "value", ...},{ "reviewerId" : "value", "reviewText": "value", ...}, ...}, "asin1" : {{ "reviewerId" : "value", "reviewText": "value", ...},..}...}
  
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
  
###  Building a dictionary containing [inverted index](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py).
  
  So we use the datastructure [inverted index](https://en.wikipedia.org/wiki/Inverted_index). Inverted index for our document is of the following structure:
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
  
  Now to 
  
  
### Building [tf-idf](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py) for all the terms in inverse index.
### Process [queries](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/process_query.py) and get products that contain the query terms.
### Generate [cosine rank](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/rank_generator.py) between query term and product document. 



## **DEPLOYING SEARCH ENGINE**
Search engine can deployed using flask and [ngok](https://ngrok.com/).
An implemented search engine is available [here](http://8f226999.ngrok.io)









