+++

title = "GamerBuddy Search Engine"

+++
### **1 INTRODUCTION**
A video game search engines that takes query string from user and provides list of games that are related to the query.


### **2 FEATURES**
The Search Engine system suggests game based on the following attributes of users favorite games:

  1. Title.
  2. Description.
  3. Reviews of those games.
  
The datasets that is used by the search engine [here](http://jmcauley.ucsd.edu/data/amazon/links.html). Both video game meta data and reviews are used for the search engine. 

### **3 IMPLEMENTATION**
The data set contains set of json string. Each json string in meta data represents a product and a json in review file represents a review of a product in meta file.
To build a search engine, the following aspects need to be implemented.
  
  1. Processing your [dataset](https://github.com/Rao-Varun/varun_repo/tree/master/gamerbuddy/gamerbuddy_dataset). 
  1. Building a dictionary containing [inverse index](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py).
  2. Building [tf-idf](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_generator/input_generator.py) for all the terms in inverse index.
  3. Process [queries](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/process_query.py) and get products that contain the query terms.
  4. Generate [cosine rank](https://github.com/Rao-Varun/varun_repo/blob/master/gamerbuddy/input_processor/rank_generator.py) between query term and product document. 



### **DEPLOYING SEARCH ENGINE**
Search engine can deployed using flask and [ngok](https://ngrok.com/).
An implemented search engine is available [here](http://8f226999.ngrok.io)









