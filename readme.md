# Analysis of Reuters Collection Volume 1

Reuters Collection Volume I is a corpus of around 800,000 documents gathered in
1997-1998 by Reuters.  The articles are labeled with 1 or more thematic category.  
The dataset is accessible through scikit-learn.  It contains a tfidf of the terms 
(47K terms) in sparse matrix format, and a separate sparse matrix indicating 
whether a given document is assigned one of 103 categories.

The reuters_classification Jupyter notebook does exploratory analysis, builds
logistic regression models for the top 5 most prevalent categories, tests on
an out-of-sample set and outputs metrics.  

main.py is a script to build models using keras.  