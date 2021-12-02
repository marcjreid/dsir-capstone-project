# Project 3 - Subreddit Classification Study


## Problem Statement

Loan agreements are mostly uncategorized and having people review every contract manually limits productivity. Automating much of this process with a machine learning algorithm, companies can reduce both costs, and human error. I propose using Natural Language Processing to help review mortgage and loan contracts. It can sort them into categories depending on the language used in the contract. After it’s put into the appropriate category, the program can then pick out what parts of the contract needs to be reviewed by a person. 

---

## Executive Summary
Machine Learning and Natural Language Processing is a very powerful tool for making predictions with categorical data, using text. Finding subreddits with similar fandoms and terminology barely throws off the model. Most models are above 90% accuracy. The method is very good at pick out which words and word combinations are unique to a given document. Even when words that make classification easy are removed the algoirthm used can still find patterns in the texts and score high on accuracy. This makes them useful for sorting documents in a professional setting.


### Contents:
- [Data Collection](#Data-Collection) 
- [Data Cleaning and EDA](#Data-Cleaning-and-EDA)
- [Preprocessing and Modeling](#Preprocessing-and-Modeling)
- [Evaluation and Conceptual Understanding](#Evaluation-and-Conceptual-Understanding)
- [Conclusion and Reccomendations](#Conclusion-and-Reccomendations)

---

## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**selftext**|*object*|pulled with API|The main body of a post from reddit|
|**title**|*object*|pulled with API|The title of a reddit post.|
'https://api.pushshift.io/reddit/search/submission' - API used to scrape post data from reddit


---

# Conclusions and Recommendations

Banks and other financial institutions have a lot to gain by automating part of their contract review process. If you start internally developing machine learning software and tailor it to their needs it would be even more accurate than what’s commercially available. A person with expertise in the field still needs to work with these models as they are not perfect, but in the long run they can save money and eliminate some repetitive tasks.