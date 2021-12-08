# Project  - Pokemon Image Classification 


## Problem Statement
The goal of my project is to find a way for card collectors to link their collection
with their online accounts or just to store their cards electronically. Typing in Pokemonnames on 
a phone or table can be cumbersome and the ability to scan an image of a card to find out
how much it is worth would be very useful to players and collectors. This is also a good 
opportunity to learn how Nueral Networks best learn patterns to classify images.


---

## Data Source
'https://api.pushshift.io/reddit/search/submission' - API used to scrape post data from reddit
'https://www.kaggle.com/lantian773030/pokemonclassification' - Kaggle dataset with centered Pokemon images.

---

# Conclusions and Recommendations
50% accuracy shows the model is detecting a signal in the card images for classification.
It is difficult to have high accuracy when there are a lot of classes, it is a good idea to use early stopping and drop out layers to prevent overfitting in model.
If I ever turn this into an app I would like to generate fake card wear and tear and analyze card for a predicted sale price.

