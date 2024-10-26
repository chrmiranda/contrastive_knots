#Contrastive Knots

Exploring braids through supervised learning.


## Dataset Generation
We generate augmented data based on the ids and braid notations of prime knots from 3 to 13 crossing number.
Each braid is being transformed by performing a sequence of Markov moves on the braid.

## 1. Directly Predicting Crossing Number from Braid Notation
We use a simple neural network to predict crossing number from braid notation.
**Goals**: 
1. Find ways to improve the performance of a simple model on the task of directly predicting crossing number. 
2. Examine the relationship between model performance and number of Markov moves applied.