# Load some data and perform few-shot text classification using the
# Latent Embeddings approach
#


## Load Data

### Identify column(s) of interest


### Compute embeddings for the product descriptions and product categories


### Load the Projection embedding we learned in the Zero-Shot script
# this embedding helps us map long sentences to short words so we'll use it all the time


### Compute a second Projection Matrix to take advantage of some labeled data
# In few-shot learning we may have _some_ labeled data that we want to learn from
# in this method, we capitalize on this data by learning a projection between the
# text embeddings and the ground truth label embeddings for our few labeled examples
# This embedding will then be used to further enhance our predictions for all other exmaples!


### Compute predictions based on cosine similarity


### Score the results!



### Visualize our data and labels


