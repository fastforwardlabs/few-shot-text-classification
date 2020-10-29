# Load some data and perform few-shot text classification using the
# Latent Embeddings approach

import torch

from tqdm.notebook import tqdm, tnrange
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

from fewshot.predictions import compute_predictions, compute_predictions_projection
from fewshot.utils import fewshot_filename, pickle_load, to_tensor, to_list, compute_projection_matrix
from fewshot.metrics import simple_accuracy, simple_topk_accuracy
from fewshot.data.loaders import Dataset, _load_agnews_dataset, _prepare_category_names, load_or_cache_data

## Load Data
newsdf = _load_agnews_dataset(split='train')

## Get a subsample
# the AG news dataset has 120k examples but we want to model what it would be like
# to have only a few labeled examples!
news_sample = newsdf.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 100), random_state=42)
)

## Create Dataset
# TODO: eventually this should be tucked away in fewshot.data.loaders
train_dataset = Dataset(
    examples=news_sample.text.to_list(),
    labels=news_sample.label.tolist(),
    categories=_prepare_category_names(news_sample)
)

# This is true for the ag news dataset only
num_categories = 4
examples_emb = train_dataset.embeddings[:-num_categories]
# This contains embeddings for the 4 labels associated with the ag news dataset
label_emb_options = train_dataset.embeddings[-num_categories:]

# However, when we're training a mapping, we need one label embedding _per example_
labels_emb = []
for label in train_dataset.labels:
    label_emb_options = to_list(train_dataset.embeddings[-num_categories:])
    labels_emb.append(label_emb_options[label])
labels_emb = to_tensor(labels_emb)

### Load the Projection embedding we learned in the Zero-Shot script
# this embedding helps us map long sentences to short words 
# TODO: get the right directory for loading the Z projection matrix
Z = pickle_load("/content/projmatrix_for_top10000_w2v_words.pkl")

# Our baseline is our ZeroShot approach -- using the SBERT embeddings and 
# the dimensionality reduction mapping that we learned before. We know this
# approach works well so this will be our PRIOR 
X_train = torch.mm(examples_emb, Z)
Y_train = torch.mm(labels_emb, Z)

### Compute a second Projection Matrix to take advantage of some labeled data
# In few-shot learning we may have _some_ labeled data that we want to learn from
# in this method, we capitalize on this data by learning a projection between the
# text embeddings and the ground truth label embeddings for our few labeled examples
# This embedding will then be used to further enhance our predictions for all other exmaples!
loss_history = model.train(train_dataloader, num_epochs, 20)

# Get our mapping from the model (weights of the Linear regression)
W = model.linear.weight.detach()

## Test 
# we learned a mapping between text examples and their associated labels 
# Time to apply this mapping to our test set

# load the test set
test_dataset = load_or_cache_data(".", "agnews")

test_examples_emb = test_dataset.embeddings[:-num_categories]
test_labels_emb = test_dataset.embeddings[-num_categories:]

# First apply our Z(ero-shot) mapping because that's our PRIOR
X_test = torch.mm(test_examples_emb, Z)
Y_test = torch.mm(test_labels_emb, Z)

# Now we'll compute predictions using the 
predictions = compute_predictions_projection(X_test, Y_test, W)

### Score the results!
score = simple_accuracy(test_dataset.labels, predictions)
print(score)
### Visualize our data and labels
