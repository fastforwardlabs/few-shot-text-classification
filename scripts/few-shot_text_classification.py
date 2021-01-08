# Load some data and perform few-shot text classification using the
# Latent Embeddings approach
import os
import torch

from tqdm.notebook import tqdm, tnrange

from fewshot.data.loaders import (
    load_or_cache_data,
    _load_agnews_dataset,
    _create_dataset_from_df
)

from fewshot.data.utils import (
    select_subsample, 
    expand_labels
)

from fewshot.eval import predict_and_score

from fewshot.utils import (
    fewshot_filename, 
    torch_load,
    torch_save,
    pickle_load
)

from fewshot.models.few_shot import (
    FewShotLinearRegression, 
    BayesianMSELoss,
    prepare_dataloader,
    train
)   


DATASET_NAME = "agnews"
DATADIR = f"data/{DATASET_NAME}"

## Load Training Data

# This loads all 120k examples in the AG News training set
df_news_train = _load_agnews_dataset(split="train")

# We want to explore learning from a limited number of training samples so we select 
# a subsample containing just 400 examples (100 from each of the 4 categories).
df_news_train_subset = select_subsample(df_news_train, sample_size=100)

# convert that DataFrame to a Dataset
ds_filename = f"{DATADIR}/{DATASET_NAME}_train_dataset.pkl"
if os.path.exists(ds_filename):
    news_train_subset = pickle_load(ds_filename)
else:
    news_train_subset = _create_dataset_from_df(df_news_train_subset, 
                                                text_column='text', 
                                                filename=ds_filename)
# this is required due the particular implementation details of our Dataset class
news_train_subset = expand_labels(news_train_subset)

## Load Zmap
# We'll proceed under the assumption that the Zmap we learned during on-the-fly 
# classification provides the best representations for our text and labels.  
Zmap = torch.load(fewshot_filename("data/maps/Zmap_20000_words.pt"))

## Prepare a Torch DataLoader for training
# convert the properly formatted training Dataset into a PyTorch DataLoader
# this function abstracts away the torch Tensor manipulation required to get 
# our text representations into the proper format for training 
data_loader = prepare_dataloader(news_train_subset, Zmap)

# instantiate the model
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
learning_rate = 0.1
lambda_regularization = 500
num_epochs = 1000

fewshot_model = FewShotLinearRegression(Zmap.size()[1], Zmap.size()[1], 
                                        loss_fcn=BayesianMSELoss(device=device), 
                                        lr=learning_rate, 
                                        device=device)
# train!
loss_history = train(fewshot_model, data_loader, num_epochs=num_epochs, lam=lambda_regularization)

# after training we can extract Wmap (the weights of the linear model)
Wmap = fewshot_model.linear.weight.detach().cpu()

## Test 
# Wmap learns to associate training examples to their associated labels 
# We can now apply Wmap to the test set 

# load the test set
test_dataset = load_or_cache_data(DATADIR, DATASET_NAME)

score = predict_and_score(test_dataset, linear_maps=[Zmap, Wmap], return_predictions=False)
print(score)

## Success! 
# Let's save this Wmap
torch_save(Wmap, fewshot_filename(f"data/maps/Wmap_{DATASET_NAME}.pt"))