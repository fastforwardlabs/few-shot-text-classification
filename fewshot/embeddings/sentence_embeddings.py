from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from fewshot.utils import create_path

MODEL_NAME = "deepset/sentence_bert"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_transformer_model_and_tokenizer(model_name_or_path=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(DEVICE)
    return model, tokenizer


def batch_tokenize(text_list, tokenizer, max_length=384):
    """
    How is this different from tokenizer.encode_plus?
    """
    features = tokenizer.batch_encode_plus(
        text_list,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return features


def prepare_dataset(features):
    dataset = TensorDataset(features["input_ids"], features["attention_mask"])
    return dataset


def compute_embeddings(dataset, model, batch_size=16, **kwargs):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_embeddings = []
    for batch in tqdm(dataloader, desc="Computing sentence representations"):
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }
            outputs = model(**inputs)[0]
            embeddings = outputs.mean(dim=1).detach().cpu()

            try:
                all_embeddings = torch.cat((all_embeddings, embeddings), 0)
            except:
                all_embeddings = embeddings

            del outputs
            del embeddings

    return all_embeddings


def get_sentence_embeddings(data, model, tokenizer, output_filename=None, **kwargs):
    """
    data -> list: list of text
    """
    # TODO(#27): logging!

    features = batch_tokenize(data, tokenizer, **kwargs)
    dataset = prepare_dataset(features)
    embeddings = compute_embeddings(dataset, model, **kwargs)

    if output_filename:
        create_path(output_filename)
        torch.save(
            {"features": features, "dataset": dataset, "embeddings": embeddings},
            output_filename,
        )

    return embeddings
