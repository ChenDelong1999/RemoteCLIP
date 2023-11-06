from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import os
import torch
import numpy as np
import open_clip
import json
from clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
from clip_benchmark.datasets.builder import get_dataset_collate_fn
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str,
        choices=['RN50', 'ViT-B-32', 'ViT-L-14'],
        help="Name of backbone. In open_clip.list_models() or hugging face transformers",
    )
    parser.add_argument(
        "--retrieval-json-dir",
        type=str,
        default=None,
        help="Path to retrieval json dataset",
    )
    parser.add_argument(
        "--retrieval-images-dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--remoteclip-path",
        default=None,
        type=str,
        help="Path to remoteclip weight",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers per GPU."
    )

    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(f'[Unknow args]: {unknown}')
    return args

def get_model(args):
    CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained='openai',
        device=args.device,
        cache_dir='cache/weights/open_clip'
    )
    tokenize = open_clip.tokenize
    checkpoint = torch.load(args.remoteclip_path, map_location="cuda")
    msg = CLIP_model.load_state_dict(checkpoint)
    print(msg)
    return CLIP_model, preprocess_train, preprocess_val, preprocess_val, tokenize

class JsonDataset(Dataset):
    def __init__(self, json_dir, img_dir, transforms):
        self.json_dir = json_dir
        self.transforms = transforms
        self.img_dir = img_dir
        self.images = []
        self.captions = []
        self.read_json()
        self.duplicate()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = Image.open(os.path.join(self.img_dir, self.images[idx]))
        images = self.transforms(images)
        texts = self.captions[idx]
        return images, texts

    def read_json(self):
        datasets = json.load(open(self.json_dir, "r"))
        for image in datasets['images']:
            if image['split'] == "test":
                for text in image['sentences']:
                    self.images.append(image['filename'])
                    self.captions.append(text['raw'].capitalize())

    def duplicate(self):
        unique_images, indexs = np.unique(self.images, return_index=True)
        if len(unique_images) != len(self.images):
            self.duplicated_images = []
            self.duplicated_captions = []
            for index in indexs:
                self.duplicated_images.append(self.images[index])
                same_indexs = [i for i, x in enumerate(self.images) if x == self.images[index]]
                captions = []
                for same_index in same_indexs:
                    captions.append(self.captions[same_index])
                self.duplicated_captions.append(captions)

            self.images = self.duplicated_images
            self.captions = self.duplicated_captions

# modified from clip_benchmark.metrics.zeroshot_retrieval
def retrieval_evaluation(args, model, preprocess, tokenize, recall_k_list=[1, 5, 10]):
    dataset = JsonDataset(
        args.retrieval_json_dir,
        args.retrieval_images_dir,
        preprocess
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=get_dataset_collate_fn('mscoco_captions')
    )
    n_batches = len(dataloader)

    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    for batch_images, batch_texts, inds in tqdm(dataloader, total=n_batches):
        batch_images = batch_images.to(args.device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        # tokenize all texts in the batch
        batch_texts = tokenize([text for i, texts in enumerate(batch_texts) for text in texts]).to(args.device)

        with torch.no_grad():
            batch_image_features = model.encode_image(batch_images)
            batch_text_features = model.encode_text(batch_texts)
            batch_images_emb = F.normalize(batch_image_features, dim=-1)
            batch_texts_emb = F.normalize(batch_text_features, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        '''
        Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        for each image, that number will be greater than 1 for text retrieval.
        However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        it over the dataset.
        '''
        metrics[f"retrieval-image2text-R@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size,
                                                                  args.device,
                                                                  k=recall_k) > 0).float().mean().item() * 100

    for recall_k in recall_k_list:
        metrics[f"retrieval-text2image-R@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size,
                                                                  args.device,
                                                                  k=recall_k) > 0).float().mean().item() * 100

    metrics[f"retrieval-mean-recall"] = np.mean(list(metrics.values()))

    for key, item in metrics.items():
        metrics[key] = round(float(item), 2)

    return metrics

if __name__ == "__main__":
    args = parse_args()
    args.device = "cuda"
    model, preprocess_train, preprocess_val, preprocess_aug, tokenize = get_model(args)

    # Image-text retrieval
    all_metrics = {}
    metrics = {}
    retrieval_metrics = retrieval_evaluation(args, model, preprocess_aug, tokenize)
    metrics.update(retrieval_metrics)
    all_metrics.update(retrieval_metrics)

    for name, val in metrics.items():
        print(name, round(val, 2))
