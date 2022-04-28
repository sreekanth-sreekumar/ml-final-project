import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from transformers import BertTokenizer
from PIL import Image

import pandas as pd
import numpy as np
import os

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class FakeRedditDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, delimiter='\t')
    
    def __len__(self):
        return len(self.df)

    def get_collate_fn():
        # Collate fn which returns the batch of data
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return default_collate(batch)

        return collate_fn
        

class FakeRedditCommentDataset(FakeRedditDataset):
    def __init__(self, csv_file):

        super(FakeRedditCommentDataset, self).__init__(csv_file)
        sentences = self.df['clean_title'].values
        max_len_bert = 0

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = torch.tolist()
        
        try:
            sentence = self.df.loc[idx, 'clean_title']
            bert_encoded_dict = self.bert_tokenizer.encode_plus(
                sentence,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=120,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            bert_input_id = bert_encoded_dict['input_ids']
            bert_attention_mask = bert_encoded_dict['attention_mask']
            label = self.df.loc[idx, '2_way_label']
            return {
                'input_ids': bert_input_id,
                'attention_masks': bert_attention_mask,
                'label': label
            }
        
        except:
            return None

class FakeRedditImageDataset(FakeRedditDataset):

    def __init__(self, csv_file, img_dir):
        super(FakeRedditImageDataset, self).__init__(csv_file)
        self.img_dir = img_dir

        # Torchvision transformer
        self.img_transform = data_transforms

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            # Get image and if needed, convert to RGB
            img_name = self.df.loc[idx, 'id'] + '.jpg'
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Target label
            label = self.df.loc[idx, '2_way_label']
            # Transform image to tensors after resize
            if self.img_transform:
                image = self.img_transform(image)

            return {
                'image': image,
                'label': label
            }
        
        except:
            return None

class FakeRedditHybridDataset(FakeRedditDataset):

    def __init__(self, csv_file, img_dir):

        super(FakeRedditHybridDataset, self).__init__(csv_file)
        self.img_dir = img_dir

        # Torchvision transformer
        self.img_transform = data_transforms

        # Bert Tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            # Get sentence and tokenize text using Bert tokenizer
            sentence = self.df.loc[idx, 'clean_title']
            bert_encoded_dict = self.bert_tokenizer.encode_plus(
                sentence,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=120,  # Pad & truncate all sentences.
                padding='max_length',
                truncation=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            
            # Get bert input ids and attention masks
            bert_input_id = bert_encoded_dict['input_ids']
            bert_attention_mask = bert_encoded_dict['attention_mask']

            # Get image and if needed, convert to RGB
            img_name = self.df.loc[idx, 'id'] + '.jpg'
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Target label
            label = self.df.loc[idx, '2_way_label']
            # Transform image to tensors after resize
            if self.img_transform:
                image = self.img_transform(image)

            return {
                'input_ids': bert_input_id,
                'attention_masks': bert_attention_mask,
                'image': image,
                'label': label
            }
        
        except:
            return None



