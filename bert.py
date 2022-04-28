import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from bert_trainer import BertTrainer
from dataset import FakeRedditCommentDataset

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_types = ['train', 'test', 'val']
file_map = {
    'train': './data/multimodal_train.tsv',
    'test': './data/multimodal_test_public.tsv',
    'val': './data/multimodal_validate.tsv'
}

fake_reddit_datasets = {x: FakeRedditCommentDataset(file_map[x]) for x in dataset_types}
fake_reddit_dataloaders = {x: DataLoader(fake_reddit_datasets[x], batch_size=16, shuffle=True, num_workers=2, collate_fn=FakeRedditCommentDataset.get_collate_fn()) for x in dataset_types}

# BERT
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification. 
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False # Whether the model returns all hidden-states.
)

model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr = 5e-5, # args.learning_rate - default is 5e-5
    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
)

# Total steps is [train_batch_size] * num_epochs
total_steps = len(fake_reddit_dataloaders['train']) * 2

# Create the learning rate scheduler.
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps = 0, # Default value in run_glue.py
    num_training_steps = total_steps
)

bert_trainer = BertTrainer(fake_reddit_dataloaders, model, device, 'bert-classifier')
# bert_trainer.train(optimizer, lr_scheduler, num_epochs=2)

bert_trainer.test()
