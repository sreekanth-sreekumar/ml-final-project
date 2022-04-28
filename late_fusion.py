import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import models

from late_fusion_trainer import LateFusionTrainer
from dataset import FakeRedditHybridDataset

from transformers import BertForSequenceClassification
from late_fusion_model import LateFusionModel

# Output a pretrained resnet model
def resnet50_2way(pretrained: bool):
    model = models.resnet50(pretrained)
    num_feats = model.fc.in_features

    # Set the number of output layers as 1 through a Linear layer
    model.fc = nn.Linear(num_feats, 1)
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = './data/public_image_set'
dataset_types = ['train', 'test', 'val']
file_map = {
    'train': './data/multimodal_train.tsv',
    'test': './data/multimodal_test_public.tsv',
    'val': './data/multimodal_validate.tsv'
}

fake_reddit_datasets = {x: FakeRedditHybridDataset(file_map[x], img_dir) for x in dataset_types}
fake_reddit_dataloaders = {x: DataLoader(fake_reddit_datasets[x], batch_size=16, shuffle=True, num_workers=2, collate_fn=FakeRedditHybridDataset.get_collate_fn()) for x in dataset_types}


resnet_model = resnet50_2way(pretrained=False)
resnet_dict = torch.load('./saved_models/resnet50.pt')
resnet_model.load_state_dict(resnet_dict)

bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased')
bert_dict = torch.load('./saved_models/bert-classifier.pt')
bert_classifier.load_state_dict(bert_dict)

model = LateFusionModel(resnet_model, bert_classifier)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

hybrid_trainer = LateFusionTrainer(fake_reddit_dataloaders, model, device, 'late-fusion-model')
# hybrid_trainer.train(optimizer, lr_scheduler, criterion, 2)

hybrid_trainer.test()