import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import models
from resnet_trainer import ResnetTrainer
from dataset import FakeRedditImageDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir = './data/public_image_set'
dataset_types = ['train', 'test', 'val']
file_map = {
    'train': './data/multimodal_train.tsv',
    'test': './data/multimodal_test_public.tsv',
    'val': './data/multimodal_validate.tsv'
}

fake_reddit_datasets = {x: FakeRedditImageDataset(file_map[x], img_dir) for x in dataset_types}
fake_reddit_dataloaders = {x: DataLoader(fake_reddit_datasets[x], batch_size=16, shuffle=True, num_workers=2, collate_fn=FakeRedditImageDataset.get_collate_fn()) for x in dataset_types}

# Output a pretrained resnet model
def resnet50_2way(pretrained: bool):
    model = models.resnet50(pretrained)
    num_feats = model.fc.in_features

    # Set the number of output layers as 1 through a Linear layer
    model.fc = nn.Linear(num_feats, 1)
    return model

model = resnet50_2way(pretrained=True)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

resnet_trainer = ResnetTrainer(fake_reddit_dataloaders, model, device, 'resnet50')
# resnet_trainer.train(optimizer, lr_scheduler, criterion, num_epochs=20)

resnet_trainer.test()