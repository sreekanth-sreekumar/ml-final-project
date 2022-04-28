import torch
import torch.nn as nn
import time
import os
import numpy as np
import copy

from trainer import ModelTrainer

class LateFusionTrainer(ModelTrainer):

    def __init__(self, dataloader: dict, model: nn.Module, device, model_name):
        super(LateFusionTrainer, self).__init__(dataloader, model, device, model_name)

    def train(self, optimizer, scheduler, criterion, num_epochs, report_len=500):
        
        since = time.time()
        best_wts = copy.deepcopy(self.model.state_dict())
        best_acc = float('-inf')

        patience = 8
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f'\nEpoch: {epoch+1}/{num_epochs}')

            self.model.train()
            torch.enable_grad()

            epoch_acc = []
            epoch_loss = []

            for i, data in enumerate(self.dataloader['train']):
                
                optimizer.zero_grad()
                
                data = {x: data[x].to(self.device) for x in data}
                
                inputs = {
                    'input_ids': data['input_ids'],
                    'attention_masks': data['attention_masks'],
                    'image': data['image']
                }

                labels = data['label']

                outputs = self.model(inputs)
                preds = outputs > 0.5

                acc = (preds.squeeze() == labels).float().sum() / len(labels)
                epoch_acc.append(acc.item())
                loss = criterion(outputs, labels.unsqueeze(-1).float())
                epoch_loss.append(loss.item())
                
                if i % report_len == 0:
                    self.report(i, np.mean(epoch_loss), np.mean(epoch_acc))
                
                loss.backward()
                optimizer.step()
            
            scheduler.step()    
            print(f'\nEpoch: {epoch+1}/{num_epochs} done, loss: {np.mean(epoch_loss)}, acc: {np.mean(epoch_acc)}')

            with torch.no_grad():
                self.model.eval()

                accuracies = []

                for _, data in enumerate(self.dataloader['val']):
                    data = {x: data[x].to(self.device) for x in data}
                    inputs = {
                        'input_ids': data['input_ids'],
                        'attention_masks': data['attention_masks'],
                        'image': data['image']
                    }

                    labels = data['label']

                    outputs = self.model(inputs)
                    preds = outputs > 0.5

                    acc = (preds.squeeze() == labels).float().sum() / len(labels)
                    accuracies.append(acc.item())

                val_acc = np.mean(accuracies)
                print(f'\nValidaton accuracies for epoch {epoch} is {val_acc}')
                print(f'\nBest Accuracy so far is {best_acc}')

                if val_acc > best_acc:
                    patience_counter = 0
                    best_acc = val_acc
                    best_wts = copy.deepcopy(self.model.state_dict())
                    self.save_model()

                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print('\nI ran out of patience')
                        break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        self.model.load_state_dict(best_wts)

    def test(self):

        if os.path.isfile(self.save_path):
            model_dict = torch.load(self.save_path)
            self.model.load_state_dict(model_dict)

        with torch.no_grad():
            self.model.eval()

            results = []
            
            for _, data in enumerate(self.dataloader['test']):
                data = {x: data[x].to(self.device) for x in data}
                inputs = {
                    'input_ids': data['input_ids'],
                    'attention_masks': data['attention_masks'],
                    'image': data['image']
                }

                labels = data['label']

                outputs = self.model(inputs)
                preds = outputs > 0.5

                preds = preds.squeeze().int().detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()

                acc, prec, recall, f1_score = self.get_testing_stats(labels, preds)

                results.append((acc, prec, recall, f1_score))

        test_acc = np.mean([item[0] for item in results])
        test_prec = np.mean([item[1] for item in results if item[1]] != None)
        test_recall = np.mean([item[2] for item in results if item[2] != None])
        test_f1_score = np.mean([item[3] for item in results if item[3] != None])
        print(f'\nTest Accuracy is {test_acc}')
        print(f'\nTest Precision is {test_prec}')
        print(f'\nTest Recall is {test_recall}')
        print(f'\nTest F1 score is {test_f1_score}')