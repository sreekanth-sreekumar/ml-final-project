import torch
import torch.nn as nn
import time
import os
import numpy as np
import copy

from trainer import ModelTrainer

class BertTrainer(ModelTrainer):

    def __init__(self, dataloader: dict, model: nn.Module, device, model_name):
        super(BertTrainer, self).__init__(dataloader, model, device, model_name)

    def get_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, optimizer, scheduler, num_epochs, report_len=500):
        
        since = time.time()
        best_wts = copy.deepcopy(self.model.state_dict())
        best_acc = float('-inf')

        patience = 8
        patience_counter = 0

        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            print(f'\nEpoch: {epoch+1}/{num_epochs}')

            self.model.train()
            torch.enable_grad()

            epoch_acc = []
            epoch_loss = []

            for i, data in enumerate(self.dataloader['train']):
                
                optimizer.zero_grad()
                
                data = {x: data[x].to(self.device) for x in data}
                input_ids = data['input_ids'].squeeze(dim=1)
                attn_masks = data['attention_masks'].squeeze(dim=1)
                labels = data['label']

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=None,
                    attention_mask=attn_masks,
                    labels=labels
                )

                loss = outputs.loss

                logits = outputs.logits.detach().cpu().numpy()
                labels = labels.cpu().numpy()

                acc = self.get_accuracy(logits, labels)
                epoch_acc.append(acc)

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
                    input_ids = data['input_ids'].squeeze(dim=1)
                    attn_masks = data['attention_masks'].squeeze(dim=1)
                    labels = data['label']

                    outputs = self.model(
                        input_ids=input_ids,
                        token_type_ids=None,
                        attention_mask=attn_masks,
                        labels=labels
                    )

                    logits = outputs.logits.detach().cpu().numpy()
                    labels = labels.cpu().numpy()

                    acc = self.get_accuracy(logits, labels)
                    accuracies.append(acc)

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
                
                input_ids = data['input_ids'].squeeze(dim=1)
                attn_masks = data['attention_masks'].squeeze(dim=1)
                labels = data['label']

                outputs = self.model(
                    input_ids=input_ids,
                    token_type_ids=None,
                    attention_mask=attn_masks
                )

                logits = outputs.logits

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()

                preds = np.argmax(logits, axis=1).flatten()
                labels = labels.flatten()

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
