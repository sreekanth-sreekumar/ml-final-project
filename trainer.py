import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix

class ModelTrainer:
    def __init__(self, dataloader: dict, model: nn.Module, device, model_name):
        self.dataloader = dataloader
        self.model = model
        self.save_path = f'./saved_models/{model_name}.pt'
        self.device = device
        self.model.to(device)

    def save_model(self):

        print(f"\nSaving model to {self.save_path}")
        torch.save(self.model.state_dict(), self.save_path)

    def report(self, iter, loss, acc):
        print(f'\nIteration: {iter}, Mean_loss: {loss}, Mean_acc: {acc}')

    def get_testing_stats(self, actual, preds):
        if not any([i for i in actual]) and not any([i for i in preds]):
            return (1.0, None, None, None)
        try:
            tn, fp, fn, tp = confusion_matrix(actual, preds).ravel()
            if (tp+fp):
                prec = tp/(tp+fp)
            else:
                prec = None
            if (tp+fn):
                recall = tp/(tp+fn)
            else:
                recall = None
            acc = (tp+tn)/(tp+tn+fp+fn)
            if prec and recall:
                f1_score = 2 * (prec * recall)/(prec + recall)
            else:
                f1_score = None
            return acc, prec, recall, f1_score
        except:
            print(actual, preds)