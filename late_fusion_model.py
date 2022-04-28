import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    
    def __init__(self, resnet_model, bert_model):
        super(LateFusionModel, self).__init__()
        
        resnet_feature_size = resnet_model.fc.in_features
        self.resnet = resnet_model
        self.resnet.fc = nn.Identity()

        # Freeze resnet
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # bert_model.config.output_hidden_states = True
        self.bert = bert_model.bert
        bert_feature_size = bert_model.classifier.in_features
        
        self.bert.eval()
        
        # Freeze bert
        for param in self.bert.parameters():
            param.requires_grad = False

        # Create the last linear layer
        self.linear = nn.Linear(bert_feature_size + resnet_feature_size, 1)
        
    def forward(self, inp):

        inp = {x: inp[x].to(next(self.parameters()).device) for x in inp}
        
        bert_output = self.bert(inp['input_ids'].squeeze(dim=1), attention_mask=inp['attention_masks'].squeeze(dim=1))
        cls_vector = bert_output.pooler_output
        
        resnet_output = self.resnet(inp['image'])

        return self.linear(torch.cat((cls_vector, resnet_output), dim=1))
