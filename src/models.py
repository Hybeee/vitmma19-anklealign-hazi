import torch
import torch.nn as nn
import numpy as np

from config import Args

def get_model(args: Args, train_labels=None):

    if args.model_name.lower() == "dummy_baseline":
        if train_labels is None:
            args.logger.error(f"No labels were given for: {args.model_name}")
            return None
        
        return DummyBaseLine(train_labels=train_labels)
    
    else:
        args.logger.error(f"Unknoqn model name specified in arguments: {args.model_name}")
        return None

class DummyBaseLine(nn.Module):
    def __init__(self, train_labels):
        super(DummyBaseLine, self).__init__()

        self.classes = np.unique(train_labels)
        self.num_classes = len(self.classes)
        self.majority_class = self._calculate_majority_class(train_labels)

        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def _calculate_majority_class(self, train_labels):
        unique, counts = np.unique(train_labels, return_counts=True)

        majority_index = np.argmax(counts)
        return unique[majority_index]
    
    def forward(self, x):
        batch_size = x.size(0)

        logits = torch.full((batch_size, self.num_classes), -100.0, device=self.dummy_param.device)

        logits[:, self.majority_class] = 100.0

        return logits