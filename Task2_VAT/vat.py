import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

def l2_norm(r):
    # Taking the norm of the current tensor on dim 1, after reshaping the tensor so all elements are in dim 1
    norm_= torch.norm(r.view(r.shape[0], -1, *(1 for _ in range(r.dim() - 2))), dim=1, keepdim=True) 
    norm_+= 1e-8 # For preventing nan
    r = r/norm_
    return r

def _entropy(logits):
    ## entropy formula: -1/n(sum(p*log_p))
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * log_p, dim=1))

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi # Perturbation parameter 
        self.eps = args.vat_eps # Final perturbation parameter
        self.vat_iter = args.vat_iter # Power iterations
        self.use_entmin = args.use_entmin # For considering entropy in the loss

    def forward(self, model, x):

        viz_adv_examples = [] # For storing viz examples

        sample_tensor = torch.randn(x.shape).to(x.device) # Sampling random Gaussian tensor
        r = l2_norm(sample_tensor) # Taking L2 norm

        with torch.no_grad():
            predictions = F.softmax(model(x), dim=1) # Getting the model predictions

        model.train()
        for k in range(self.vat_iter):
            r.requires_grad_()
            advExamples = x + self.xi * r # Adding perturbation
            advPredictions = F.log_softmax(model(advExamples), dim=1) # Predictions on perturbed data
            advDistance = F.kl_div(advPredictions, predictions, reduction='batchmean') # Calculating Divergence (log_p,p)
            advDistance.backward() # Passing the gradients back
            r = l2_norm(r.grad) # Taking gradient wrt to r
            model.zero_grad()

        # Final perturbation for loss
        r_adv = r * self.eps 
        viz_adv_examples.append([x,(x+r_adv)]) # Adding examples for visualization
        advPredictions = F.log_softmax(model(x + r_adv), dim=1)
        loss = F.kl_div(advPredictions, predictions, reduction='batchmean')

        # For entropy in the loss (if "--use_entmin" paramater as True)
        if self.use_entmin:
            loss += _entropy(model(x + r_adv))

        model.train()
        return loss
