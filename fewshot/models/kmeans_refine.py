import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fewshot.models.model_factory import RegisterModel
from fewshot.models.basic import Protonet
from fewshot.models.utils import *
import pdb

@RegisterModel("kmeans-refine")
class KMeansRefine(Protonet):

    def forward(self, sample, super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)#_alphabet

        xs = batch.x_train # (1, 5, 1, 28, 28)
        xq = batch.x_test # (1, 35, 1, 28, 28)

        nClusters = len(np.unique(batch.y_train.data.cpu().numpy()))
        h_train = self._run_forward(xs) # (1, 5, 1, 28, 28) -> (1, 5, 64)
        h_test = self._run_forward(xq) # (1, 5, 1, 28, 28) -> (1, 35, 64)

        prob_train = one_hot(batch.y_train, nClusters).to(const.device) # eye(5)

        protos = self._compute_protos(h_train, prob_train) # (1, 5, 64)

        bsize = h_train.size()[0]
        radii = Variable(torch.ones(bsize, nClusters)).to(const.device) * torch.exp(self.log_sigma_l)

        #deal with semi-supervised data
        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel) # (1, 50, 64)
            h_all = torch.cat([h_train, h_unlabel], dim=1) # (1, 55, 64)

            #perform some clustering steps
            for ii in range(self.config.num_cluster_steps):

                prob_unlabel = assign_cluster_radii(protos, h_unlabel, radii) # (1, 50, 5)
                prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).to(const.device)

                prob_all = torch.cat([prob_train, prob_unlabel_nograd], dim=1) # (1, 55, 5)
                protos = self._compute_protos(h_all, prob_all) # Sum to 1 in dimension -1

        logits = compute_logits_radii(protos, h_test, radii)
        labels = batch.y_test 

        _, y_pred = torch.max(logits, dim=2)
        loss = F.cross_entropy(logits.squeeze(), batch.y_test.squeeze())
        
        # acc_val = torch.eq(y_pred.squeeze(), batch.y_test.squeeze()).float().mean().data[0]
        acc_val = torch.eq(y_pred.squeeze(), batch.y_test.squeeze()).float().mean().item()

        return loss, {
            'loss': loss.item(),
            'acc': acc_val,
            'logits': logits.detach().cpu().numpy()
            }
