import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fewshot.models.model_factory import RegisterModel
from fewshot.models.basic import *
from fewshot.models.utils import *
import fewshot.utils.data_utils
from fewshot.data.episode import Episode
import pdb


@RegisterModel("crp")
class CRPModel(Protonet):

    def __init__(self,
            config,
            data,
            dataset=None):
        super(CRPModel, self).__init__(config, data)

        self.base_distribution = Variable(0.0 * torch.randn(1, 1, config.dim)).to(const.device)

    def _add_cluster(self,
            nClusters,
            protos,
            counts,
            radii,
            cluster_type='unlabeled',
            ex=None,
            use_mean=False,
            nc=1.):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            counts: [B, nClusters] number of examples in each cluster
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0] # 1
        dimension = protos.size()[2] # 64

        zero_count = Variable(torch.zeros(bsize, 1)).to(const.device)
        if ex is None:
            counts = torch.cat([counts, zero_count], dim=1) # Count # examples for each cluster
        else:
            counts[0, -1] = 1
            counts = torch.cat([counts, zero_count], dim=1)

        d_radii = Variable(torch.ones(bsize, 1), requires_grad=False).to(const.device) # (1, 1)
        d_radii = d_radii * (torch.exp(self.log_sigma_u) + torch.exp(self.log_sigma_l))

        new_proto = Variable(self.base_distribution.data.to(const.device))
        if use_mean:
            new_proto = torch.mean(protos, dim=1).unsqueeze(
                0)  # + 0.01*Variable(torch.randn(1, 1, protos.shape[-1])).to(const.device)

        if ex is not None:
            protos[:, -1, :] = torch.exp(self.log_sigma_l) * new_proto + torch.exp(self.log_sigma_u) * Variable(
                ex.unsqueeze(0).unsqueeze(0).to(const.device)) / (torch.exp(self.log_sigma_l) + torch.exp(self.log_sigma_u) * nc)
        protos = torch.cat([protos, new_proto], dim=1)
        radii = torch.cat([radii, d_radii], dim=1)
        return nClusters, protos, counts, radii

    def _compute_priors(self,
            counts):
        """
        Args:
            counts: [B, nClusters] number of elements in each cluster
        Returns:
            priors: [B, nClusters] crp prior
        """
        ALPHA = self.config.ALPHA
        nClusters = counts.size()[1]

        crp_prior_old = torch.ones_like(counts)
        crp_prior_new = Variable(torch.FloatTensor([ALPHA])).to(const.device)
        crp_prior_new = torch.cat([crp_prior_new.unsqueeze(0)] * nClusters, 1)
        indices = counts.view(-1).nonzero()
        if torch.numel(indices) != 0:
            values = torch.masked_select(crp_prior_old, torch.gt(counts, 0.0))
            crp_priors = crp_prior_new.put_(indices,
                                            values)  # if going for uniform, can switch values to torch.ones_like(values)
        else:
            crp_priors = crp_prior_new

        return crp_priors

    def forward(self,
            sample,
            super_classes=False):
        batch = self._process_batch(sample, super_classes=super_classes)

        print(self.log_sigma_l.data)
        print(self.log_sigma_u.data)

        eps = self.config.eps
        xs = batch.x_train
        xq = batch.x_test

        # TODO: Is this trained?
        h_train = self._run_forward(xs) # [1, 30, 1, 28, 28]


        bsize = h_train.size()[0]
        h_test = self._run_forward(xq) # [1, 10, 1, 28, 28]

        y_train_np = batch.y_train.data.cpu().numpy() # (1, 30)

        nClusters = len(np.unique(y_train_np)) # 5
        kShot = int(batch.y_train.shape[1] / nClusters)
        prob_train = one_hot(batch.y_train, nClusters).to(const.device) # deterministic. 1 for that class and 0 otherwise

        protos = self._compute_protos(h_train, prob_train)
        radii = (torch.exp(self.log_sigma_l) * torch.exp(self.log_sigma_u) / (
                    torch.exp(self.log_sigma_l) + torch.exp(self.log_sigma_u) * kShot)) * Variable(
            torch.ones(bsize, nClusters), requires_grad=False).to(const.device) # (1, 5)

        total_prob = prob_train
        mu0 = torch.mean(protos, dim=1).unsqueeze(0) # (1, 1, 64), mean of all protos

        counts = self._get_count(total_prob, soft=True)
        priors = self._compute_priors(counts)

        # cluster unlabeled data
        if batch.x_unlabel is not None:
            h_unlabel = self._run_forward(batch.x_unlabel)
            h_all = torch.cat([h_train, h_unlabel], dim=1)

            nClusters, protos, counts, radii = self._add_cluster(nClusters, protos, counts, radii)

            priors = self._compute_priors(counts)

            # add one more cluster to the total probability tracked
            total_prob = torch.cat(
                [total_prob, Variable(torch.zeros(total_prob.size()[0], total_prob.size()[1], 1)).to(const.device)], dim=2)
            prob_train = torch.cat(
                [prob_train, Variable(torch.zeros(prob_train.size()[0], prob_train.size()[1], 1)).to(const.device)], dim=2)

            for ii in range(self.config.num_cluster_steps):
                all_unlabel_probs = None
                for i, ex in enumerate(h_unlabel[0]):
                    prob_unlabel = assign_cluster_radii_crp(protos, ex.unsqueeze(0).unsqueeze(0), radii,
                                                            priors)  # , priors)

                    if all_unlabel_probs is None:
                        all_unlabel_probs = prob_unlabel.data
                    else:
                        all_unlabel_probs = torch.cat([all_unlabel_probs, prob_unlabel.data], dim=1)

                        # adding a cluster!
                    if (prob_unlabel[0, 0, -1].item() > eps):
                        nClusters, protos, counts, radii = self._add_cluster(nClusters, protos, counts, radii,
                                                                             ex=ex.data, nc=eps)
                        prob_train = torch.cat(
                            [prob_train, Variable(torch.zeros(prob_train.size()[0], prob_train.size()[1], 1)).to(const.device)],
                            dim=2)
                        all_unlabel_probs = torch.cat([all_unlabel_probs, torch.zeros(all_unlabel_probs.size()[0],
                                                                                      all_unlabel_probs.size()[1],
                                                                                      1).to(const.device)], dim=2)

                        priors = self._compute_priors(counts)

                    total_prob = torch.cat([prob_train.data, all_unlabel_probs], dim=1)
                    counts = Variable(self._get_count(total_prob, soft=True))

                    priors = self._compute_priors(counts)

                total_prob = torch.cat([prob_train, Variable(all_unlabel_probs, requires_grad=False)], dim=1)
                counts = self._get_count(total_prob, soft=True)

                protos = self._compute_protos(h_all, total_prob)
                priors = self._compute_priors(counts)

        logits = compute_logits_radii_crp(protos, h_test, radii, priors)

        y_pred = torch.argmax(logits.data, dim=2).cpu()

        loss = F.cross_entropy(logits.squeeze(), batch.y_test.squeeze())

        acc_val = torch.eq(y_pred.squeeze(), batch.y_test.cpu()).float().mean()

        return loss, {
            'loss'  : loss.item(),
            'acc'   : acc_val.item(),
            'logits': logits.detach().cpu().numpy()
        }
