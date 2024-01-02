import logging
import math

import numpy as np
from torch.optim import SGD
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge:
    """
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    """

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert len(candidate) == 1
        assert len(refs) > 0
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(
                rec_max + self.beta ** 2 * prec_max
            )
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert gts.keys() == res.keys()
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def freeze_model_parameters(model, freeze_layers):
    modules = list()
    logging.info("freeze layers: %s" % str(freeze_layers))
    for layer_idx in freeze_layers:
        if layer_idx == "e":
            modules.append(model.distilbert.embeddings)
        else:
            modules.append(model.distilbert.transformer.layer[int(layer_idx)])
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
    logging.info(get_parameter_number(model))


def build_optimizer(model, iteration_in_total, args):
    warmup_steps = math.ceil(iteration_in_total * args.warmup_ratio)
    args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps
    logging.info("warmup steps = %d" % args.warmup_steps)
    # freeze exps only apply for distilbert
    # if args.model_type == "distilbert":
    #    freeze_model_parameters(model)
    if args.optimizer == "AdamW":
        optimizer = AdamW(
            model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon
        )
    else:
        optimizer = SGD(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=iteration_in_total,
    )
    return optimizer, scheduler
