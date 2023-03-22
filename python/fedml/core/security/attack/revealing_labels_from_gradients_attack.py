import numpy as np
from typing import Dict, Any
from .attack_base import BaseAttackMethod
from ..common import utils

"""
attack @ server, 07/09/2022
Revealing and Protecting Labels in Distributed Training
https://github.com/googleinterns/learning-bag-of-words

this work supports image classification over ResNet or EfficientNet.
"""


# todo: need to check together with other label revealing attacks
# todo: the author indicates this attack can be applied to a parameter update generated from a mini-batch (1-step, N -sample),
#  or after several steps (K-step, 1-sample each),
#  or even the most general case of K-step update of N-sample at each step.
#  Currently we only implement this attack on local gradients.  -- to design a new attack API?? gradient --> labels
# todo: need to test on real data


class RevealingLabelsFromGradientsAttack(BaseAttackMethod):
    def __init__(self, batch_size, model_type):
        self.batch_size = batch_size
        self.model_type = model_type

    def reconstruct_data(self, a_gradient: dict, extra_auxiliary_info: Any = None):
        vec_local_weight = utils.vectorize_weight(a_gradient)
        print(vec_local_weight)

        gt_labels = set(extra_auxiliary_info.tolist())
        # for item_index, (k, v) in enumerate(local_w.items()):
        for k in a_gradient.keys():
            if utils.is_weight_param(k):
                self._attack_on_gradients(gt_labels, a_gradient[k])
        return

    def _attack_on_gradients(self, gt_labels, v):
        grads = np.sign(v)
        _, pred_labels = self._infer_labels(grads, gt_k=self.batch_size, epsilon=1e-10)
        print("In gt, not in pr:", [i for i in gt_labels if i not in pred_labels])
        print("In pr, not in gt:", [i for i in pred_labels if i not in gt_labels])

    def _infer_labels(self, grads, gt_k=None, epsilon=1e-8):
        m, n = np.shape(grads)
        B, s, C = np.linalg.svd(grads, full_matrices=False)
        pred_k = np.linalg.matrix_rank(grads)
        k = gt_k or pred_k
        print("Predicted length of target sequence:", pred_k)
        print("Finding SVD of W...")
        print(s[:gt_k])
        print(s[gt_k])
        C = C[:k, :].astype(np.double)

        # Find x: x @ C has only one positive element
        # Filter possible labels using perceptron algorithm
        bow = []
        if self.model_type == "ResNet50":
            bow = np.reshape(np.where(np.min(grads, 0) < 0), -1).tolist()
        for i in range(n):
            if i in bow:
                continue
            indices = [j for j in range(n) if j != i]
            np.random.shuffle(indices)
            if self._solve_perceptron(
                X=np.concatenate([C[:, i : i + 1], C[:, indices[:999]]], 1).transpose(),
                y=np.array([1 if j == 0 else -1 for j in range(1000)]),
                fit_intercept=True,
                max_iter=1000,
                tol=1e-3,
            ):
                bow.append(i)

        # Get the final set with linear programming
        ret_bow = []
        for i in bow:
            if i in ret_bow:
                continue
            indices = [j for j in range(n) if j != i]
            D = np.concatenate([C[:, i : i + 1], C[:, indices]], 1)
            indices2 = np.argsort(np.linalg.norm(D[:, 1:], axis=0))[-199:]
            grads = np.concatenate([D[:, 0:1], -D[:, 1 + indices2]], 1).transpose()
            if self.solve_lp(
                grads=grads,
                b=np.array([-epsilon] + [0] * len(indices2)),
                c=np.array(C[:, i : i + 1]),
            ):
                ret_bow.append(i)
        return pred_k, ret_bow

    @staticmethod
    def _solve_perceptron(X, y, fit_intercept=True, max_iter=1000, tol=1e-3, eta0=1.0):
        from sklearn.linear_model import Perceptron

        clf = Perceptron(
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, eta0=eta0
        )
        clf.fit(X, y)
        if not fit_intercept:
            pass
        if clf.score(X, y) > 0.9:
            return True
        return False

    @staticmethod
    def solve_lp(grads, b, c):
        # from cvxopt import matrix, solvers

        np.solvers.options["show_progress"] = False
        np.random.seed(None)
        for t in range(1):
            grads, b, c = np.matrix(grads), np.matrix(b), np.matrix(c)
            sol = np.solvers.lp(c, grads, b)
            x = sol["x"]
            if x is not None:
                ret = grads * x
                if (
                    ret[0] < -0.1
                    and np.max(ret[1:]) < 1e-2
                    and np.count_nonzero(np.array(ret[1:]) <= 0) > 0.5 * len(ret)
                ):
                    return True
        return False
