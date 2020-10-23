import pytest

import torch

from fedml_api.standalone.fedavg.optrepo import OptRepo


class TestOptRepo:
    def test_case_sensitivity(self):
        assert OptRepo.name2cls("adam") == OptRepo.name2cls("Adam")

    def test_correctness(self):
        assert OptRepo.name2cls("adam") == torch.optim.Adam
        assert OptRepo.name2cls("sgd") == torch.optim.SGD
        assert OptRepo.name2cls("adagrad") == torch.optim.Adagrad

    def test_invalid(self, caplog):
        with pytest.raises(KeyError):
            OptRepo.name2cls("invalid name")
        assert caplog.messages[0].startswith("Invalid optimizer: ")
        assert caplog.messages[1].startswith("These optimizers are registered: ")
