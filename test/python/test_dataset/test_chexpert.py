import os
import pytest
from fedml.data.chexpert import CheXpertSmall

data_path = os.path.join(os.path.dirname(__file__), 'data')

pytest.skip("skipping windows-only tests", allow_module_level=True)

class TestCheXpert:
    def test_chexpert_train(self):
        cs = CheXpertSmall(data_path, train=True, transform=None, download=False, policy='zeros')
        img, label = cs[0]
        assert len(cs) == 223414

    def test_chexpert_valid(self):
        cs = CheXpertSmall(data_path, train=False, transform=None, download=False, policy='zeros')
        assert len(cs) == 234

if __name__ == '__main__':
    tc = TestCheXpert()
    tc.test_chexpert_train()
    tc.test_chexpert_valid()