import torch
from fedml.core.security.defense.cclip_defense import CClipDefense
from fedml.core.security.test.utils import create_fake_model_list

def test_defense():
    model_list = create_fake_model_list(20)
    cclip = CClipDefense(tau=10)
    averaged_params, cclip_scores = cclip.defend(model_list, model_list[0][1])
    print(f"averaged_params={averaged_params}")
    print(f"cclip_scores={cclip_scores}")

if __name__ == "__main__":
    test_defense()
