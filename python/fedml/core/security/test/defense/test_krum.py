import torch
from fedml.core.security.defense.krum_defense import KrumDefense
from fedml.core.security.test.utils import create_fake_model_list

def test_defense():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    krum = KrumDefense(k=2, multi=False)
    filtered_model_list, krum_scores = krum.defend(model_list)
    print(f"filtered_model_list={filtered_model_list}")
    print(f"krum_scores={krum_scores}")

    print(f"test multi-krum")
    krum = KrumDefense(k=2, multi=True)
    filtered_model_list, krum_scores = krum.defend(model_list)
    print(f"filtered_model_list={filtered_model_list}")
    print(f"krum_scores={krum_scores}")

if __name__ == "__main__":
    test_defense()
