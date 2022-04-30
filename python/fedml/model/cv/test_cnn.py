from ptflops import get_model_complexity_info

from .cnn import CNN_DropOut

if __name__ == "__main__":
    # net = CNN_OriginalFedAvg()
    net = CNN_DropOut()

    flops, params = get_model_complexity_info(
        net, (1, 28, 28), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print(params)
    print("{:<30}  {:<8}".format("Computational complexity: ", flops))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
