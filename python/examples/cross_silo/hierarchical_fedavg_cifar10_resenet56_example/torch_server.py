import fedml
import torch
from fedml.cross_silo.hierarchical import Server

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x.reshape(x.shape[0],-1)))
        return outputs

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # Set inra-silo argiments
    args.local_rank = 0
    args.nproc_per_node = 1
    args.silo_proc_num = 1
    args.silo_proc_rank = 0
    if not hasattr(args, 'enable_cuda_rpc'):
        args.enable_cuda_rpc = False

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # model = fedml.model.create(args, output_dim)
    model = LogisticRegression(3072, output_dim)

    # start training
    server = Server(args, device, dataset, model)
    server.run()
