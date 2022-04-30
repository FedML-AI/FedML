import fedml
import torch
from fedml.cross_silo.hierarchical import Client
from data_loader import load_data
import torch.multiprocessing as mp

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x.reshape(x.shape[0],-1)))
        return outputs

def main(process_rank, args):
    # Set inra-silo argiments
    args.local_rank = process_rank
    args.silo_proc_num = args.nnode * args.nproc_per_node
    args.silo_proc_rank = args.node_rank * args.nproc_per_node + args.local_rank
    args.pg_master_port += args.rank

    if not hasattr(args, 'enable_cuda_rpc'):
        args.enable_cuda_rpc = False

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # model = fedml.model.create(args, output_dim)
    model = LogisticRegression(3072, output_dim)

    # start training
    client = Client(args, device, dataset, model)
    client.run()


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # Swapn multiple processes in the node based on nproc_per_node
    mp.spawn(main,
             args=(args,),
             nprocs=args.nproc_per_node,
             join=True)

    