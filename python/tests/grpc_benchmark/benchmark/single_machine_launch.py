import torch.multiprocessing as mp
import pt_rpc_client
import pt_rpc_server
import grpc_client
import grpc_server


def main():
    ctx = mp.get_context("spawn")

    print("#######==== PyTorch RPC ===#######")
    targets = [pt_rpc_client.run, pt_rpc_server.run]
    processes = [ctx.Process(target=t) for t in targets]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("#######==== grpc ===#######")
    targets = [grpc_client.run, grpc_server.run]
    processes = [ctx.Process(target=t) for t in targets]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
