import pt_rpc_client
import pt_rpc_server
import grpc_client
import grpc_server

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29501")
    parser.add_argument("--role", type=str, default="client")
    parser.add_argument("--comm", type=str, default="ptrpc")

    args = parser.parse_args()

    if args.role == "client":
        if args.comm == "ptrpc":
            pt_rpc_client.run(addr=args.master_addr, port=args.master_port)
        elif args.comm == "grpc":
            grpc_client.run(addr=args.master_addr, port=args.master_port)
        else:
            raise ValueError(f"Unexpected role {args.comm}")
    elif args.role == "server":
        if args.comm == "ptrpc":
            pt_rpc_server.run(addr=args.master_addr, port=args.master_port)
        elif args.comm == "grpc":
            grpc_server.run(addr=args.master_addr, port=args.master_port)
        else:
            raise ValueError(f"Unexpected role {args.comm}")
    else:
        raise ValueError(f"Unexpected role {args.role}")


if __name__ == "__main__":
    main()
