import socket
import sys
import numpy as np
import pandas as pd
import logging


def scheduler(round_idx, client_num_in_total, client_num_per_round):
    # read feedback
    logging.info("start read feedBack")
    logging.info("stop read feedBack")
    
    # random sample clients
    if client_num_in_total == client_num_per_round:
        client_indexes = [client_index for client_index in range(client_num_in_total)]
    else:
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
    logging.info("client_indexes = %s" % str(client_indexes))

    # random local iterations
    local_itr = np.random.randint(2) + 1

    # radio resources
    radio_res = np.sort(np.random.uniform(0, 1, len(client_indexes)))
    radio_res = radio_res/sum(radio_res)

    s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
    s_local_itr = str(local_itr)
    s_radio_res = str(list(radio_res))[1:-1].replace(',', '')

    return s_client_indexes + "," + s_local_itr + "," + s_radio_res

client_num_in_total = int(sys.argv[1])
client_num_per_round = int(sys.argv[2])

logging.basicConfig()

#setup the socket server
host = socket.gethostname()
server = socket.socket()
server.bind((host, 8999))

server.listen(5) # maximum connections 5.

round_idx = 0

while True:
    client, addr = server.accept() # connected to client.
    print("round " + str(round_idx) + " connected address: " + str(addr))

    mes = scheduler(round_idx, client_num_in_total, client_num_per_round)
    client.send(mes.encode())

    client.close() # close the connection

    round_idx = round_idx + 1
