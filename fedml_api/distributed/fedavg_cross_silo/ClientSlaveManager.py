import logging


class ClientSlaveManager:
    def __init__(self, args, dist_worker):
        self.dist_worker = dist_worker
        self.args = args
        self.round_idx = 0
        self.num_rounds = args.comm_round
        self.finished = False
        self.dist_worker.update_dataset()

    def train(self):
        self.dist_worker.train(self.round_idx)
        self.round_idx += 1
        if self.round_idx == self.num_rounds:
            self.finish()

    def finish(self):
        # pass
        self.dist_worker.cleanup_pg()
        logging.info(
            "Training finsihded for slave client rank %s in silo %s" % (self.args.silo_proc_rank, self.args.silo_rank)
        )
        self.finished = True

    def run(self):
        while not self.finished:
            self.train()
