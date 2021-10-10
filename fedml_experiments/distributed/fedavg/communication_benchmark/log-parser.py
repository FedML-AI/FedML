import sys


# def add_round(round_logs, round_number):
#     if len(round_logs) != round_number:
#         raise Exception("Round Mismatch")
#     round_logs.append({"uplink_ticks": {}, "uplink_tocks": {}, "downlink_ticks": {}, "downlink_tocks": {}})
#     # print(round_logs[round_number])
#     # uplink_ticks = round_logs[round_number]['uplink_ticks']
#     # uplink_tocks = round_logs[round_number]['uplink_tocks']
#     # downlink_ticks = round_logs[round_number]['downlink_ticks']
#     # downlink_tocks = round_logs[round_number]['downlink_tocks']
#     # return (uplink_ticks, uplink_tocks, downlink_ticks, downlink_tocks)


# (uplink_ticks, uplink_tocks, downlink_ticks, downlink_tocks) = add_round(round_logs, self.round_numbers[_client_id])
# (uplink_ticks, uplink_tocks, downlink_ticks, downlink_tocks) = add_round(round_logs, round_numbers[_client_id])


# round_numbres = {0: 1, 1: 2}


# round_logs = {
#     0: {
#         "uplink_ticks": {0: [10, 20, 30, 40], 1: [10, 20, 30, 40]},
#         "uplink_tocks": {0: [10, 20, 30, 40], 1: [10, 20, 30, 40]},
#         "downlink_ticks": {0: [10, 20, 30, 40], 1: [10, 20, 30, 40]},
#         "downlink_tocks": {0: [10, 20, 30, 40], 1: [10, 20, 30, 40]},
#     }
# }
# self.rounds_start = {0: {0: 10, 1: 10, 2: 10}, 1: {0: 10, 1: 10, 2: 10}}
# self.rounds_end = {0: {0: 10, 1: 10, 2: 10}, 1: {0: 10, 1: 10, 2: 10}}


class Parser:
    def __init__(self):
        self.round_logs = {}
        self.round_numbers = {}
        self.rounds_start = {}
        self.rounds_end = {}

    def parse(self):
        with open(sys.argv[1], "r") as fi:
            for ln in fi:
                if ln.startswith("INFO:root:--Benchmark end round"):
                    parsed = ln.split()
                    _round = int(parsed[3])
                    _client_id = parsed[5]
                    # Making sure we are ending correct round
                    if _round != self.round_numbers[_client_id]:
                        raise Exception("Round Mismatch")
                    if not (_round in self.rounds_end):
                        self.rounds_end[_round] = {}
                    self.rounds_end[_round][_client_id] = parsed[7]
                    self.round_numbers[_client_id] += 1

                if ln.startswith("INFO:root:--Benchmark start round"):
                    parsed = ln.split()
                    _round = int(parsed[3])
                    _client_id = parsed[5]
                    if not (_client_id in self.round_numbers):
                        self.round_numbers[_client_id] = 0
                    # Making sure we are starting the correct round
                    if _round != self.round_numbers[_client_id]:
                        raise Exception("Round Mismatch")
                    if not (_round in self.rounds_start):
                        self.rounds_start[_round] = {}
                    self.rounds_start[_round][_client_id] = parsed[7]

                elif ln.startswith("INFO:root:--Benchmark"):
                    parsed = ln.split()
                    if parsed[1] == "tick":
                        if parsed[5] == "0":
                            self.append_log("uplink_ticks", parsed[3], float(parsed[7]))
                        else:
                            self.append_log("downlink_ticks", parsed[5], float(parsed[7]))
                    elif parsed[1] == "tock":
                        if parsed[5] == "0":
                            self.append_log("uplink_tocks", parsed[3], float(parsed[7]))
                        else:
                            self.append_log("downlink_tocks", parsed[5], float(parsed[7]))

    def append_log(
        self,
        key,
        client_id,
        value,
    ):
        if not (client_id in self.round_numbers):
            self.round_numbers[client_id] = 0
        _round_for_client = self.round_numbers[client_id]
        if not (_round_for_client in self.round_logs):
            self.round_logs[_round_for_client] = {}
        round_timestapms = self.round_logs[_round_for_client]

        # Key can be uplink_ticks, uplink_tocks, downlink_ticks or downlink_tocks
        if not (key in round_timestapms):
            round_timestapms[key] = {}
        if not (client_id in round_timestapms[key]):
            round_timestapms[key][client_id] = []
        round_timestapms[key][client_id].append(value)

    def print_resutls(self):
        round_uplink_delays = []
        round_downlink_delays = []
        round_uplink_delay_sum = []
        round_downlink_delay_sum = []
        for round_number, logs in self.round_logs.items():
            print(logs)
            print("ROUND " + str(round_number))
            uplink_ticks = logs["uplink_ticks"]
            uplink_tocks = logs["uplink_tocks"]
            downlink_ticks = logs["downlink_ticks"]
            downlink_tocks = logs["downlink_tocks"]
            uplink_delays = {}
            for key in uplink_ticks:
                process_total_delay = 0
                for i in range(len(uplink_ticks[key])):
                    process_total_delay += uplink_tocks[key][i] - uplink_ticks[key][i]
                uplink_delays[key] = process_total_delay

            downlink_delays = {}
            for key in downlink_ticks:
                process_total_delay = 0
                if len(downlink_ticks[key]) != len(downlink_tocks[key]):
                    print("DONWLINK AND UPLINK LENGTH DIFFER FOR: " + key)
                    print(str(len(downlink_ticks[key])) + " != " + str(len(downlink_tocks[key])))
                for i in range(min(len(downlink_ticks[key]), len(downlink_tocks[key]))):
                    process_total_delay += downlink_tocks[key][i] - downlink_ticks[key][i]
                    if downlink_tocks[key][i] < downlink_ticks[key][i]:
                        print("Tock before tick for")
                        print("Process_id: ", key)
                        print("Round: ", round_number)
                        print("Index: ", i)

                downlink_delays[key] = process_total_delay

            print("Uplink delays: ", uplink_delays)
            print("Downlink delays: ", downlink_delays)
            uplink_sum = 0
            for key in uplink_delays:
                uplink_sum += uplink_delays[key]

            downlink_sum = 0
            for key in downlink_delays:
                downlink_sum += downlink_delays[key]

            print("Uplink Sum: ", uplink_sum)
            print("Downlink Sum: ", downlink_sum)
            round_uplink_delays.append(uplink_delays)
            round_downlink_delays.append(downlink_delays)
            round_uplink_delay_sum.append(uplink_sum)
            round_downlink_delay_sum.append(downlink_sum)

        round_durations = {}
        print(self.rounds_end)
        print(self.rounds_start)
        for round_number in self.rounds_end.keys():
            round_durations[round_number] = {}
            for client_id in self.rounds_end[round_number].keys():
                round_durations[round_number][client_id] = float(self.rounds_end[round_number][client_id]) - float(self.rounds_start[round_number][client_id])


        print("!!!!!!!!!!!!!!!!!!!!!")
        # print("ROUND UPLINK DELAYS: ", round_uplink_delays)
        # print("ROUND DOWNLINK DELAYS: ", round_downlink_delays)
        print("ROUND UPLINK DELAYS SUM: ", round_uplink_delay_sum)
        print("ROUND DOWNLINK DELAYS SUM: ", round_downlink_delay_sum)
        print("ROUND DURATIONS: ", round_durations)

        print(self.round_logs)


if __name__ == "__main__":
    parser = Parser()
    parser.parse()
    parser.print_resutls()
