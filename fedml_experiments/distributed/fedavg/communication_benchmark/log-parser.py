import sys



def append_to_dic(dic, key, value):
    if not( key in dic):
        dic[key] = []
    dic[key].append(value)


def main():
    uplink_ticks = {}

    uplink_tocks = {}

    downlink_ticks = {}

    downlink_tocks = {}
    with open(sys.argv[1],"r") as fi:
        id = []
        for ln in fi:
            if ln.startswith("INFO:root:--Benchmark"):
                parsed = ln.split()
                if parsed[1] == 'tick':
                    if parsed[5] == '0':
                        append_to_dic(uplink_ticks, parsed[3], float(parsed[7]))
                    else:
                        append_to_dic(downlink_ticks, parsed[5], float(parsed[7]))
                elif parsed[1] == 'tock':
                    if parsed[5] == '0':
                        append_to_dic(uplink_tocks, parsed[3], float(parsed[7]))
                    else:
                        append_to_dic(downlink_tocks, parsed[5], float(parsed[7]))


    uplink_delays = {}
    for key in uplink_ticks:
        process_total_delay = 0
        for i in range(len(uplink_ticks[key])):
            process_total_delay += uplink_tocks[key][i] - uplink_ticks[key][i]
        uplink_delays[key] = process_total_delay


    downlink_delays = {}
    for key in downlink_ticks:
        process_total_delay = 0
        if (len(downlink_ticks[key]) != len(downlink_tocks[key])):
            print("DONWLINK AND UPLINK LENGTH DIFFER FOR: "+ key)
            print(str(len(downlink_ticks[key])) + " != " + str(len(downlink_tocks[key])))
        for i in range(min(len(downlink_ticks[key]),len(downlink_tocks[key]))):
            process_total_delay += downlink_tocks[key][i] - downlink_ticks[key][i]
            if (downlink_tocks[key][i] < downlink_ticks[key][i]):
                print("^^^^^^^^^^^^^^^^")
                print("key: ", key)
                print("i: ", i)
                print(downlink_tocks[key][i])
                print(downlink_ticks[key][i])

        downlink_delays[key] = process_total_delay


    print("Uplink delays: ",uplink_delays)
    print("Downlink delays: ",downlink_delays)
    uplink_sum = 0
    for key in uplink_delays:
        uplink_sum+= uplink_delays[key]
        
    downlink_sum = 0
    for key in downlink_delays:
        downlink_sum+= downlink_delays[key]

    print("Uplink Sum: ", uplink_sum)
    print("Downlink Sum: ", downlink_sum)

if __name__ == "__main__":
    main()