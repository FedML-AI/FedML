import csv


def build_ip_table(path):
    ip_config = dict()
    with open(path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip header line
        next(csv_reader)

        for row in csv_reader:
            receiver_id, receiver_ip = row
            ip_config[receiver_id] = receiver_ip
    return ip_config
