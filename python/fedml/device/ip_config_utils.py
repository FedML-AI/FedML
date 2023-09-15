import csv


def build_ip_table(path):
    """
    Build an IP table from a CSV file containing receiver IDs and their corresponding IP addresses.

    Args:
        path (str): The path to the CSV file.

    Returns:
        dict: A dictionary mapping receiver IDs to their respective IP addresses.
    """
    ip_config = dict()
    with open(path, newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip header line
        next(csv_reader)

        for row in csv_reader:
            receiver_id, receiver_ip = row
            ip_config[receiver_id] = receiver_ip
    return ip_config
