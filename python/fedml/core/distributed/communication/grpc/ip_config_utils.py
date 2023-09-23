import csv


def build_ip_table(path):
    """
    Builds an IP table from a CSV file.

    Args:
        path (str): The path to the CSV file containing receiver IDs and IP addresses.

    Returns:
        dict: A dictionary mapping receiver IDs to IP addresses.
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
