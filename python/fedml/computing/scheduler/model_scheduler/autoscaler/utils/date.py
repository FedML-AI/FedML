import datetime

def convert_to_datetime(ts, datetime_format):
    return datetime.datetime.strptime(ts, datetime_format)
