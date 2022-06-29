from ..core import mlops


def init(args):
    mlops.init(args)


def event(event_name, event_started=True, event_value=None, event_edge_id=None):
    mlops.event(event_name, event_started, event_value, event_edge_id)


def log(metrics):
    mlops.log(metrics)

