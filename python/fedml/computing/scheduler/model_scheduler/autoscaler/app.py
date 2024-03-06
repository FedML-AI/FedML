import conf

from autoscaler.autoscaler_app import app
from utils import logger

if __name__ == '__main__':

    # if run in debug mode, process will be single threaded by default
    # however, we set debug=False because we want to avoid double initialization
    app.run(host=conf.AUTOSCALER_HOST, port=conf.AUTOSCALER_PORT, debug=False)
