import os
import json
import time
import platform
import logging

def logging_config(args, process_id):
    # customize the log format
    while logging.getLogger().handlers:
        logging.getLogger().handlers.clear()
    console = logging.StreamHandler()
    if args.level == 'INFO':
        console.setLevel(logging.INFO)
    elif args.level == 'DEBUG':
        console.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    formatter = logging.Formatter(str(process_id) + 
        ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)
    # logging.getLogger().info("test")
    logging.basicConfig()
    logger = logging.getLogger()
    if args.level == 'INFO':
        logger.setLevel(logging.INFO)
    elif args.level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    else:
        raise NotImplementedError
    logging.info(args)

