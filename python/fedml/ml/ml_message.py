class MLMessage(object):
    """
    message type definition
    """


    """
        message payload keywords definition
    """
    MODEL_PARAMS = "model_params"
    GRAD_PARAMS = "grad_params"
    # PARAMS_TO_SERVER_OPTIMIZER = "params_to_server_optimizer"
    # PARAMS_TO_CLIENT_OPTIMIZER = "params_to_client_optimizer"

    SAMPLE_NUM_DICT = "sample_num_dict"
    TRAINING_NUM_IN_ROUND = "training_num_in_round"

    LOCAL_AGG_RESULT = "local_agg_result"
    LOCAL_COLLECT_RESULT = "local_collect_result"


