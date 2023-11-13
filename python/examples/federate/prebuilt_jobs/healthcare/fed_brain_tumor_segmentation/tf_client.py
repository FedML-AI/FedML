import logging
import fedml
from fedml import FedMLRunner
from tf_model_trainer import TfModelTrainerCLS
from tf_model import multi_unet_model, dice_coef_multilabel, dice_coef_multilabel_loss
from data_loader import load_data
import subprocess
import sys
from keras.metrics import MeanIoU

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
if __name__ == "__main__":
    # install("tensorflow")
    # init FedML framework
    logging.info("client/server - started")
    args = fedml.init()
    logging.info("client/server - args loaded")

    # init device
    device = fedml.device.get_device(args)
    logging.info("client/server - device inited")
    
    # load data
    dataset = load_data(args)
    logging.info("client/server - data loaded")

    model = multi_unet_model((args.input_dim, args.input_dim, args.input_channels))
    logging.info("client/server - model created")
    
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss=dice_coef_multilabel_loss,
                  metrics=[dice_coef_multilabel, MeanIoU(num_classes=6)])

    logging.info("client/server - model compiled")
    
    # create model trainer
    model_trainer = TfModelTrainerCLS(model, dataset, args)
    logging.info("Model trainer created")

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=model_trainer)
    fedml_runner.run()
    logging.info("client/server - fedml run is run")
