import logging
from fedml.core import ClientTrainer
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pathlib

class TfModelTrainerCLS(ClientTrainer):
    def __init__(self, model, dataset, args):
        super().__init__(model, args)
        self.model = model
        self.train_dataset_length = dataset[0]
        self.train_img_datagen = dataset[2]
        # self.val_dataset_length = dataset[1]
        # self.val_img_datagen = dataset[4]
        logging.info("client trainer - constructor called")
        
    def get_model_params(self):
        logging.info("client trainer - get_model_params")
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        logging.info("client trainer - set_model_params")
        self.model.set_weights(model_parameters)

    def train(self, train_data, device, args):
        logging.info("client trainer - Start training on Trainer {}".format(self.id))
        batch_size: int = args.batch_size
        epochs: int = args.epochs
        # FIXME (Mariem Abdou) changed from os.makedirs to pathlib.Path
        pathlib.Path("Model_dot_h5").mkdir(parents=True, exist_ok=True)
        filepath= os.path.join("Model_dot_h5", 'client_model.h5')
        checkpoint = ModelCheckpoint(filepath, 
                                     save_best_only=True, monitor='loss',
                                     verbose=1, mode='min', save_freq="epoch")
        
        steps_per_epoch = self.train_dataset_length // batch_size
        validation_steps = (self.train_dataset_length * 0.1) // batch_size
        
        logging.info("client trainer - here before training")
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.train_img_datagen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # FIXME (Mariem Abdou) Fix the validation!
            # validation_split=0.1, 
            # validation_data= self.val_img_datagen,
            # validation_steps = validation_steps,
            callbacks=checkpoint,
            verbose=1)
        logging.info("client trainer -here after training")

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        results = {
            "loss": history.history["loss"][0],
            "dice_coef": history.history["dice_coef_multilabel"][0],
            "val_loss": history.history["val_loss"][0],
        }
        
        logging.info("client trainer - metrics - loss", history.history["loss"][0])
        logging.info("client trainer - metrics - dice_coef_multilabel", history.history["dice_coef_multilabel"][0])
        logging.info(
                "client trainer - Client Index = {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, results["loss"], results["dice_coef_multilabel"])
                )
        
        return parameters_prime, results
