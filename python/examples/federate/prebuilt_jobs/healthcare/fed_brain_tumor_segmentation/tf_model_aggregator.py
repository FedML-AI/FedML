import logging
import tensorflow as tf
from fedml import mlops
from fedml.core import ServerAggregator
import os

class TfServerAggregator(ServerAggregator):
    def __init__(self, model, dataset, args):
        super().__init__(model, args)
        self.model = model
        self.test_img_datagen = dataset[3] # test_data_global = imageLoader(test)
        self.test_dataset_length = dataset[1] # test_data_num
        logging.info("server aggregator - constructor initailized")

    def get_model_params(self):
        logging.info("server aggregator - get_model_params")
        return self.model.get_weights()

    def set_model_params(self, model_parameters):
        logging.info("server aggregator - set_model_params")
        self.model.set_weights(model_parameters)

    def test(self, test_data, device, args):
        logging.info("server aggregator - Evaluating on Trainer ID: {}".format(self.id))
        loss, dice_coef_multilabel,mean_io_u_4 = self.model.evaluate(self.test_img_datagen, steps=self.test_dataset_length//args.batch_size)
        metrics = {"loss": loss, 
                   "dice_coef_multilabel": dice_coef_multilabel,
                   "mean_io_u_4": mean_io_u_4,
                   }
        logging.info("server aggregator - loss", loss)
        logging.info("server aggregator - dice_coef", dice_coef_multilabel)
        logging.info("server aggregator - mean_io_u_4", mean_io_u_4)

        self.model.save(os.path.join("", "content", "Aggregated_model.h5"))
        return metrics

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        loss, dice_coef_multilabel,mean_io_u_4 = self.model.evaluate(self.test_img_datagen, steps=self.test_dataset_length//args.batch_size)
        metrics = {"loss": loss, 
                   "dice_coef_multilabel": dice_coef_multilabel,
                   "mean_io_u_4": mean_io_u_4,
                   }
        logging.info("server aggregator - loss", loss)
        logging.info("server aggregator - dice_coef", dice_coef_multilabel)
        logging.info("server aggregator - mean_io_u_4", mean_io_u_4)
        self.model.save(os.path.join("", "content", "Aggregated_model.h5"))
        return True
