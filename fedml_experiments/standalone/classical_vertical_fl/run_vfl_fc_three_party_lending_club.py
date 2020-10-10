import os
import random
import sys

import numpy as np
import torch
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.data_preprocessing.lending_club_loan.lending_club_dataset import loan_load_three_party_data
from fedml_api.model.finance.vfl_models_standalone import LocalModel, DenseModel
from fedml_api.standalone.classical_vertical_fl.party_models import VFLGuestModel, VFLHostModel
from fedml_api.standalone.classical_vertical_fl.vfl import VerticalMultiplePartyLogisticRegressionFederatedLearning
from fedml_api.standalone.classical_vertical_fl.vfl_fixture import FederatedLearningFixture


def run_experiment(train_data, test_data, batch_size, learning_rate, epoch):
    Xa_train, Xb_train, Xc_train, y_train = train_data
    Xa_test, Xb_test, Xc_test, y_test = test_data

    print("################################ Wire Federated Models ############################")

    # create local models for both party A, party B and party C
    party_a_local_model = LocalModel(input_dim=Xa_train.shape[1], output_dim=10, learning_rate=learning_rate)
    party_b_local_model = LocalModel(input_dim=Xb_train.shape[1], output_dim=10, learning_rate=learning_rate)
    party_c_local_model = LocalModel(input_dim=Xc_train.shape[1], output_dim=10, learning_rate=learning_rate)
    # create lr model for both party A, party B and party C. Each party has a part of the whole lr model and
    # only party A has the bias since only party A has the labels.
    party_a_dense_model = DenseModel(party_a_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=True)
    party_b_dense_model = DenseModel(party_b_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=False)
    party_c_dense_model = DenseModel(party_c_local_model.get_output_dim(), 1, learning_rate=learning_rate, bias=False)
    partyA = VFLGuestModel(local_model=party_a_local_model)
    partyA.set_dense_model(party_a_dense_model)
    partyB = VFLHostModel(local_model=party_b_local_model)
    partyB.set_dense_model(party_b_dense_model)
    partyC = VFLHostModel(local_model=party_c_local_model)
    partyC.set_dense_model(party_c_dense_model)

    party_B_id = "B"
    party_C_id = "C"
    federatedLearning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.add_party(id=party_C_id, party_model=partyC)

    print("################################ Train Federated Models ############################")

    fl_fixture = FederatedLearningFixture(federatedLearning)

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train, party_C_id: Xc_train}}

    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test, party_C_id: Xc_test}}

    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=epoch, batch_size=batch_size)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(3)
    random.seed(0)

    print("################################ Prepare Data ############################")
    data_dir = "../../../data/lending_club_loan/"
    train, test = loan_load_three_party_data(data_dir)
    Xa_train, Xb_train, Xc_train, y_train = train
    Xa_test, Xb_test, Xc_test, y_test = test

    batch_size = 256
    epoch = 100
    lr = 0.01

    Xa_train, Xb_train, Xc_train, y_train = shuffle(Xa_train, Xb_train, Xc_train, y_train)
    Xa_test, Xb_test, Xc_test, y_test = shuffle(Xa_test, Xb_test, Xc_test, y_test)
    train = [Xa_train, Xb_train, Xc_train, y_train]
    test = [Xa_test, Xb_test, Xc_test, y_test]
    run_experiment(train_data=train, test_data=test, batch_size=batch_size, learning_rate=lr, epoch=epoch)
