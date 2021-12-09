class VerticalMultiplePartyLogisticRegressionFederatedLearning(object):

    def __init__(self, party_A, fascilator,main_party_id="_main"):
        super(VerticalMultiplePartyLogisticRegressionFederatedLearning, self).__init__()
        self.fascilator = fascilator
        self.main_party_id = main_party_id
        # party A is the parity with labels
        self.party_a = party_A
        # the party dictionary stores other parties that have no labels
        self.party_dict = dict()
        self.is_debug = False

    def set_debug(self, is_debug):
        self.is_debug = is_debug

    def get_main_party_id(self):
        return self.main_party_id

    def add_party(self, *, id, party_model):
        self.party_dict[id] = party_model

    def fit(self, y, party_X_dict, global_step):
        if self.is_debug: print("==> start fit")

        self.party_a.set_batch(y, global_step)

        if self.is_debug: print("==> Set Batch for all hosts")
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, global_step)

        if self.is_debug: print("==> Facilitator receives intermediate computing results from hosts")
        for party_id, party in self.party_dict.items():
            activations = party.send_components()
            self.fascilator.receive_activations(activations,party_id)

        if self.is_debug: print("==> Facilitator concatenates results")
        concatenation = self.fascilator.receive_concatination()

        if self.is_debug: print("==> Guest computes loss")
        self.party_a.receive_concatination(concatenation)
        loss = self.party_a.send_loss()
        grad_result = self.party_a.send_gradients()
        self.fascilator.receive_gradients(grad_result)

        if self.is_debug: print("==> Hosts receive common grad from facilitator and perform training")

        for party_id, party in self.party_dict.items():
            back_prop = self.fascilator.perform_back(party_id)
            party.receive_gradients(back_prop)

        return loss

    def predict(self,party_X_dict):

        for id, party_X in party_X_dict.items():
            print(len(party_X))
            activations = self.party_dict[id].send_predict(party_X)
            self.fascilator.receive_activations(activations,id)

        self.fascilator.receive_concatination()
        return self.party_a.predict(activations)