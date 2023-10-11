class VerticalMultiplePartyLogisticRegressionFederatedLearning(object):
    """
    Federated Learning class for logistic regression with multiple parties.

    Args:
        party_A (VFLGuestModel): The party with labels (party A).
        main_party_id (str, optional): The ID of the main party. Defaults to "_main".

    Methods:
        set_debug(is_debug):
            Set the debug mode for the federated learning.
        get_main_party_id():
            Get the ID of the main party.
        add_party(id, party_model):
            Add a party to the federated learning.

    Attributes:
        main_party_id (str): The ID of the main party.
        party_a (VFLGuestModel): The party with labels (party A).
        party_dict (dict): A dictionary to store other parties without labels.
        is_debug (bool): Flag to enable or disable debug mode.
    """
    def __init__(self, party_A, main_party_id="_main"):
        """
        Initialize the VerticalMultiplePartyLogisticRegressionFederatedLearning.

        Args:
            party_A (VFLGuestModel): The party with labels (party A).
            main_party_id (str, optional): The ID of the main party. Defaults to "_main".
        """
        super(VerticalMultiplePartyLogisticRegressionFederatedLearning, self).__init__()
        self.main_party_id = main_party_id
        # party A is the parity with labels
        self.party_a = party_A
        # the party dictionary stores other parties that have no labels
        self.party_dict = dict()
        self.is_debug = False

    def set_debug(self, is_debug):
        """
        Set the debug mode for the federated learning.

        Args:
            is_debug (bool): True to enable debug mode, False to disable.
        """
        self.is_debug = is_debug

    def get_main_party_id(self):
        """
        Get the ID of the main party.

        Returns:
            str: The ID of the main party.
        """
        return self.main_party_id

    def add_party(self, *, id, party_model):
        """
        Add a party to the federated learning.

        Args:
            id (str): The ID of the party.
            party_model: The model associated with the party.
        """
        self.party_dict[id] = party_model

    def fit(self, X_A, y, party_X_dict, global_step):
        """
        Perform the federated learning training.

        Args:
            X_A: The batch data for party A (with labels).
            y: The labels for party A.
            party_X_dict (dict): A dictionary of batch data for other parties.
            global_step: The global training step.

        Returns:
            float: The loss after training.
        """
        if self.is_debug:
            print("==> start fit")

        # set batch data for party A.
        # only party A has label.
        self.party_a.set_batch(X_A, y, global_step)

        # set batch data for all other parties
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, global_step)

        if self.is_debug:
            print("==> Guest receive intermediate computing results from hosts")
        comp_list = []
        for party in self.party_dict.values():
            logits = party.send_components()
            comp_list.append(logits)
        self.party_a.receive_components(component_list=comp_list)

        if self.is_debug:
            print("==> Guest train and computes loss")
        self.party_a.fit()
        loss = self.party_a.get_loss()

        if self.is_debug:
            print("==> Guest sends out common grad")
        grad_result = self.party_a.send_gradients()

        if self.is_debug:
            print("==> Hosts receive common grad from guest and perform training")
        for party in self.party_dict.values():
            party.receive_gradients(grad_result)

        return loss

    def predict(self, X_A, party_X_dict):
        """
        Perform predictions using the federated learning model.

        Args:
            X_A: The input data for party A (with labels).
            party_X_dict (dict): A dictionary of input data for other parties.

        Returns:
            array: Predicted labels.
        """
        comp_list = []
        for id, party_X in party_X_dict.items():
            comp_list.append(self.party_dict[id].predict(party_X))
        return self.party_a.predict(X_A, component_list=comp_list)
