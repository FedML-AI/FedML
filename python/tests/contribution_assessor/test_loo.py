import fedml
from fedml.core import ContributionAssessorManager
from utils import create_fake_model_list

if __name__ == "__main__":
    args = fedml.init()
    # load data
    dataset, output_dim = fedml.data.load(args)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    model_list_from_client_update = create_fake_model_list(4)
    model_aggregated = model_list_from_client_update[0]
    model_last_round = model_list_from_client_update[1]
    acc_on_aggregated_model = 0.85
    val_dataloader = test_data_local_dict[0]

    contribution_assessor_mgr = ContributionAssessorManager(args)
    contribution_vector = contribution_assessor_mgr.run(
        model_list_from_client_update, model_aggregated, model_last_round, acc_on_aggregated_model, val_dataloader
    )


