import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
from torch.optim import lr_scheduler


import flamby

from flamby.datasets.fed_kits19 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedKits19,
    evaluate_dice_on_tests,
    metric,
    softmax_helper,
)
from flamby.utils import (
    check_dataset_from_config,
    create_config,
    get_config_file_path,
    write_value_in_config,
)
import argparse
import csv
import os
import shutil
from collections import defaultdict

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    save_json,
    subfolders,
)

from flamby.datasets.fed_kits19.dataset_creation_scripts.utils.set_environment_variables import (
    set_environment_variables,
)
from flamby.utils import get_config_file_path, read_config, write_value_in_config
from flamby.datasets.fed_kits19.dataset_creation_scripts.parsing_and_adding_metadata import (
    read_csv_file,
)


def preprocess(args):
    import argparse
    import nnunet
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
    from nnunet.experiment_planning.utils import crop
    from nnunet.paths import (
        nnUNet_raw_data,
        nnUNet_cropped_data,
        preprocessing_output_dir,
    )
    import shutil
    from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
    from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
    from nnunet.training.model_restore import recursive_find_python_class

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task_ids",
        nargs="+",
        help="List of integers belonging to the task ids you wish to run"
        " experiment planning and preprocessing for. Each of these "
        "ids must, have a matching folder 'TaskXXX_' in the raw "
        "data folder",
    )
    parser.add_argument(
        "-pl3d",
        "--planner3d",
        type=str,
        default="ExperimentPlanner3D_v21",
        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
        "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
        "configured",
    )
    parser.add_argument(
        "-pl2d",
        "--planner2d",
        type=str,
        default="ExperimentPlanner2D_v21",
        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
        "Can be 'None', in which case this U-Net will not be configured",
    )
    parser.add_argument(
        "-no_pp",
        action="store_true",
        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
        "will only run the experiment planning and create the plans file",
    )
    parser.add_argument(
        "-tl",
        type=int,
        required=False,
        default=8,
        help="Number of processes used for preprocessing the low resolution data for the 3D low "
        "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
        "RAM",
    )
    parser.add_argument(
        "-tf",
        type=int,
        required=False,
        default=8,
        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
        "3D U-Net. Don't overdo it or you will run out of RAM",
    )
    parser.add_argument(
        "--verify_dataset_integrity",
        required=False,
        default=False,
        action="store_true",
        help="set this flag to check the dataset integrity. This is useful and should be done once for "
        "each dataset!",
    )
    parser.add_argument(
        "-overwrite_plans",
        type=str,
        default=None,
        required=False,
        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
        "configure automatically. This will overwrite everything: intensity normalization, "
        "network architecture, target spacing etc. Using this is useful for using pretrained "
        "model weights as this will guarantee that the network architecture on the target "
        "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
        "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
        "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
        "Make sure to only use plans files that were "
        "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
        "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
        "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
        "no support given.\n"
        "Note that this will first print the old plans (which are going to be overwritten) and "
        "then the new ones (provided that -no_pp was NOT set).",
    )
    parser.add_argument(
        "-overwrite_plans_identifier",
        type=str,
        default=None,
        required=False,
        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
        "where to look for the correct plans and data. Assume your identifier is called "
        "IDENTIFIER, the correct training command would be:\n"
        "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
        "-pretrained_weights FILENAME'",
    )

    args = parser.parse_args(args=args)
    logging.info("Preprocessing... with arguments: {}".format(args))

    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            print(
                "Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                "skip 2d planning and preprocessing."
            )
        assert planner_name3d == "ExperimentPlanner3D_v21_Pretrained", (
            "When using --overwrite_plans you need to use "
            "'-pl3d ExperimentPlanner3D_v21_Pretrained'"
        )

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        crop(task_name, False, tf)

        tasks.append(task_name)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class(
            [search_in], planner_name3d, current_module="nnunet.experiment_planning"
        )
        if planner_3d is None:
            raise RuntimeError(
                "Could not find the Planner class %s. Make sure it is located somewhere in "
                "nnunet.experiment_planning" % planner_name3d
            )
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class(
            [search_in], planner_name2d, current_module="nnunet.experiment_planning"
        )
        if planner_2d is None:
            raise RuntimeError(
                "Could not find the Planner class %s. Make sure it is located somewhere in "
                "nnunet.experiment_planning" % planner_name2d
            )
    else:
        planner_2d = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        # splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        # lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, "dataset.json"))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = (
            True if (("CT" in modalities) or ("ct" in modalities)) else False
        )
        dataset_analyzer = DatasetAnalyzer(
            cropped_out_dir, overwrite=False, num_processes=tf
        )  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset(
            collect_intensityproperties
        )  # this will write output files that will be used by the ExperimentPlanner

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(
            join(cropped_out_dir, "dataset_properties.pkl"),
            preprocessing_output_dir_this_task,
        )
        shutil.copy(
            join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task
        )

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")

        if planner_3d is not None:
            if args.overwrite_plans is not None:
                assert (
                    args.overwrite_plans_identifier is not None
                ), "You need to specify -overwrite_plans_identifier"
                exp_planner = planner_3d(
                    cropped_out_dir,
                    preprocessing_output_dir_this_task,
                    args.overwrite_plans,
                    args.overwrite_plans_identifier,
                )
            else:
                exp_planner = planner_3d(
                    cropped_out_dir, preprocessing_output_dir_this_task
                )
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)
        if planner_2d is not None:
            exp_planner = planner_2d(
                cropped_out_dir, preprocessing_output_dir_this_task
            )
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads)


def config_dataset(args):
    dict_, config_file_ = create_config(args.data_cache_dir, args.debug, "fed_kits19")
    logging.info(f"dict_: {dict_}, config_file_: {config_file_}")

    set_environment_variables(args.debug)
    from nnunet.paths import base, nnUNet_raw_data

    path_to_config_file = get_config_file_path("fed_kits19", debug=args.debug)
    dict = read_config(path_to_config_file)
    base = base + "data"
    task_id = 64
    task_name = "KiTS_labelsFixed"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)
    csv_path = os.path.join(
        os.path.dirname(flamby.__file__),
        "datasets",
        "fed_kits19",
        "metadata",
    )
    logging.info(f"csv_path: {csv_path}")
    case_ids, site_ids, unique_hospital_ids, thresholded_ids = read_csv_file(
        csv_path=csv_path
    )

    logging.info(f"thresholded_ids: {thresholded_ids}")
    if args.debug:
        logging.info("debug mode")
        train_patients = thresholded_ids[:1]
        test_patients = all_cases[210:211]  # we do not need the test data
    else:
        train_patients = thresholded_ids
        test_patients = all_cases[210:211]  # we do not need the test data

    for p in train_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    logging.info(f"train_patient_names: {train_patient_names}")

    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)

    logging.info(f"test_patient_names: {test_patient_names}")

    json_dict = {}
    json_dict["name"] = "KiTS"
    json_dict["description"] = "kidney and kidney tumor segmentation"
    json_dict["tensorImageSize"] = "4D"
    json_dict["reference"] = "KiTS data for nnunet_library"
    json_dict["licence"] = ""
    json_dict["release"] = "0.0"
    json_dict["modality"] = {
        "0": "CT",
    }
    json_dict["labels"] = {"0": "background", "1": "Kidney", "2": "Tumor"}

    json_dict["numTraining"] = len(train_patient_names)
    json_dict["numTest"] = len(test_patient_names)
    json_dict["training"] = [
        {
            "image": "./imagesTr/%s.nii.gz" % i.split("/")[-1],
            "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1],
        }
        for i in train_patient_names
    ]
    json_dict["test"] = [
        "./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names
    ]
    logging.info(f"saving json_dict...")
    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    logging.info(path_to_config_file)
    write_value_in_config(path_to_config_file, "download_complete", True)
    logging.info(f"download_complete: {True}")

    # from nnunet.experiment_planning.nnUNet_plan_and_preprocess import main
    if not args.preprocessed:
        preprocess(
            ["-t", "064", "-tf", str(args.worker_num), "-tl", str(args.worker_num)]
        )

    path_to_config_file = get_config_file_path("fed_kits19", False)
    write_value_in_config(path_to_config_file, "preprocessing_complete", True)
    logging.info(f"preprocessing_complete: {True}")

    # dict_ = check_dataset_from_config(dataset_name="fed_kits19", debug=False)
    # print(dict_)


def load_partition_fed_kits19(args):
    torch.use_deterministic_algorithms(False)

    config_dataset(args)

    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    nc = 3

    if args.process_id == 0:  # server
        pass
    else:  # client
        logging.info(f"load center {int(args.process_id)-1} data")
        client_idx = int(args.process_id) - 1
        train_dataset = FedKits19(
            center=client_idx, train=True, pooled=True, debug=args.debug
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.worker_num,
        )
        test_dataset = FedKits19(
            center=client_idx, train=False, pooled=True, debug=args.debug
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.worker_num,
            drop_last=True,
        )
        data_local_num_dict[client_idx] = len(train_dataset)
        train_data_local_dict[client_idx] = train_dataloader
        test_data_local_dict[client_idx] = test_dataloader

    # logging.info(f"train_data_num: {train_data_num}")
    # logging.info(f"test_data_num: {test_data_num}")
    # logging.info(f"train_data_global: {train_data_global}")
    # logging.info(f"test_data_global: {test_data_global}")
    # logging.info(f"data_local_num_dict: {data_local_num_dict}")
    # logging.info(f"train_data_local_dict: {train_data_local_dict}")
    # logging.info(f"test_data_local_dict: {test_data_local_dict}")
    # logging.info(f"nc: {nc}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        nc,
    )
