import argparse
import hashlib
import os
import sys

import pandas as pd
import wget

from flamby.utils import accept_license, create_config, write_value_in_config


def main(output_folder, debug=False):
    """Download the heart disease dataset.

    Parameters
    ----------
    output_folder : str
        The folder where to download the dataset.
    """

    # location of the files in the UCI archive
    # accept_license(
    #     "https://archive-beta.ics.uci.edu/ml/datasets/heart+disease", "fed_heart_disease"
    # )
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    centers = ["cleveland", "hungarian", "switzerland", "va"]
    sha256_hashes = [
        'a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8',
        'd1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796',
        '834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314',
        'e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f'
    ]

    os.makedirs(output_folder, exist_ok=True)

    print(
        "This dataset is licensed under a Creative Commons Attribution 4.0 "
        "International (CC BY 4.0) license."
    )

    print("See https://archive-beta.ics.uci.edu/ml/datasets/heart+disease.\n")

    print(
        "Creators of the dataset:\n"
        "  1. Hungarian Institute of Cardiology."
        " Budapest: Andras Janosi, M.D.\n"
        "  2. University Hospital,"
        " Zurich, Switzerland: William Steinbrunn, M.D.\n"
        "  3. University Hospital,"
        " Basel, Switzerland: Matthias Pfisterer, M.D.\n"
        "  4. V.A. Medical Center, Long Beach and"
        " Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.\n"
    )

    print(
        "To cite this dataset, cite the following:"
        " Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, Detrano,"
        " Robert & M.D., M.D.. (1988). Heart Disease."
        " UCI Machine Learning Repository.\n"
    )

    # Creating config file with path to dataset
    dict, config_file = create_config(output_folder, debug, "fed_heart_disease")
    if dict["download_complete"]:
        print("You have already downloaded the heart disease dataset, aborting.")
        return

    # get status of download
    downloaded_status_file_path = os.path.join(output_folder, "download_status_file.csv")
    if not (os.path.exists(downloaded_status_file_path)):
        downloaded_status_file = pd.DataFrame()
        downloaded_status_file["Status"] = ["Not found"] * 4

        downloaded_status_file.to_csv(downloaded_status_file_path, index=False)
    else:
        downloaded_status_file = pd.read_csv(downloaded_status_file_path)

    # for each center, check if downloaded and download if necessary
    for i, center in enumerate(centers):
        file_status_ok = downloaded_status_file.loc[i, "Status"] == "Downloaded"

        if not file_status_ok:
            fname = wget.download(
                base_url + "processed." + center + ".data", out=output_folder
            )

            hash_sha256 = hashlib.sha256()
            with open(fname, "rb") as f:
                hash_sha256.update(f.read())

                if hash_sha256.hexdigest() == sha256_hashes[i]:
                    downloaded_status_file.loc[i, "Status"] = "Downloaded"
                else:
                    downloaded_status_file.loc[i, "Status"] = "Corrupted"

        print()

    # We assert we have everything and write it
    if all((downloaded_status_file["Status"] == "Downloaded").tolist()):
        write_value_in_config(config_file, "download_complete", True)
        write_value_in_config(config_file, "preprocessing_complete", True)
        downloaded_status_file.to_csv(downloaded_status_file_path, index=False)

    else:
        print("Downloading failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Where to store the downloaded data.",
        required=True,
    )

    args = parser.parse_args()
    main(args.output_folder)
