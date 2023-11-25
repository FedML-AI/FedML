import os
import urllib.request

S3_ROOT = "https://fedml-scalellm.s3.us-west-1.amazonaws.com/"
LOCAL_ROOT = os.environ.get("LOCAL_ROOT", "/data/scalellm_share_dir")
ENGINE_DICT = {
    "stable-diffusion": {
        "intermediate_path": "Stable-Diffusion/",
        "files": [
            "engine_xl_base/clip.plan",
            "engine_xl_base/clip2.plan",
            "engine_xl_base/unetxl.plan",
            "engine_xl_refiner/clip2.plan",
            "engine_xl_refiner/unetxl.plan",
        ],
    },
}


def download_engine(engine_name: str, hash_check: bool = False) -> str:
    if engine_name not in ENGINE_DICT:
        return ""

    s3_address_prefix = S3_ROOT
    local_dst_refix = LOCAL_ROOT
    intermediate_path = ENGINE_DICT[engine_name]["intermediate_path"]

    # Local engine folder
    engine_dir = os.path.join(os.path.expanduser(local_dst_refix), intermediate_path)
    # Remote s3 address
    s3_folder_address = s3_address_prefix + intermediate_path

    if not os.path.exists(engine_dir):
        os.makedirs(engine_dir)
        for file_name in ENGINE_DICT[engine_name]["files"]:
            print(f"Downloading {file_name} from {s3_folder_address} to {engine_dir} ... May take a while.")
            if not os.path.exists(os.path.dirname(os.path.join(engine_dir, file_name))):
                os.makedirs(os.path.dirname(os.path.join(engine_dir, file_name)))

            urllib.request.urlretrieve(s3_folder_address + file_name, os.path.join(engine_dir, file_name))
    else:
        if not hash_check:
            print(f"Engine folder {engine_dir} already exists. Skipping download.")
        else:
            if "hashes" not in ENGINE_DICT[engine_name]:
                print(f"No hash list found for {engine_name}. Please delete the folder and retry.")
                return ""
            hash_list = ENGINE_DICT[engine_name]["hashes"]
            import hashlib
            hash_sha256 = hashlib.sha256()
            for file_name in ENGINE_DICT[engine_name]["files"]:
                file = os.path.join(engine_dir, file_name)
                with open(file, "rb") as f:
                    hash_sha256.update(f.read())
                if hash_sha256.hexdigest() != hash_list[file_name]:
                    print(f"Hash check failed for {file_name}. Please delete the file and retry.")
                    return ""
                else:
                    print(f"Hash check passed for {file_name}.")
    return engine_dir

if __name__ == "__main__":
    download_engine("stable-diffusion", hash_check=False)