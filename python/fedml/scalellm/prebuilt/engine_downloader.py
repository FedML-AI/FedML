import os
import urllib.request

S3_ROOT = "https://fedml-scalellm.s3.us-west-1.amazonaws.com"
LOCAL_ROOT = "~/.fedml/fedml_scalellm_engine"
ENGINE_DICT = {
    "Llama-2-13b-chat-hf": {
        "intermediate_path": "/Llama-2-13b-chat-hf/trt_engines/int8_kv_cache_weight_only/1-gpu/",
        "files": ["config.json", "llama_float16_tp1_rank0.engine", "model.cache"],
        "hashes": {
                "config.json": "be2a724bf1fbdf34eef09beb0a3c6d5bdb18387d3195391726f35c6d7cc354a5",
                "llama_float16_tp1_rank0.engine": "98a1be04eca94e52e5566461df0f3c72550e90b342b33d3fdf89ed70b774aff3",
                "model.cache": "4523b09813df3cf507d791aea0e99173ea596402b8890875c9a4ee1423650939"
        }
    },
    "MythoMax-L2-13b": {
        "intermediate_path": "/MythoMax-L2-13b/trt_engines/bf16/1-gpu/",
        "files": ["config.json", "llama_bfloat16_tp1_rank0.engine", "model.cache"],
    },
    "Nous-Hermes-13b-bf16": {
        "intermediate_path": "/Nous-Hermes-13b/trt_engines/bf16/1-gpu/",
        "files": ["config.json", "llama_bfloat16_tp1_rank0.engine", "model.cache"],
    },
    "Nous-Hermes-13b-int8-kv": {
        "intermediate_path": "/Nous-Hermes-13b/trt_engines/int8_kv_cache_weight_only/1-gpu/",
        "files": ["config.json", "llama_float16_tp1_rank0.engine", "model.cache"],
    }
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
