import argparse

import fedml
from fedml.api.modules import model


def create_and_push_model(
        model_name, model_config, tag_name,
        api_key=None, config_version="release"):
    fedml.set_env_version(config_version)
    model.create(model_name, model_config=model_config)
    model.push(
        model_name, api_key=api_key,
        tag_names=[tag_name])


if __name__ == "__main__":
    print("Creating model card...\n")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", "-n", type=str)
    parser.add_argument("--model_config", "-cf", type=str)
    parser.add_argument("--tag_name", "-tn", type=str, default=None)
    parser.add_argument("--api_key", "-k", type=str, default=None)
    parser.add_argument("--version", "-v", type=str, default="release")
    args = parser.parse_args()

    create_and_push_model(
        args.model_name, args.model_config, args.tag_name,
        api_key=args.api_key, config_version=args.version
    )

