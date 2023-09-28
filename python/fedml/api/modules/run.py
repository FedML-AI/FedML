from fedml.api.modules.utils import authenticate


def command(commands, version, api_key):
    authenticate(api_key, version)
