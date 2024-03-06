import os
import yaml

__CONF_DIR__ = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(__CONF_DIR__, "conf.yaml"), "r", encoding="utf-8") as f:
    _CONF = yaml.safe_load(f)

# APP Configurations.
CONFIG_VERSION_NAME = _CONF.get("Config").get("Version")
CONFIG_VERSION_DEFAULT = _CONF.get("Config").get("Default")
CONFIG_DATETIME_FORMAT = _CONF.get("Config").get("DatetimeFormat")

# Server settings
AUTOSCALER_HOST = _CONF.get('Server').get('Host')
AUTOSCALER_PORT = _CONF.get('Server').get('Port')

# Autoscaler in-memory db
AUTOSCALER_VALUE_CACHE_SIZE = _CONF.get('DB').get('CacheValueSize')

# Autoscaler on-disk db
AUTOSCALER_DB_SQLITE_NAME = _CONF.get('DB').get('SqliteName')
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
AUTOSCALER_DB_SQLITE_PATH = os.path.join(DATA_DIR, AUTOSCALER_DB_SQLITE_NAME)