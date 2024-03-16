import os
import yaml

__CONF_DIR__ = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(__CONF_DIR__, "config.yaml"), "r", encoding="utf-8") as f:
    _CONF = yaml.safe_load(f)

CONFIG_TIMESERIES_LENGTH = _CONF.get("Config").get("TimeSeriesLength")
CONFIG_DATETIME_FORMAT = _CONF.get("Config").get("DatetimeFormat")
