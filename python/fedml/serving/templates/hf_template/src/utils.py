from functools import lru_cache, wraps
import os
import threading
from typing import Callable, Type, TypeVar
from typing_extensions import ParamSpec
import uuid

from .integrations import is_jinja2_available
from .typing import PathType

if is_jinja2_available():
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

T = TypeVar("T")
P = ParamSpec("P")

ClassType = Type[T]


def get_real_path(path: PathType) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def is_file(path: PathType) -> bool:
    return os.path.isfile(get_real_path(path))


def is_directory(path: PathType) -> bool:
    return os.path.isdir(get_real_path(path))


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@lru_cache
def is_jinja_template(template: str) -> bool:
    if not is_jinja2_available():
        return False

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception

    try:
        jinja_env.from_string(template)
        return True
    except TemplateError:
        return False


def singleton(replace_init: bool = False) -> Callable[[ClassType], ClassType]:
    """

    Args:
        replace_init: set to `True` to replace `__init__` call. By default, only `__new__` is replaced.
            If set to `True`, repeated call to `__init__` will be ignored.

    Returns:

    """

    def _singleton(input_cls: ClassType) -> ClassType:
        # adapted from https://igeorgiev.eu/python/design-patterns/python-singleton-pattern-decorator/
        cls_new_func = input_cls.__new__
        cls_init_func = input_cls.__init__

        cls_instance = None
        should_init = False
        cls_lock = threading.Lock()

        @wraps(input_cls.__new__)
        def __new__(cls, *args, **kwargs) -> T:
            # see https://stackoverflow.com/a/65575927
            nonlocal cls_instance, should_init

            # see https://medium.com/analytics-vidhya/how-to-create-a-thread-safe-singleton-class-in-python-822e1170a7f6
            with cls_lock:
                if cls_instance is None:
                    cls_instance = cls_new_func(cls)
                    should_init = True
            return cls_instance

        input_cls.__new__ = __new__

        if replace_init:
            @wraps(input_cls.__init__)
            def __init__(self, *args, **kwargs) -> None:
                nonlocal should_init

                with cls_lock:
                    if should_init:
                        cls_init_func(self, *args, **kwargs)
                        should_init = False

            input_cls.__init__ = __init__

        return input_cls

    return _singleton
