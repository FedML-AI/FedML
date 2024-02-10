from typing import Callable
import os
import functools

def debug(_func: Callable=None, *, output_file="output.txt"):

    def decorator(func: Callable):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"\033[93m{k}:\033[0m{v}" for k, v in kwargs.items()]
            signature = "\n".join([*args_repr, *kwargs_repr])
            callable_func = f"{func.__name__}({signature})"

            home_directory = os.path.expanduser("~")
            file_path = os.path.join(home_directory, output_file)
            file_mode = "a" if os.path.exists(file_path) else "w+"

            with open(file_path, file_mode) as f:
                if file_mode == "w":
                    print(f"\033[92mFile {file_path} created.\033[0m")
                print(f"\033[94mDEBUG Started for the following func:\n\033[0m{callable_func}\n", file=f)
                print(f"\033[94mCalling {callable_func}\n\033[0m", file=f)

            value = func(*args, **kwargs)

            with open(file_path, "a") as f:
                print(f"\033[94mDEBUG ENDED: {func.__name__} returned {value!r} \n\n\n\033[0m", file=f)

            return value

        return wrapper

    if _func is None:
        return decorator

    return decorator(_func)
