from peft.import_utils import is_bnb_available

try:
    from peft.import_utils import is_bnb_4bit_available
except ImportError:
    def is_bnb_4bit_available():
        if not is_bnb_available():
            return False

        import bitsandbytes as bnb

        return hasattr(bnb.nn, "Linear4bit")
