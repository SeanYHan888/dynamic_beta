from importlib import util
from pathlib import Path


def _load_impl():
    module_path = Path(__file__).with_name("training-fsdp.py")
    spec = util.spec_from_file_location("training_fsdp_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()
train = _impl.train


if __name__ == "__main__":
    train()
