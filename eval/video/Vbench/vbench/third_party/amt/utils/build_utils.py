import importlib
import importlib.util
import os
import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CUR_DIR, "../"))


def base_build_fn(module, cls, params):
    try:
        imported = importlib.import_module(module, package=None)
    except ModuleNotFoundError:
        module_path = os.path.join(CUR_DIR, "../", *module.split(".")) + ".py"
        if not os.path.isfile(module_path):
            raise
        spec_name = module.replace(".", "_").replace("-", "_")
        spec = importlib.util.spec_from_file_location(spec_name, module_path)
        imported = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported)
    return getattr(imported, cls)(**params)


def build_from_cfg(config):
    module, cls = config['name'].rsplit(".", 1)
    params = config.get('params', {})
    return base_build_fn(module, cls, params)
