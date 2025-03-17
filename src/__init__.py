import pkgutil
import importlib

def import_all_submodules(package_name):
    package = importlib.import_module(package_name)
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__,
                                                              package_name + "."):
        importlib.import_module(module_name)

# Import everything in sub-folders of src
import_all_submodules(__name__)