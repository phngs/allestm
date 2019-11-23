import importlib


def name_to_feature(name):
    """
    Converts a feature name, e.g. continuous.Pssm to a class instance.
    Basically the name is appended to mllib.features to form
    mllib.features.continuous.Pssm, the module is loaded, and an instance created.
    Args:
        name: Name of the module and the class.

    Returns:
        Instance of the loaded class.
    """
    module_name, class_name = name.rsplit(".", maxsplit=1)

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()
