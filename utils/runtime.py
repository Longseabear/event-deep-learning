from importlib import import_module

def get_instance_from_name(module_path, class_name, *args, **kwargs):
    m = import_module(module_path, class_name)
    return getattr(m, class_name)(*args, **kwargs)

def get_class_object_from_name(module_path, class_name):

    m = import_module(module_path, class_name)
    return getattr(m, class_name)
