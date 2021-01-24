import os
import yaml
import re
import copy

loader = yaml.FullLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

class Config(object):
    def __init__(self, dict_config=None):
        super().__init__()
        self.set_attribute(dict_config)

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as stream:
            return Config(yaml.load(stream, Loader=loader))

    @staticmethod
    def from_dict(dict):
        return Config(dict)

    @staticmethod
    def get_empty():
        return Config()

    def __getattr__(self, item):
        return self.__dict__[item]

    def __setattr__(self, key, value):
        self.set_attribute({key:value})

    def set_attribute(self, dict_config):
        if dict_config is None:
            return

        for key in dict_config.keys():
            if isinstance(dict_config[key], dict):
                self.__dict__[key] = Config(dict_config[key])
            else:
                self.__dict__[key] = dict_config[key]

    def get(self, key, default, possible_none=True):
        """
        same dictionary.get. this method Configruation Not safe!!!
        :param key:
        :param default:
        :param possible_none: if true, None also returned
        :return:
        """
        out = self.__dict__.get(key, default)
        if possible_none is False and out is None:
            return default
        return out

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def copy(self):
        return Config.from_dict(self.extraction_dictionary(self))

    def update(self, dict_config):
        """
        Combines the current configuration with the parameter configuration.
        This operation performs a deepcopy on the parameter configuration.
        Therefore, it is configuration safe.

        :param dict_config:
        :return:
        """
        for key in dict_config.keys():
            if key in self.__dict__.keys():
                if isinstance(dict_config[key], Config):
                    self.__dict__[key].update(dict_config[key])
                else:
                    self.__dict__[key] = copy.deepcopy(dict_config[key])
            else:
                self.__setitem__(key, copy.deepcopy(dict_config[key]))
        return self

    @classmethod
    def extraction_dictionary(cls, config):
        out = {}
        for key in config.keys():
            if isinstance(config[key], Config):
                out[key] = cls.extraction_dictionary(config[key])
            else:
                out[key] = copy.deepcopy(config[key])
        return out

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result