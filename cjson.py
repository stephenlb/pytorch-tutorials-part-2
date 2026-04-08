import json
from os import path

class Config:
    def __init__(self, config_name='config.json'):
        self.__config_data_name = config_name
        self.__config_data = self.__load()

    def __load(self):
        if path.exists(self.__config_data_name):
            with open(self.__config_data_name, 'r') as r:
                return json.loads(r.read())
        else:
            with open(self.__config_data_name, 'w') as w:
                w.write(json.dumps({}, indent=4))
            return {}

    def get(self, name):
        self.__load()
        if name in self.__config_data:
            return self.__config_data.get(name)
        else:
            raise Exception(f"json['{name}'] key doesn't exists")

    def __write(self):
        with open(self.__config_data_name, 'w') as w:
            w.write(json.dumps(self.__config_data, indent=4))

    def update(self, dic):
        result = self.__config_data.update(dic)
        self.__write()
        return result

    def set(self, key, value):
        self.__load()
        self.__config_data[key] = value
        self.__write()

    def keys(self):
        self.__load()
        return self.__config_data.keys()

    def values(self):
        self.__load()
        return self.__config_data.values()

    def delete(self, name):
        self.__load()
        if name not in self.__config_data:
            raise Exception(f"json['{name}'] key doesn't exist")
        else:
            del self.__config_data[name]
            self.__write()

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, item):
        return self.get(item)

    def __delitem__(self, key):
        self.delete(key)

    def __enter__(self):
        self.__load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__config_data != {}:
            return

    def __len__(self):
        return len(self.__config_data)

    def __contains__(self, item):
        self.__load()
        if item in self.__config_data:
            return True
        else:
            return False

    def __repr__(self):
        return str(self.__config_data)

    def __iter__(self):
        self.__load()
        for x in self.__config_data.items():
            yield x

    def __iadd__(self, other: dict):
        self.__load()
        self.__config_data.update(other)
        self.__write()
