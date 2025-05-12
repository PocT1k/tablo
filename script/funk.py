from typing import Any


"""Получить или сдел зн. по умолчанию: (dict, ключ, зн. по умолч.)"""
def dict_get_or_set(dict_data: dict, key: str, default_value: Any) -> Any:
    value = dict_data.get(key)

    if value is None: # Если ключ не найден
        dict_data[key] = default_value
        value = default_value

    return value
