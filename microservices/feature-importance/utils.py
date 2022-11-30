"""Utility functions"""
from collections.abc import Iterable

def sort_dictionary_by_value(dict_input: dict) -> dict:

    # Validate input
    if not isinstance(dict_input, dict):
        raise TypeError("--dict_input-- must be dict type.")
    
    iterables = [isinstance(val, Iterable) for val in dict_input.values()]
    if not all(iterables):
        raise TypeError("--dict_input-- values can't be iterables.")

    # Sorting the input
    sorted_dict = {
        key: val for key, val in sorted(
            dict_input.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    return sorted_dict