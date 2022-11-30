"""Utility functions"""

def sort_dictionary_by_value(dict_input: dict) -> dict:
    """
    It sorts a dictionary by value.
    
    Args:
      dict_input (dict): dict
    
    Returns:
      A dictionary with the same keys as the input dictionary, but with the values sorted in descending
    order.
    """

    # Validate input
    if not isinstance(dict_input, dict):
        raise TypeError("--dict_input-- must be dict type.")
    
    valid_types = [isinstance(val, (str, int, float)) for val in dict_input.values()]
    if not all(valid_types):
        raise TypeError("--dict_input-- values must be strings, integers or floats.") 

    first_type = type(list(dict_input.values())[0])
    same_types = [isinstance(val, first_type) for val in dict_input.values()]
    if not all(same_types):
        raise TypeError("--dict_input-- values must be of the same type.")

    # Sorting the input
    sorted_dict = {
        key: val for key, val in sorted(
            dict_input.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    }

    return sorted_dict