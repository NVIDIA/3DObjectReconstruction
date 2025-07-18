from pathlib import Path

def dataclass_to_dict(obj):
    """
    Recursively convert a dataclass object and its nested attributes to a dictionary.
    
    Args:
        obj: A dataclass object or any other Python object
        
    Returns:
        dict: A dictionary representation of the object with all nested objects converted
    """
    if obj is None:
        return {}
    
    # Get the object's dictionary
    if hasattr(obj, '__dict__'):
        result = vars(obj)
    else:
        return obj
    
    # Recursively convert nested objects
    for key, value in result.items():
        if hasattr(value, '__dict__'):
            result[key] = dataclass_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [dataclass_to_dict(item) if hasattr(item, '__dict__') else item for item in value]
        elif isinstance(value, dict):
            result[key] = {k: dataclass_to_dict(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
        elif isinstance(value, Path):
            result[key] = str(value)
    
    return result 