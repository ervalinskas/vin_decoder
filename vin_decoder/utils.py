import json
from typing import Dict


def save_txt(filepath: str, file_body: str) -> None:
    """Save a text file.

    Args:
        filepath (str): location of file.
        file_body (str): data to save.
    """
    with open(filepath, "w") as fp:
        fp.write(file_body)
        fp.write("\n")


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): location of file.
    Returns:
        Dict: loaded JSON data.
    """
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): data to save.
        filepath (str): location of where to save the data.
        cls (optional): encoder to use on dict data. Defaults to None.
        sortkeys (bool, optional): whether to sort keys alphabetically. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")
