import yaml
import pickle as pkl

def load_yaml(path: str) -> dict:
    """
    Load a YAML file and return the parsed Python object.
    
    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    
    Returns
    -------
    Any
        Parsed content of the YAML file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_pickle(path):
    """
    Load a pickle file and return the deserialized Python object.

    Parameters
    ----------
    path : str or Path
        Path to the pickle file.

    Returns
    -------
    Any
        The unpickled Python object.
    """
    path = Path(path)
    with path.open("rb") as f:
        return pkl.load(f)


def pretty_print(summary): ## TOOD -- should this be generalized or only for run summaries? Should this be here or do we need a logging module?
    return ''