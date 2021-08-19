import json


def extract_validation_entries(path, kw="validation"):
    """
    Load json file of `stats`.
    """
    print(f"Reading json file {path}")
    validation = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().replace("'", '"')
            line = line.replace("nan", '"nan"')
            try:
                data = json.loads(line)
            except:
                print(line)
                raise
            if data["_meta"]["type"] == kw:
                validation.append(data)
    return validation