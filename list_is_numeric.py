def list_is_numeric(l: list) -> bool:
    """
    Check if a given list only contains numeric string, None or and empty string.
    """
    for v in l:
        try:
            if v != None and v != '':
                float(v)
        except:
            return False
    return True
