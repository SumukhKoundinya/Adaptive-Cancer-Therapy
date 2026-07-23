def encode_response(x):
    if str(x).upper() in ["CR", "PR", "SD"]:
        return 1
    if str(x).upper() in ["PD"]:
        return 0
    return None


def encode_yesno(x):
    if str(x).lower() in ["yes", "1", "true"]:
        return 1
    if str(x).lower() in ["no", "0", "false"]:
        return 0
    return None