from datetime import datetime


def generate_version(method="second"):
    """
    Returns version based on second, minute, hour, day, month and year.
    Used to generate ML model versions and MLFlow experiment versions
    """
    if method == "year":
        version = f"{datetime.now().year}"
    elif method == "month":
        version = f"{datetime.now().year}_{datetime.now().month}"
    elif method == "day":
        version = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}"
    elif method == "hour":
        version = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}_" + \
                  f"{datetime.now().hour}"
    elif method == "minute":
        version = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}_" + \
                  f"{datetime.now().hour}_{datetime.now().minute}"
    else:
        version = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}_" + \
                  f"{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}"
    return version
