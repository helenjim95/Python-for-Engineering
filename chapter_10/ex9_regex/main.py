import re

import numpy as np


def get_props(text) -> list[float]:
    props = []
    s = re.findall(r'property\s*?=\s*?[−+]?\d+.?\d+', text)
    for line in s:
        line = line.replace('property', '').replace("=", '').strip()
        if (line.startswith('−')):
            props.append(float(line[1:]) * -1)
        else:
            props.append(float(line))
    return props


def check_mail(text) -> bool:
    """
    :param text:
    :return:
    """
    regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
    if re.fullmatch(regex, text):
        return True
    return False


def replace_name(text) -> str:
    """
    :param text:
    :return:
    """
    regex = re.compile(r'Mrs?. (\w+) (\w+)')
    return re.sub(regex, r'\2, \1', text)


# TODO: waiting for atom file
def get_atoms(path) -> (list[str], np.array):
    pass


def main():
    text = "blue property = 1.453, red property = −4.58, teal property =−1.678"
    print(get_props(text))
    text_email = "gfh@email.com"
    text_not_email = "gfhemail.com"
    print(check_mail(text_email))
    print(check_mail(text_not_email))
    mr = "Mr. Firstname Surname"
    mrs = "Mrs. Firstname Surname"
    print(replace_name(mr))
    print(replace_name(mrs))


if __name__ == "__main__":
    main()
