version_info = (0, 0, 1)
# format:
# ('major', 'minor', 'patch')


def get_version():
    "Returns the version as a human-format string."
    return '{}.{}.{}'.format(*version_info)


__version__ = get_version()
