import os

__all__ = ["unique_filename"]


def unique_filename(filepath):
    if not os.path.exists(filepath):
        return filepath

    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        new_filename = f"{name}{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1
