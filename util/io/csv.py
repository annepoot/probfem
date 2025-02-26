import pandas as pd

__all__ = ["read_csv_from"]


def read_csv_from(fname, line, **kwargs):
    with open(fname) as f:
        pos = 0
        cur_line = f.readline()
        while not cur_line.startswith(line):
            pos = f.tell()
            cur_line = f.readline()
            if cur_line == "":
                raise EOFError("Line not found!")
        f.seek(pos)
        return pd.read_csv(f, **kwargs)
