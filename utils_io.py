import numpy as np


def read_lines_file(file_path, sep=" ", merge=False):
    output = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if sep is None:
                output += [line]
            else:
                if merge:
                    output.extend(line.split(sep))
                else:
                    output.append(line.split(sep))
    return output


def read_dict_file(file_path, sep_key=" ", sep_val=",", one2many=True):
    output = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            key, values = line.strip().split(sep_key)
            output[key] = values.split(sep_val)
    return output


def read_trials_file(file_path, sep=" "):
    trials = []
    labels = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(sep)
            enr, tst, label = parts[0], parts[1], parts[2]
            trials += [(enr, tst)]
            labels += [label]
    return trials, np.array(labels)


def write_dict_file(file_path, dictionary, sep_key=" ", sep_val=" ", one2many=False):
    with open(file_path, "w") as f:
        for (key, values) in dictionary.items():
            if one2many:
                assert isinstance(values, (list, tuple))
                values_str = sep_val.join([str(v) for v in values])
            else:
                values_str = values
            f.write(f"{key}{sep_key}{values_str}\n")
