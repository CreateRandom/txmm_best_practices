import pandas as pd


def get_column_names(param_dict, frame):
    columns = []
    for name in param_dict.keys():
        old_name = 'param_' + name
        frame.rename(columns={old_name: name}, inplace=True)

        # unpack tuples for plotting
        if frame[name].apply(lambda x: isinstance(x, tuple)).all():
            # always take the last element
            frame[name] = frame[name].apply(lambda x: x[-1])

        # try to force to numeric
        frame[name] = pd.to_numeric(frame[name], errors='ignore')
        columns.append(name)
    return columns