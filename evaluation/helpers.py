import numpy as np

def get_classification_status(y_true, y_pred):
    states = []
    # make pairs
    pairs = list(zip(y_true, y_pred))

    for pair in pairs:
        if pair[0] == 0:
            if pair[1] == 0:
                states.append('TN')
            else:
                states.append('FP')
        else:
            if pair[1] == 0:
                states.append('FN')
            else:
                states.append('TP')

    return np.array(states)