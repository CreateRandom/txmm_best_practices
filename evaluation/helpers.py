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

def print_examples(val_text, selected_examples):
    ind = 0
    while ind < len(selected_examples):
        input_text = input('Print? ')
        if input_text == 'n':
            break
        text_index = selected_examples[ind]
        print(val_text.iloc[text_index])
        ind += 1