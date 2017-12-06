import re, numpy as np

def format_losses():
    alltxt = ""
    with open("results_graphic_only.txt", "r") as f:
        alltxt = f.readlines()

    loss_values = []
    epoch_values = [1]
    for line in alltxt:
        # print line
        loss_matcher = r'(loss\:\W)(\d*\W\d*)'
        epoch_matcher = r'(Epoch\W)(\d*)'
        loss_match = re.search(loss_matcher, line, re.M|re.I)
        epoch_match = re.search(epoch_matcher, line, re.M|re.I)
        if loss_match:
            loss_values.append(float(loss_match.group(2)))
        if epoch_match:
            epoch_values.append(int(epoch_match.group(2)))

    loss_values = np.array(loss_values)
    for i in loss_values:
        print i


def