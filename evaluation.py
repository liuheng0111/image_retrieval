# coding=utf-8


def confusion_matrix(image_labels, image_predicted):
    """
    混淆矩阵
    :param image_labels: dict, {image_id: [label1, label2,...]}
    :param image_predicted: dict, {image_id: [pred1, pred2,...]}
    :return: 
    """""
    mat = {'TP': 0.0, 'FP': 0.0, 'FN': 0.0, 'TN': 0.0}

    for idx in image_labels.keys():
        labels = image_labels[idx]
        predicted = image_predicted[idx]
        if len(labels) == 0:
            mat['FP'] += len(predicted)
        else:
            for l in labels:
                pred_len = len(predicted)
                if l in predicted:
                    mat['TP'] += 1
                    pred_len -= 1
                else:
                    mat['FN'] += 1
            mat['FP'] += pred_len
    return mat
