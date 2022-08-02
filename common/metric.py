from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def get_f1score_transformer(label, pred):
    return f1_score(label, pred)


def get_acc_transformer(label, pred):
    return accuracy_score(label, pred)


def get_recall_transformer(label, pred):
    return recall_score(label, pred)


def get_precision_transformer(label, pred):
    return precision_score(label, pred)
