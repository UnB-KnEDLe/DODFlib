import numpy as np
import torch
import utils


def devicefy(lis, device):
    return [i.to(device) for i in lis]


def preprocess_pred_targ(model, dataloader, device):
    """
    Transform predictions and targets from torch.Tensors to lists
    """
    full_pred = []
    full_targ = []
    with torch.no_grad():
        for tup in dataloader:
            sent, tag, word, mask = devicefy(tup, device)
            pred, _ = model.decode(sent, word, mask)
            
            for i in range(len(pred)):
                full_pred.append(pred[i, :mask[i].sum()].tolist())
                full_targ.append(tag[i, :mask[i].sum()].tolist())
    
    return full_pred, full_targ


def IOBES_tags(predictions, tag2idx):
    """
    Transform tags from indices to class name (string var)
    """

    idx2tag = {idx: tag for (tag, idx) in tag2idx.items()}
    IOBES_tags = predictions.copy()
    for tags in IOBES_tags:
        for j, tag in enumerate(tags):
            tags[j] = idx2tag[tag]
    return IOBES_tags


def exact_f1_score(model, dataloader, device, tag2idx):
    n = (model.num_classes-2) // 2
    TP, FP, FN = np.zeros((3, n))

    for tup in dataloader:
        sent, tag, word, mask = devicefy(tup, device)
        pred, _ = model.eval().decode(sent, word, mask)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            predicted_entities = utils.find_iobes_entities(pred[i], tag2idx)
            real_entities = utils.find_iobes_entities(tag[i], tag2idx)
            for entity in predicted_entities:
                if entity in real_entities:
                    TP[(entity[2]//2)-1] += 1
                else:
                    FP[(entity[2]//2)-1] += 1
            for entity in real_entities:
                if entity not in predicted_entities:
                    FN[(entity[2]//2)-1] += 1

    precision = TP/(TP+FP+0.000001)
    recall = TP/(TP+FN+0.000001)
    macro_f1 = 2*(precision*recall)/(precision+recall)
    macro_f1 = [0 if np.isnan(f1) else f1 for f1 in macro_f1]
    occurrences = TP + FN
    averaged_macro_f1 = (macro_f1*occurrences).sum()/occurrences.sum()

    TP = TP.sum()
    FP = FP.sum()
    FN = FN.sum()
    if TP == 0:
        return 0, 0

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    micro_f1 = 2*(precision*recall)/(precision+recall)

    return averaged_macro_f1, micro_f1


def exact_micro_f1_score(model, dataloader, device, tag2idx):
    TP, FP, FN = 0, 0, 0

    for tup in dataloader:
        sent, tag, word, mask = devicefy(tup, device)
        pred, _ = model.eval().decode(sent, word, mask)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            predicted_entities = utils.find_iobes_entities2(pred[i], tag2idx)
            real_entities = utils.find_iobes_entities2(pred[i], tag2idx)
            for entity in predicted_entities:
                if entity in real_entities:
                    TP += 1
                else:
                    FP += 1
            for entity in real_entities:
                if entity not in predicted_entities:
                    FN += 1
    if TP == 0:
      return 0
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*(precision*recall)/(precision+recall)
    return f1


def exact_macro_f1_score(model, dataloader, device):
    # TODO: fix it. `find_entities` was most likely
    # supposed to be `find_iobes_entity`
    n = (model.num_classes-2) // 2
    TP, FP, FN = np.zeros((3, n))

    for tup in dataloader:
        sent, tag, word, mask = devicefy(tup, device)
        pred, _ = model.eval().decode(sent, word, mask)

        batch_size = pred.shape[0]
        for i in range(batch_size):
            predicted_entities = utils.find_entities(pred[i])
            real_entities = utils.find_entities(tag[i])
            for entity in predicted_entities:
                if entity in real_entities:
                    TP[(entity[2]//2)-1] += 1
                else:
                    FP[(entity[2]//2)-1] += 1
            for entity in real_entities:
                if entity not in predicted_entities:
                    FN[(entity[2]//2)-1] += 1

    precision = TP/(TP+FP+0.000001)
    recall = TP/(TP+FN+0.000001)
    f1 = 2*(precision*recall)/(precision+recall)

    occurrences = TP + FN
    return occurrences, f1


