import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
import tensorflow as tf


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(dataset, model, model_name="ResNet18"):
    print("Collecting performance..")

    if dataset is None:
        return np.zeros([0, 10]), np.zeros([0])

    if dataset is not None:
        probs = []
        # labels = []

        output = model.predict(dataset)
        if model_name in ["MitB0"]:
            output = output.logits

        prob = tf.nn.softmax(
            output,
            axis=1
        )

        probs.append(prob)
        if model_name not in ["MitB0"]:
            labels = np.concatenate([y for _, y in dataset], axis=0)
        else:
            labels = np.concatenate([np.squeeze(y) for _, y in dataset], axis=0)
        numpy_prob = tf.concat(prob, axis=0).numpy()
        # print(np.shape(labels))
        # print(np.shape(numpy_prob))
        return numpy_prob, labels
    else:
        return None, None


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    # MIA-efficacy: the number of forgetting data examples classified as non-training.

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)
        # target_train --> original train, retrained/resumed non-train
        # train 1
        # non train 0
        # low accuracy means most of the data are predicted to be non-training
        # original: should be high
        # retrained/resumed: should be low

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

        # target_train --> original train, retrained/resumed non-train
        # train 1
        # non train 0
        # low accuracy means most of the data are predicted to be training
        # original: should be low
        # retrained/resumed: should be high

    return np.mean(accs)


def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model, model_name):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model, model_name)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model, model_name)

    target_train_prob, target_train_labels = collect_prob(target_train, model, model_name)
    # target_test_prob, target_test_labels = collect_prob(target_test, model, model_name)

    shadow_train_prob = torch.from_numpy(shadow_train_prob)
    shadow_train_labels = torch.from_numpy(shadow_train_labels)
    shadow_test_prob = torch.from_numpy(shadow_test_prob)
    shadow_test_labels = torch.from_numpy(shadow_test_labels)
    target_train_prob = torch.from_numpy(target_train_prob)
    target_train_labels = torch.from_numpy(target_train_labels)

    # target_test_prob = torch.from_numpy(target_test_prob)
    # target_test_labels = torch.from_numpy(target_test_labels)

    # shadow_train_corr = (
    #     torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    # ).int()
    # shadow_test_corr = (
    #     torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    # ).int()
    # target_train_corr = (
    #         torch.argmax(target_train_prob, axis=1) == target_train_labels
    #     ).int()
    # target_test_corr = (
    #     torch.argmax(target_test_prob, axis=1) == target_test_labels
    # ).int()

    # shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    # shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    # target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    # target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    acc_conf = SVC_fit_predict_segmentation(
        shadow_train_prob, shadow_train_labels,
        shadow_test_prob, shadow_test_labels,
        target_train_prob, target_train_labels,
        None, None,
    )
    return acc_conf

def extract_true_label_confidences(segmentation_prob, segmentation_labels, mode="mean"):
    """
    Extracts confidence scores for the true labels in a segmentation task.
    Args:
        segmentation_prob (torch.Tensor): Shape (N, C, H, W) - Predicted probabilities.
        segmentation_labels (torch.Tensor): Shape (N, H, W) - True labels (integer class indices).
    Returns:
        torch.Tensor: Shape (N,) - Aggregated confidence scores for each image.
    """
    # Ensure true labels have a channel dimension for indexing
    # segmentation_labels = segmentation_labels.unsqueeze(1)  # Shape: (N, 1, H, W)

    # Gather the confidence for the true labels
    true_label_confidences = torch.gather(segmentation_prob, 1, segmentation_labels)  # Shape: (N, 1, H, W)

    # Remove the channel dimension (now pixel-level confidences)
    # true_label_confidences = true_label_confidences.squeeze(1)  # Shape: (N, H, W)
    # print(true_label_confidences)

    if mode == "mean":
        # Aggregate confidence scores across all pixels for each image
        # aggregated_confidences = true_label_confidences.view(true_label_confidences.size(0), -1).mean(dim=1)  # Shape: (N,)

        aggregated_confidences = torch.mean(true_label_confidences, dim=(1,2,3))
        # print("mean ", aggregated_confidences.shape)
        # print(aggregated_confidences)
    return aggregated_confidences


def SVC_fit_predict_segmentation(shadow_train_prob, shadow_train_labels,
                                 shadow_test_prob, shadow_test_labels,
                                 target_train_prob=None, target_train_labels=None,
                                 target_test_prob=None, target_test_labels=None,
                                 aggregation_mode="mean"):
    """
    Perform membership inference attack on segmentation tasks using confidence scores.
    """
    print("Train shadow model")
    # Extract features (confidence scores for true labels, aggregated)
    shadow_train_features = extract_true_label_confidences(shadow_train_prob, shadow_train_labels, mode=aggregation_mode)
    shadow_test_features = extract_true_label_confidences(shadow_test_prob, shadow_test_labels, mode=aggregation_mode)
    target_train_features = extract_true_label_confidences(target_train_prob, target_train_labels, mode=aggregation_mode) if target_train_prob is not None else None
    target_test_features = extract_true_label_confidences(target_test_prob, target_test_labels, mode=aggregation_mode) if target_test_prob is not None else None

    # Labels for shadow data
    Y_shadow = torch.cat([torch.ones(len(shadow_train_features)), torch.zeros(len(shadow_test_features))])
    X_shadow = torch.cat([shadow_train_features, shadow_test_features]).unsqueeze(1)

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow.cpu().numpy(), Y_shadow.cpu().numpy())

    # Evaluate on target data
    accs = []
    if target_train_features is not None:
        acc_train = clf.predict(target_train_features.unsqueeze(1).cpu().numpy()).mean()
        accs.append(acc_train)
    if target_test_features is not None:
        acc_test = 1 - clf.predict(target_test_features.unsqueeze(1).cpu().numpy()).mean()
        accs.append(acc_test)

    return sum(accs) / len(accs) if accs else None


def JSDiv(p, q):
    p = torch.from_numpy(p)
    q = torch.from_numpy(q)
    m = (p + q) / 2
    js_div_value = 0.5 * F.kl_div(torch.log(p), m) + 0.5 * F.kl_div(torch.log(q), m)
    return js_div_value.item()


# ZRF/UnLearningScore
def UnLearningScore(tmodel, gold_model, forget_dl):
    # model_preds = []
    # gold_model_preds = []
    # with torch.no_grad():
    #     for batch in forget_dl:
    #         x, y, cy = batch
    #         x = x.to(device)
    #         model_output = tmodel(x)
    #         gold_model_output = gold_model(x)
    #         model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
    #         gold_model_preds.append(F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds, _ = collect_prob(dataset=forget_dl, model=tmodel)
    gold_model_preds, _ = collect_prob(dataset=forget_dl, model=gold_model)

    # model_preds = torch.cat(model_preds, axis=0)
    # gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)

