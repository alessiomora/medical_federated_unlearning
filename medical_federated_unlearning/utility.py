import tensorflow as tf
import numpy as np
from medical_federated_unlearning.prostate_utility import prostate_utility


def list_clients_to_string(unlearned_cid):
    s = ""
    for u in unlearned_cid:
        if s != "":
            s = s+"_"
        s = s + str(u)
    print(f"--- unlearning cid string {s} -----")
    return s


def get_test_and_train_dataset():
    sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
    # train_samples = np.array([156, 94, 280, 246, 230, 105])

    samples_per_client = []
    for site in range(len(sites)):
        ds = prostate_utility.get_dataset_site(sites[site], 'test')
        samples_per_client.append(ds.cardinality().numpy())
        print(f"Samples per client[{site}]: {samples_per_client[site]}")

    min_samples_test = min(samples_per_client)
    print("Min samples test per client: ", min_samples_test)

    ft = True
    ds_test_sites = []

    for site in range(len(sites)):
        # loading test dataset
        ds = prostate_utility.get_dataset_site(sites[site], 'test')
        ds_test_sites.append(ds.map(prostate_utility.element_norm_fn).batch(32))

        if ft:
            ds_test = ds
            ds_test_balanced = ds.shuffle(128).take(min_samples_test)
            ft = False
        else:
            ds_test = ds_test.concatenate(ds)
            ds_test_balanced = ds_test_balanced.concatenate(ds.take(min_samples_test))

    ds_test_union = ds_test.map(
        prostate_utility.element_norm_fn).batch(32)

    ds_test_mia_balanced = ds_test_balanced.map(
        prostate_utility.element_norm_fn).batch(32)


    ds_train_sites = []
    for site in range(len(sites)):
        # loading test dataset
        ds = prostate_utility.get_dataset_site(sites[site], 'train')
        ds_train_sites.append(ds.map(prostate_utility.element_norm_fn).batch(32))

    return ds_test_sites, ds_train_sites, ds_test_union, ds_test_mia_balanced, samples_per_client, min_samples_test


def get_union_retain_dataset(cid, test_cardinality):
    sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
    n = test_cardinality
    total_samples = 0

    first_time = True
    for i in range(6):
        if i not in cid:
            ds = prostate_utility.get_dataset_site(sites[i], 'train')
            total_samples += ds.cardinality().numpy()

            if first_time:
                ds_retain_song = ds.take(n)
                ds_retain_yeom = ds
                first_time = False
            else:
                ds_retain_song = ds_retain_song.concatenate(ds.take(n))
                ds_retain_yeom = ds_retain_yeom.concatenate(ds)

    ds_retain_song = ds_retain_song.shuffle(total_samples)
    ds_retain_song = ds_retain_song.map(prostate_utility.element_norm_fn)
    ds_retain_song = ds_retain_song.batch(32, drop_remainder=False)
    ds_retain_song = ds_retain_song.cache()
    ds_retain_song = ds_retain_song.prefetch(tf.data.AUTOTUNE)

    ds_retain_yeom = ds_retain_yeom.map(prostate_utility.element_norm_fn)
    ds_retain_yeom = ds_retain_yeom.batch(32, drop_remainder=False)
    ds_retain_yeom = ds_retain_yeom.cache()
    ds_retain_yeom = ds_retain_yeom.prefetch(tf.data.AUTOTUNE)

    return ds_retain_song, ds_retain_yeom


def get_forget_dataset(cid):
    print("CID, ",cid)
    sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']

    samples_per_client = 0

    first_time_ds = True
    for u in cid:
        ds = prostate_utility.get_dataset_site(sites[u], 'train')
        samples_per_client += ds.cardinality().numpy()

        if first_time_ds:
            ds_train_client = ds
            first_time_ds = False
        else:
            ds_train_client = ds_train_client.concatenate(ds)
    # print("AAAAAAAA", ds_train_client)

    ds_train_client = ds_train_client.shuffle(samples_per_client, reshuffle_each_iteration=False)
    ds_train_client = ds_train_client.map(prostate_utility.element_norm_fn).batch(32, drop_remainder=False)
    ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)

    return ds_train_client


def evaluate_model(list_of_test_partitions, test_ds, list_of_train_partitions, train_ds, model):
    print("Evaluating model...")
    i = 0
    mean_loss = 0
    mean_dice_score = 0
    for ds in list_of_test_partitions:
        print(f"[Test] Evaluating client {i}")
        loss, dice_score = model.evaluate(ds, verbose=2)
        mean_loss += mean_loss/len(list_of_test_partitions)

        mean_dice_score += mean_dice_score/len(list_of_test_partitions)
        i += 1

    print("[Test] Average")
    print(f"loss: {mean_loss}, dice_score: {mean_dice_score}")
    print("[Test] Evaluating on concat test data")
    # Compute test accuracy
    loss, dice_score = model.evaluate(test_ds, verbose=2)

    for ds in list_of_train_partitions:
        print(f"[Test] Evaluating client {i}")
        loss, dice_score = model.evaluate(ds, verbose=2)
        mean_loss += mean_loss/len(list_of_test_partitions)

        mean_dice_score += mean_dice_score/len(list_of_test_partitions)
        i += 1

    print("[Forget] Evaluating on concat test data")
    # Compute forget accuracy
    loss, dice_score = model.evaluate(train_ds, verbose=2)

    print("[Train] Computing MIA")



    return