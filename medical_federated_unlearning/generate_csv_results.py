
import os

from medical_federated_unlearning import utility
from medical_federated_unlearning.prostate_utility import prostate_utility
from medical_federated_unlearning.prostate_utility.unet import UNet

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
import hydra

import numpy as np

from medical_federated_unlearning.utility import list_clients_to_string
from medical_federated_unlearning.mia_svc import SVC_MIA
from medical_federated_unlearning.utility import get_test_and_train_dataset

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

pd.set_option('display.max_columns', None)



def find_last_checkpoint(dir):
    exist = os.path.exists(dir)
    if not exist:
        return -1
    else:
        filenames = os.listdir(dir)  # get all files' and folders' names in the current directory

    dirnames = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(dir, filename)):  # check whether the current object is a folder or not
            filename = int(filename.replace("R_", ""))
            dirnames.append(filename)
    if not dirnames:
        return -1
    last_round_in_checkpoints = max(dirnames)
    print(f"Last checkpoint found in {dir} is from round {last_round_in_checkpoints}")
    return last_round_in_checkpoints


def find_configs_in_folder(dir):
    exist = os.path.exists(dir)
    if not exist:
        return -1
    else:
        filenames = os.listdir(dir)  # get all files' and folders' names in the current directory

    dirnames = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(dir, filename)):  # check whether the current object is a folder or not
            dirnames.append(filename)

    if not dirnames:
        return -1

    return dirnames


def compute_yeom_mia(model, train_data, forget_data, model_name="ResNet18"):
    print("------ Yeom -------")
    loss_average_train, _ = model.evaluate(train_data)
    print(f"Average loss on train data: {loss_average_train}")
    total_member_count = 0
    total_size = 0
    for step, batch in enumerate(forget_data):
        x_batch, y_batch = batch
        y_pred = model.predict(x_batch, verbose=0)
        if model_name in ["MitB0"]:
            y_pred = y_pred.logits
        if model_name in ["UNET"]:
            # Calculate the loss
            ce_loss = prostate_utility.JointLoss(reduction="none")
        else:
            ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction="none")
        loss_forget = ce_loss(y_batch, y_pred)

        member_count = tf.reduce_sum(tf.cast(loss_forget <= loss_average_train, tf.int32))
        # print(member_count)
        total_member_count = total_member_count + member_count
        # print(loss_forget)
        total_size = total_size + tf.shape(loss_forget)[0]

    # print(total_member_count)
    # print(total_size)
    yeom_mia = (total_member_count/total_size) *100
    print(f"Yeom MIA success rate: {yeom_mia}")
    return yeom_mia.numpy()



@hydra.main(config_path="conf", config_name="generate_tables", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']

    #  build base config
    dataset = cfg.dataset
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    model_string = cfg.model
    seed  = cfg.seed
    algorithm = cfg.algorithm
    filter_string = cfg.filter_string
    natural = False
    result_folder = "results_csv"

    config_dir = os.path.join(f"{dataset}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )

    ds_test_sites, ds_train_sites, ds_test_union, ds_test_mia_balanced, samples_per_client_test, min_samples_test = utility.get_test_and_train_dataset()
    # test_cardinality = np.sum(samples_per_client)
    # print(f"Test cardinality: {test_cardinality}")

    entry_list = []
    entry_list_bc = []

    # for cid in [[0], [1], [2], [3], [4], [5]]:
    for cid in [[0], [1], [2], [3], [4]]:
        print(f"---------------------------- Client {cid} ----------------------------")
        unlearned_cid_string = list_clients_to_string(cid)

        ds_retain, ds_retain_yeom = utility.get_union_retain_dataset(cid, min_samples_test)

        client_train_ds = utility.get_forget_dataset(cid)
        forgetting_ds = client_train_ds

        original_computed = False

        for what in ["original", "retrained", algorithm]:
            print(f"What: {what}")
            entry_pd = {}
            if what in ["original", "retrained"]:
                what_string = "" if what == "original" else "_retrained"
                client_dir = os.path.join("checkpoints") if what == "original" else os.path.join(f"client{unlearned_cid_string}", "checkpoints")
                exist = os.path.exists(os.path.join("model_checkpoints" + what_string, config_dir, client_dir))
                print(exist)
                if exist:
                    if what in ["retrained"] or not original_computed:
                        last_checkpoint_retrained = find_last_checkpoint(os.path.join("model_checkpoints"+what_string, config_dir, client_dir))

                        model_checkpoint_dir = os.path.join("model_checkpoints"+what_string, config_dir,
                                                            client_dir,
                                                            f"R_{last_checkpoint_retrained}")

                        print(f"-- {what} last saved round: {last_checkpoint_retrained} --")

                        model = UNet(output_channels=2)
                        model.build((None, 3, 384, 384))

                        model.load_weights(model_checkpoint_dir)

                        optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

                        ce_dice_loss = prostate_utility.JointLoss()
                        dice_coeff = prostate_utility.dice_metric
                        model.compile(
                            optimizer=optimizer,
                            loss=ce_dice_loss,
                            metrics=[dice_coeff]
                        )
                        # --- MIA Yeom et al.
                        yeom_mia = compute_yeom_mia(model, train_data=ds_retain_yeom,
                                         forget_data=forgetting_ds, model_name=model_string)

                        # --- MIA Efficiency
                        results_mia = SVC_MIA(shadow_train=ds_retain,
                                              shadow_test=ds_test_mia_balanced,
                                              target_train=forgetting_ds,
                                              target_test=None,
                                              model=model,
                                              model_name=model_string)

                        union_test_loss, union_test_acc = model.evaluate(ds_test_union)
                        union_test_acc = union_test_acc * 100
                        train_loss, train_acc = model.evaluate(forgetting_ds)
                        train_acc = train_acc * 100

                        print("Per-site Test Evaluation")
                        j = 0
                        average_loss = 0.0
                        average_acc = 0.0

                        for ds in ds_test_sites:
                            print(f"[{sites[j]}]")
                            loss, acc = model.evaluate(ds)
                            acc = acc * 100
                            average_loss = average_loss + loss*samples_per_client_test[j]
                            average_acc = average_acc + acc*samples_per_client_test[j]
                            j += 1
                        average_test_loss = average_loss / np.sum(samples_per_client_test)
                        average_test_acc = average_acc / np.sum(samples_per_client_test)

                        # print("Per-site Train Evaluation")
                        # j = 0
                        # for ds in ds_train_sites:
                        #     print(f"[{sites[j]}]")
                        #     loss, acc = model.evaluate(ds)
                        #     acc = acc * 100
                        #     j += 1

                        # ua = 100.0 - train_acc
                        mia = results_mia * 100
                        print("mia: ", mia)

                        entry_pd["mia"] = mia
                        entry_pd["yeom_mia"] = yeom_mia
                        entry_pd["name"] = f"{what}_client{unlearned_cid_string}"
                        entry_pd["cid"] = unlearned_cid_string
                        entry_pd["union_test_acc"] = union_test_acc
                        entry_pd["union_test_loss"] = union_test_loss
                        entry_pd["average_test_acc"] = average_test_acc
                        entry_pd["average_test_loss"] = average_test_loss
                        entry_pd["train_acc"] = train_acc
                        entry_pd["train_loss"] = train_loss

                        # entry_pd["prediction_overlap"] = prediction_overlap.numpy()
                        # entry_pd["kl_div"] = kl_div.numpy()
                        entry_pd["algorithm"] = what

                        if what in ["retrained"]:
                            retrained_union_test_acc = union_test_acc
                            retrained_union_test_loss = union_test_loss
                            retrained_average_test_acc = average_test_acc
                            retrained_average_test_loss = average_test_loss
                            retrained_train_acc = train_acc
                            retrained_train_loss = train_loss
                            retrained_mia = mia
                            retrained_yeom_mia = yeom_mia
                        if what in ["original"]:
                            original_computed = True

                        print(entry_pd)
                        entry_list.append(entry_pd)
            # elif what in ["natural"]:
            #     substrings = [""]
            #     rounds_to_check = [1, 5, 10, 12]
            #     # rounds_to_check = [1, ]
            #     model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
            #                                         config_dir,
            #                                         what)
            #
            #     list_configs = find_configs_in_folder(model_checkpoint_dir)
            #     print(list_configs)
            #     print(model_checkpoint_dir)
            #     for unlearning_config in list_configs:
            #         # if rounds_to_check in unlearning_config:
            #         if True:
            #             model_checkpoint_base_dir = os.path.join(model_checkpoint_dir,
            #                                                      unlearning_config,
            #                                                      f"client{unlearned_cid_string}",
            #                                                      "checkpoints")
            #
            #             exist = os.path.exists(model_checkpoint_base_dir)
            #             if exist:
            #                 print(f"-- Resumed config {unlearning_config} client {unlearned_cid_string}--")
            #                 for round_recovery in rounds_to_check:
            #                     entry_pd = {}
            #                     r = last_checkpoint_retrained + round_recovery
            #                     print(
            #                         f"-- Resumed config {unlearning_config} client {unlearned_cid_string} round {r}--")
            #                     model_checkpoint_round_dir = os.path.join(
            #                         model_checkpoint_base_dir,
            #                         f"R_{r}")
            #                     exist = os.path.exists(model_checkpoint_round_dir)
            #                     if not exist:
            #                         break
            #
            #                     model = create_model(dataset=dataset,
            #                                          total_classes=total_classes)
            #                     model.load_weights(model_checkpoint_round_dir)
            #
            #                     model.compile(optimizer='sgd',
            #                                   loss=tf.keras.losses.SparseCategoricalCrossentropy(
            #                                       from_logits=True),
            #                                   metrics=['accuracy'])
            #
            #                     _, test_acc = model.evaluate(ds_test_batched)
            #                     test_acc = test_acc * 100
            #
            #                     # --- MIA Yeom et al.
            #                     # yeom_mia = compute_yeom_mia(model,
            #                     #                             train_data=ds_retain_yeom,
            #                     #                             forget_data=forgetting_ds,
            #                     #                             model_name=model_string)
            #
            #                     # --- MIA Efficiency
            #                     results_mia = SVC_MIA(shadow_train=ds_retain,
            #                                           shadow_test=ds_test,
            #                                           target_train=forgetting_ds,
            #                                           target_test=None,
            #                                           model=model,
            #                                           model_name=model_string)
            #
            #                     _, train_acc = model.evaluate(client_train_ds)
            #                     train_acc = train_acc * 100
            #                     ua = 100.0 - train_acc
            #                     mia = results_mia["confidence"] * 100
            #
            #                     # overlap
            #                     # logit_u = model.predict(client_train_ds)
            #                     #
            #                     # prediction_overlap = compute_overlap_predictions(
            #                     #     logit_1=logit_retrained,
            #                     #     logit_2=logit_u
            #                     # )
            #
            #                     # kl_div = compute_kl_div(logit_u, total_classes)
            #
            #                     print(
            #                         f"Test acc: {test_acc} -- Train acc: {train_acc} -- ua {ua} -- mia: {mia}")
            #                     entry_pd["algorithm"] = what
            #                     entry_pd["round_recovery"] = round_recovery
            #                     entry_pd["name"] = unlearning_config
            #                     entry_pd["test_acc"] = test_acc
            #                     entry_pd["train_acc"] = train_acc
            #                     entry_pd["ua"] = ua
            #                     entry_pd["mia"] = mia
            #                     # entry_pd["yeom_mia"] = yeom_mia
            #                     entry_pd["cid"] = unlearned_cid_string
            #                     # entry_pd[
            #                     #     "prediction_overlap"] = prediction_overlap.numpy()
            #                     # entry_pd["kl_div"] = kl_div.numpy()
            #                     # delta with retrained model
            #                     entry_pd[
            #                         "delta_test_acc"] = test_acc - retrained_test_acc
            #                     entry_pd[
            #                         "delta_train_acc"] = train_acc - retrained_train_acc
            #                     entry_pd["delta_mia"] = mia - retrained_mia
            #                     # entry_pd[
            #                     #     "delta_yeom_mia"] = yeom_mia - retrained_yeom_mia
            #                     print(entry_pd)
            #                     entry_list.append(entry_pd)
            #
            #
            else:  # resumed
                model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                                config_dir,
                                                                what)

                list_configs = find_configs_in_folder(model_checkpoint_dir)

                print(list_configs)
                print(model_checkpoint_dir)
                for unlearning_config in list_configs:
                    if filter_string in unlearning_config:
                        model_checkpoint_base_dir = os.path.join(model_checkpoint_dir,
                                                                 unlearning_config,
                                                                 f"client{unlearned_cid_string}",
                                                                 "checkpoints")

                        exist = os.path.exists(model_checkpoint_base_dir)
                        if exist:
                            print(f"-- Resumed config {unlearning_config} client {unlearned_cid_string}--")
                            found = False
                            r_recovery = find_last_checkpoint(model_checkpoint_base_dir)
                            # for round_recovery in range(1, rounds_recovery+1):
                            for round_recovery in range(r_recovery, r_recovery+1):
                                if not found:
                                    entry_pd = {}
                                    # r = last_checkpoint_retrained + round_recovery
                                    r = round_recovery
                                    print(f"-- Resumed config {unlearning_config} client {unlearned_cid_string} round {r}--")
                                    model_checkpoint_round_dir = os.path.join(
                                            model_checkpoint_base_dir,
                                            f"R_{r}")
                                    exist = os.path.exists(model_checkpoint_round_dir)
                                    if not exist:
                                        break

                                    print(
                                        f"-- {what} last saved round: {last_checkpoint_retrained} --")

                                    model = UNet(output_channels=2)
                                    model.build((None, 3, 384, 384))

                                    model.load_weights(model_checkpoint_round_dir)

                                    optimizer = tf.keras.optimizers.SGD(
                                        learning_rate=1.0)

                                    ce_dice_loss = prostate_utility.JointLoss()
                                    dice_coeff = prostate_utility.dice_metric
                                    model.compile(
                                        optimizer=optimizer,
                                        loss=ce_dice_loss,
                                        metrics=[dice_coeff]
                                    )

                                    union_test_loss, union_test_acc = model.evaluate(
                                        ds_test_union)
                                    union_test_acc = union_test_acc * 100

                                    if union_test_acc >= retrained_union_test_acc:
                                        found = True

                                        # --- MIA Yeom et al.
                                        yeom_mia = compute_yeom_mia(model,
                                                                    train_data=ds_retain_yeom,
                                                                    forget_data=forgetting_ds,
                                                                    model_name=model_string)

                                        # --- MIA Efficiency
                                        results_mia = SVC_MIA(shadow_train=ds_retain,
                                                              shadow_test=ds_test_mia_balanced,
                                                              target_train=forgetting_ds,
                                                              target_test=None,
                                                              model=model,
                                                              model_name=model_string)

                                        train_loss, train_acc = model.evaluate(
                                            forgetting_ds)
                                        train_acc = train_acc * 100

                                        print("Per-site Test Evaluation")
                                        j = 0
                                        average_loss = 0.0
                                        average_acc = 0.0

                                        for ds in ds_test_sites:
                                            print(f"[{sites[j]}]")
                                            loss, acc = model.evaluate(ds)
                                            acc = acc * 100
                                            average_loss = average_loss + loss * \
                                                           samples_per_client_test[j]
                                            average_acc = average_acc + acc * \
                                                          samples_per_client_test[j]
                                            j += 1
                                        average_test_loss = average_loss / np.sum(
                                            samples_per_client_test)
                                        average_test_acc = average_acc / np.sum(
                                            samples_per_client_test)

                                        # print("Per-site Train Evaluation")
                                        # j = 0
                                        # for ds in ds_train_sites:
                                        #     print(f"[{sites[j]}]")
                                        #     loss, acc = model.evaluate(ds)
                                        #     acc = acc * 100
                                        #     j += 1

                                        # ua = 100.0 - train_acc
                                        mia = results_mia * 100
                                        print("mia: ", mia)

                                        entry_pd["algorithm"] = what
                                        entry_pd["round_recovery"] = r - last_checkpoint_retrained
                                        entry_pd["name"] = unlearning_config
                                        entry_pd["union_test_acc"] = union_test_acc
                                        entry_pd["union_test_loss"] = union_test_loss
                                        entry_pd["average_test_acc"] = average_test_acc
                                        entry_pd["average_test_loss"] = average_test_loss
                                        entry_pd["train_acc"] = train_acc
                                        entry_pd["train_loss"] = train_loss

                                        entry_pd["mia"] = mia
                                        entry_pd["yeom_mia"] = yeom_mia

                                        entry_pd["cid"] = unlearned_cid_string

                                        # delta with retrained model
                                        entry_pd["delta_union_test_acc"] = union_test_acc - retrained_union_test_acc
                                        entry_pd["delta_union_test_loss"] = union_test_loss - retrained_union_test_loss
                                        entry_pd["delta_average_test_acc"] = average_test_acc - retrained_average_test_acc
                                        entry_pd["delta_average_test_loss"] = average_test_loss - retrained_average_test_loss
                                        entry_pd["delta_train_acc"] = train_acc - retrained_train_acc
                                        entry_pd["delta_train_loss"] = train_loss - retrained_train_loss

                                        entry_pd["delta_train_acc"] = train_acc - retrained_train_acc
                                        entry_pd["delta_mia"] = mia - retrained_mia
                                        entry_pd["delta_yeom_mia"] = mia - retrained_yeom_mia

                                        print(entry_pd)
                                        entry_list.append(entry_pd)

    # if natural:
    if False:
        df = pd.DataFrame(entry_list)
        print(df)
        filename = f'results_unlearning_natural.csv'
        path_to_save = os.path.join(result_folder, dataset)
        exist = os.path.exists(path_to_save)
        if not exist:
            os.makedirs(path_to_save)
        df.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)
        # Group by two columns and calculate mean and std for other columns
        result = df.groupby(["algorithm", "name", "round_recovery"]).agg(
            delta_train_acc_mean=("delta_train_acc", lambda x: x.abs().mean()),
            delta_train_acc_std=("delta_train_acc", lambda x: x.abs().std()),
            delta_test_acc_mean=("delta_test_acc", lambda x: x.abs().mean()),
            delta_test_acc_std=("delta_test_acc", lambda x: x.abs().std()),
            delta_mia_mean=("delta_mia", lambda x: x.abs().mean()),
            delta_mia_std=("delta_mia", lambda x: x.abs().std()),
            # delta_yeom_mia_mean=("delta_yeom_mia", lambda x: x.abs().mean()),
            # delta_yeom_mia_std=("delta_yeom_mia", lambda x: x.abs().std()),
        ).reset_index()
        result = result.round(2)
        print(result)
        filename = f'results_unlearning_natural_aggregated.csv'
        path_to_save = os.path.join(result_folder, dataset)
        exist = os.path.exists(path_to_save)
        if not exist:
            os.makedirs(path_to_save)
        result.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)
    else:
        # df_bc = pd.DataFrame(entry_list_bc)
        # print(df_bc)
        # filename = f'results_unlearning_before_recovery.csv'
        # path_to_save = os.path.join(result_folder, dataset)
        # exist = os.path.exists(path_to_save)
        # if not exist:
        #     os.makedirs(path_to_save)
        # df_bc.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)
        # # Group by two columns and calculate mean and std for other columns
        # result = df_bc.groupby(["algorithm", "name"]).agg(
        #     delta_train_acc_mean=("delta_train_acc", lambda x: x.abs().mean()),
        #     delta_train_acc_std=("delta_train_acc", lambda x: x.abs().std()),
        #     delta_test_acc_mean=("delta_test_acc", lambda x: x.abs().mean()),
        #     delta_test_acc_std=("delta_test_acc", lambda x: x.abs().std()),
        #     delta_mia_mean=("delta_mia", lambda x: x.abs().mean()),
        #     delta_mia_std=("delta_mia", lambda x: x.abs().std()),
        #     delta_yeom_mia_mean=("delta_yeom_mia", lambda x: x.abs().mean()),
        #     delta_yeom_mia_std=("delta_yeom_mia", lambda x: x.abs().std()),
        # ).reset_index()
        # result = result.round(2)
        # print(result)
        # filename = f'results_unlearning_before_recovery_aggregated.csv'
        # path_to_save = os.path.join(result_folder, dataset)
        # exist = os.path.exists(path_to_save)
        # if not exist:
        #     os.makedirs(path_to_save)
        # result.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)
        #
        df = pd.DataFrame(entry_list)
        print(df)
        # re-order columns

        df = df[[ "cid", "algorithm", "name", "round_recovery",
                  "delta_union_test_acc", "delta_union_test_loss",
                  "delta_average_test_acc", "delta_average_test_loss",
                  "delta_train_acc", "delta_train_loss",
                  "mia", "delta_mia",
                  "yeom_mia", "delta_yeom_mia",
                  ]]

        filename = f'results_unlearning_after_recovery.csv'
        path_to_save = os.path.join(result_folder, dataset)
        exist = os.path.exists(path_to_save)
        if not exist:
            os.makedirs(path_to_save)
        df.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)

        # aggregated results
        filtered_df = df[(df["algorithm"] != "original") & (df["algorithm"] != "retrained")]

        # Group by two columns and calculate mean and std for other columns
        result = filtered_df.groupby(["algorithm", "name" ]).agg(
            delta_round_mean=("round_recovery", lambda x: x.abs().mean()),
            delta_round_std=("round_recovery", lambda x: x.abs().std()),
            delta_union_test_acc=("delta_union_test_acc",  lambda x: x.abs().mean()),
            delta_union_test_loss=("delta_union_test_loss", lambda x: x.abs().std()),
            delta_average_test_acc=("delta_average_test_acc",  lambda x: x.abs().mean()),
            delta_average_test_loss=("delta_average_test_loss",  lambda x: x.abs().mean()),
            delta_train_acc=("delta_train_acc",  lambda x: x.abs().mean()),
            delta_train_acc_std=("delta_train_acc",  lambda x: x.abs().std()),
            delta_train_loss=("delta_train_loss",  lambda x: x.abs().mean()),
            delta_train_loss_std=("delta_train_loss",  lambda x: x.abs().std()),
            delta_mia=("delta_mia",  lambda x: x.abs().mean()),
            delta_mia_std=("delta_mia",  lambda x: x.abs().std()),
            delta_yeom_mia=("delta_yeom_mia",  lambda x: x.abs().mean()),
            delta_yeom_mia_std=("delta_yeom_mia",  lambda x: x.abs().std()),
        ).reset_index()
        result = result.round(2)
        print(result)
        filename = f'results_unlearning_after_recovery_aggregated.csv'
        path_to_save = os.path.join(result_folder, dataset)
        exist = os.path.exists(path_to_save)
        if not exist:
            os.makedirs(path_to_save)
        result.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)


if __name__ == "__main__":
    main()

