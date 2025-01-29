import warnings
import os
import shutil
import hydra
import numpy as np

# from medical_federated_unlearning.generate_csv_results import compute_yeom_mia
# from medical_federated_unlearning.mia_svc import SVC_MIA

import logging, os

from medical_federated_unlearning.prostate_utility import prostate_utility
from medical_federated_unlearning.prostate_utility.unet import UNet
from medical_federated_unlearning.utility import list_clients_to_string

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

SAVE_ROUND_CLIENTS = 500


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


def client_update(cid, local_model, learning_rate, sites, local_batch_size, model_checkpoint_dir, t=1, verbose=2, epochs=1, retraining=False):
    # Local training
    print(f"[Client {cid}] Local Training ---")
    ds_train_client = prostate_utility.get_dataset_site(sites[cid], 'train')
    ds_train_client = ds_train_client \
            .map(prostate_utility.element_norm_fn) \
            .map(prostate_utility.rotate_flip_fn) \
            .shuffle(buffer_size=1024, seed=2) \
            .batch(local_batch_size)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                    weight_decay=1e-4,
                                    epsilon=1e-08)
    # compile clients' model
    ce_dice_loss = prostate_utility.JointLoss()
    dice_coeff = prostate_utility.dice_metric

    local_model.compile(opt,
                             loss=ce_dice_loss,
                             metrics=[dice_coeff]
                             )


    # Local training
    print(f"[Client {cid}] Local training..")
    local_model.fit(
            ds_train_client,
            epochs=epochs,
            verbose=verbose
        )

    # for projected ga
    if t == SAVE_ROUND_CLIENTS:
        if not retraining:
            print("Saving client models for PGA")
            location = os.path.join(model_checkpoint_dir, f"client_models_R{t}", f"client{cid}")
            print(f"[Client {cid}] Saving model checkpoint at {location}")
            exist = os.path.exists(location)
            if not exist:
                os.makedirs(location)

            local_model.save(location)

    return local_model.get_weights()


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    total_rounds = cfg.total_rounds
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    learning_rate = cfg.learning_rate
    resume_training = cfg.resume_training  # resume training after unlearning
    retraining = cfg.retraining  # retrain baseline
    restart_training = cfg.restart_training  # restart training from checkpoint
    seed  = cfg.seed
    dataset = cfg.dataset
    algorithm = cfg.resuming_after_unlearning.algorithm

    resumed_round = 0
    unlearned_cid = list(cfg.unlearned_cid)
    unlearned_cid_string = list_clients_to_string(unlearned_cid)
    first_time = True
    checkpoint_frequency = 1

    # server model
    sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']

    ft = True
    ds_test_sites = []
    for site in range(len(sites)):
        # loading test dataset
        ds = prostate_utility.get_dataset_site(sites[site], 'test').map(
            prostate_utility.element_norm_fn).batch(32)
        ds_test_sites.append(ds)
        if ft:
            ds_test = ds
            ft = False
        else:
            ds_test = ds_test.concatenate(ds)


    # train, test, val dataset partitioned as fedfa (0.6, 0.2, 0.2)
    local_n_examples = np.array([156, 94, 280, 246, 230, 105])

    server_model = UNet(output_channels=2)
    server_model.build((None, 3, 384, 384))
    server_model.summary()

    model_string = "UNET"
    config_dir = os.path.join(f"{dataset}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )

    if resume_training:
        # creating config string for resume training
        algorithm = cfg.resuming_after_unlearning.algorithm
        frozen_layers = cfg.resuming_after_unlearning.frozen_layers
        learning_rate_unlearning = cfg.resuming_after_unlearning.unlearning_lr
        epochs_unlearning = cfg.resuming_after_unlearning.unlearning_epochs

        unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                 config_dir,
                                                 algorithm,
                                                 unlearning_config,
                                                 "client" + unlearned_cid_string)

    elif retraining:
        model_checkpoint_dir = os.path.join("model_checkpoints_retrained", config_dir, "client"+unlearned_cid_string)
    else:
        model_checkpoint_dir = os.path.join("model_checkpoints", config_dir)

    if restart_training:
        if retraining:
            model_checkpoint_base_dir = os.path.join(
                model_checkpoint_dir,
                "checkpoints")
            print(f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir_to_load = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir_to_load)
                resumed_round = last_round
            else:
                print("Checkpoint not found. Start from round 0.")

        elif resume_training:
            if algorithm not in ["natural"]:
                #--- Load the retrained model ----
                client_dir_r = os.path.join(f"client{unlearned_cid_string}", "checkpoints")

                last_checkpoint_retrained = find_last_checkpoint(
                    os.path.join("model_checkpoints_retrained", config_dir, client_dir_r))

                model_checkpoint_dir_retrained = os.path.join("model_checkpoints_retrained",
                                                              config_dir,
                                                              client_dir_r,
                                                              f"R_{last_checkpoint_retrained}")

                model_retrained = UNet(output_channels=2)
                model_retrained.build((None, 3, 384, 384))
                model_retrained.load_weights(model_checkpoint_dir_retrained)

                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                               weight_decay=1e-4,
                                               epsilon=1e-08)
                # compile clients' model
                ce_dice_loss = prostate_utility.JointLoss()
                dice_coeff = prostate_utility.dice_metric

                model_retrained.compile(opt,
                                        loss=ce_dice_loss,
                                        metrics=[dice_coeff]
                                        )

                print("----- Retrained model -----")
                print("Test")
                _, test_acc_retrained = model_retrained.evaluate(ds_test, verbose=2)
                # -------------------------

            if algorithm in ["natural"]:
                print("[Natural baseline] Searching for checkpoints...")
                model_checkpoint_base_dir = os.path.join(model_checkpoint_dir,
                                            "checkpoints")
                last_round = find_last_checkpoint(model_checkpoint_base_dir)
                if last_round > 0:
                    print("[Natural baseline] Found previous checkpoint.")
                    model_checkpoint_dir_to_load = os.path.join(model_checkpoint_base_dir,
                                                        f"R_{last_round}")
                    server_model.load_weights(model_checkpoint_dir_to_load)
                    resumed_round = last_round
                else:
                    print("Checkpoint not found. Start from original model.")
                    model_checkpoint_base_dir = os.path.join("model_checkpoints",
                                                             config_dir,
                                                             "checkpoints")
                    print(
                        f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")
                    last_round = find_last_checkpoint(model_checkpoint_base_dir)
                    if last_round > 0:
                        model_checkpoint_dir_to_load = os.path.join(model_checkpoint_base_dir,
                                                            f"R_{last_round}")
                        server_model.load_weights(model_checkpoint_dir_to_load)
                        resumed_round = last_round
                    else:
                        print("Checkpoint not found. Start from round 0.")


        else:  # continue original training
            model_checkpoint_base_dir = os.path.join(model_checkpoint_dir,
                                                     "checkpoints")
            print(f"[Server] Loading checkpoint at {model_checkpoint_base_dir} ")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round
            else:
                print("Checkpoint not found. Start from round 0.")

    server_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    ce_dice_loss = prostate_utility.JointLoss()
    dice_coeff = prostate_utility.dice_metric
    server_model.compile(
        optimizer=server_optimizer,
        loss=ce_dice_loss,
        metrics=[dice_coeff]
    )

    test_loss, test_acc = server_model.evaluate(ds_test, verbose=2)

    early_stop_recovery = False
    if resume_training:

        if algorithm not in ["natural"] and test_acc >= test_acc_retrained:
            early_stop_recovery = True
            if resumed_round == SAVE_ROUND_CLIENTS:
                if resume_training:
                    if algorithm not in ["pseudo_gradient_ascent_single", "pseudo_gradient_ascent"]:
                        dir = os.path.join(model_checkpoint_dir, "checkpoints")
                        shutil.rmtree(dir, ignore_errors=True)
                        print("Saving checkpoint global model......")
                        server_model.save(
                            os.path.join(model_checkpoint_dir, "checkpoints", f"R_{resumed_round}"))
                        print("[Info] Already recovered. Not performing any round.")
                    else:
                        early_stop_recovery = False

    if not early_stop_recovery:
        # retrained_computed = False
        for r in range(resumed_round + 1, resumed_round + total_rounds + 1):
            delta_w_aggregated = tf.nest.map_structure(lambda a, b: a - b,
                                                       server_model.get_weights(),
                                                       server_model.get_weights())

            if resume_training or retraining:
                m = max(total_clients * active_clients, 1) - len(unlearned_cid)
            else:
                m = max(total_clients * active_clients, 1)

            client_list = list(range(total_clients))
            if resume_training or retraining:
                for u in unlearned_cid:
                    client_list.remove(u)

            print(client_list)
            sampled_clients = np.random.choice(
                np.asarray(client_list, np.int32),
                size=int(m),
                replace=False)

            print(f"[Server] Round {r} -- Selected clients: {sampled_clients}")

            selected_client_examples = local_n_examples[sampled_clients.tolist()]
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)

            total_examples = np.sum(selected_client_examples)

            global_weights = server_model.get_weights()
            for k in sampled_clients:
                print("Client: ", k)
                # local_samples = selected_client_examples[k]
                local_samples = local_n_examples[k]
                print("Local samples: ", local_samples)

                client_model = server_model
                client_model.set_weights(global_weights)
                client_update(k, client_model, learning_rate=learning_rate, sites=sites, local_batch_size=local_batch_size, model_checkpoint_dir=model_checkpoint_dir, t=r, retraining=retraining)

                # FedAvg aggregation
                delta_w_local = tf.nest.map_structure(lambda a, b: a - b,
                                                          client_model.get_weights(),
                                                          global_weights,
                                                          )
                delta_w_aggregated = tf.nest.map_structure(
                        lambda a, b: a + b * (local_samples / total_examples),
                        delta_w_aggregated,
                        delta_w_local)

            # apply the aggregated updates
            # --> sgd with 1.0 lr
            new_global_weights = tf.nest.map_structure(lambda a, b: a + b,
                                                       global_weights,
                                                       delta_w_aggregated)
            server_model.set_weights(new_global_weights)

            # logging global model performance
            test_loss, test_acc = server_model.evaluate(ds_test, verbose=2)
            print(f'[Server] Round {r} -- Test accuracy: {test_acc}')

            print("Saving checkpoint...")
            dir = os.path.join(model_checkpoint_dir, "checkpoints")
            if resume_training:  # need for all the checkpoints for the analysis
                checkpoint_frequency = 1

            if r % checkpoint_frequency == 0:
                if first_time:
                    exist = os.path.exists(dir)
                    if not exist:
                        os.makedirs(dir)
                    else:
                        if not resume_training:
                            shutil.rmtree(dir, ignore_errors=True)
                    first_time = False
                if resume_training and algorithm not in ["natural"]:
                    shutil.rmtree(dir, ignore_errors=True)
                print("Saving checkpoint global model......")
                server_model.save(os.path.join(model_checkpoint_dir, "checkpoints", f"R_{r}"))

            # if retraining or resume_training:
            #     first_time_ds = True
            #     for u in unlearned_cid:
            #         ds = prostate_utility.get_dataset_site(sites[u], 'train')
            #         if first_time_ds:
            #             ds_train_client = ds
            #
            #         ds_train_client = ds_train_client.concatenate(ds)
            #         first_time_ds = False
            #
            #     ds_train_client = ds_train_client.map(prostate_utility.element_norm_fn)
            #     ds_train_client = ds_train_client.batch(local_batch_size,
            #                                                       drop_remainder=False)
            #     ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)
            #
            #     # print("[Retrained] Train")
            #     # _, train_acc_retrained = model_retrained.evaluate(ds_train_client)
            #     print("Forget Accuracy ")
            #     loss, acc = server_model.evaluate(ds_train_client, verbose=2)
            #
            #     if resume_training and algorithm not in ["natural"] and test_acc >= test_acc_retrained:
            #         break


if __name__ == "__main__":
    main()
