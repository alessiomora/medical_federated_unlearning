""" Minimal script to simualte a federated training on ProstateMRI dataset.
    Clients are represented by 6 sites, which hold datasets with heterogeneous features.
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import data_utility
import tensorflow as tf
import numpy as np
import pickle
import time
from tensorflow import keras
import fedfa_utility
import residual_model_custom as res
# from non_iid_algorithms.FedDynModel import FedDynModel
# from non_iid_algorithms.FedGKDModel import FedGKDModel
# from non_iid_algorithms.FedNTDModel import FedNTDModel
# from non_iid_algorithms.FedProxModel import FedProxModel
# from non_iid_algorithms.FedLGICModel import FedLGICModel
# from non_iid_algorithms.FedLGICDModel import FedLGICDModel
# from non_iid_algorithms.FedMLB2Model import FedMLB2Model
# from non_iid_algorithms.FedLCModel import FedLCModel
# from non_iid_algorithms.MoonModel import MoonModel
# from non_iid_algorithms.FedDynMLBModel import FedDynMLBModel
# from non_iid_algorithms.FedMAXModel import FedMAXModel
# from training_loop_custom import train_one_epoch
import prostate_utility.prostate_utility as utility
from prostate_utility.unet import UNet
import tensorflow_datasets as tfds
import logging_utility as logging
from typing import Optional
import gc

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

PATH = os.path.join("medical_img_results")


# ---------- clear memory -----------------
def reset_tensorflow_keras_backend():
    # to be further investigated, but this seems to be enough
    tf.keras.backend.clear_session()
    # tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()
    return


def create_model(architecture, num_classes, norm, l2_weight_decay=0.0, seed: Optional[int] = None):
    if architecture == "resnet18":
        return res.create_resnet18(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                   l2_weight_decay=l2_weight_decay, seed=seed)
    else:
        return res.create_resnet8(input_shape=(32, 32, 3), num_classes=num_classes, norm=norm,
                                  l2_weight_decay=l2_weight_decay, seed=seed)


def create_server_optimizer(optimizer="sgd", momentum=0.0, lr=1.0):
    if optimizer == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    elif optimizer == "adam":
        return keras.optimizers.Adam(learning_rate=lr, epsilon=1e-3)


if __name__ == '__main__':
    # Building a dictionary of hyperparameters
    hp = {}
    hp["algorithm"] = ["fedavg_eps"]
    hp["local_batch_size"] = [16]
    hp["E"] = [1]  # local_epochs
    hp["C"] = [6]  # n clients
    hp["total_clients"] = [6]
    hp["rounds"] = [500]
    hp["lr_client"] = [1e-4]
    hp["momentum"] = [0.0]
    hp["weight_decay"] = [1e-4]
    hp["server_side_optimizer"] = ["sgd"]
    hp["lr_server"] = [1.0]
    hp["server_momentum"] = [0.0]
    hp["lr_decay"] = [1.0]
    hp["seed"] = [0]  # seed for client selection, model initialization, client data shuffling
    hp["rotate_flip"] = ["yes"]
    # algorithm-specific hyperparameter
    cd = {}
    # Creating a list of dictionaries
    # each one for a combination of hp + algorithm-specific hyperparams
    settings = logging.get_combinations(hp, cd)

    # Running a simulation for each of the setting in hp possible combinations
    for setting in settings:
        print("Simulation start with configuration " + str(setting))
        total_rounds = setting["rounds"]
        k = setting["C"]
        total_clients = setting["total_clients"]
        local_epochs = setting["E"]  # local_epochs
        local_batch_size = setting["local_batch_size"]
        test_batch_size = 256
        algorithm = setting["algorithm"]
        random_seed = setting["seed"]
        lr_client_initial = setting["lr_client"]
        l2_weight_decay = setting["weight_decay"]
        momentum = setting["momentum"]
        lr_server = setting["lr_server"]
        server_side_optimizer = setting["server_side_optimizer"]
        server_momentum = setting["server_momentum"]
        exp_decay = setting["lr_decay"]
        num_classes = 2
        tf.keras.utils.set_random_seed(random_seed)
        norm = 'batch'
        lr_client = lr_client_initial

        sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
        # train, test, val dataset partitioned as fedfa (0.6, 0.2, 0.2)
        local_n_examples = np.array([156, 94, 280, 246, 230, 105])
        #                           167, 100, 299, 263, 245, 112.

        server_model = UNet(output_channels=2)
        server_model.build((None, 3, 384, 384))
        server_model.summary()
        server_optimizer = keras.optimizers.SGD(learning_rate=lr_server)

        ce_dice_loss = utility.JointLoss()
        dice_coeff = utility.dice_metric
        server_model.compile(
            optimizer=server_optimizer,
            loss=ce_dice_loss,
            metrics=[dice_coeff]
        )
        server_model.summary(expand_nested=True)

        # Clients share the same architecture and local config.
        client_model = UNet(output_channels=2)
        client_model.build((None, 3, 384, 384))

        # ..../dataset_alpha--_C--_k--/algorithm/general_hyperparameters/specific_hyperparameters,seed---
        dataset = 'prostateMRI'
        logdir = os.path.join(PATH, dataset, algorithm)

        global_summary_writer = [tf.summary.create_file_writer(os.path.join(logdir, sites[site], "global_test"))
                                 for site in range(len(sites))]
        global_summary_writer_mean = tf.summary.create_file_writer(os.path.join(logdir, "average", "global_test"))

        for rnd in range(1, total_rounds + 1):
            print("[Server] Evaluation - Round: ", rnd)
            mean_client_loss = 0
            mean_client_dice_coef = 0
            # here
            for site in range(len(sites)):
                test_ds = utility.get_dataset_site(sites[site], 'test').map(utility.element_norm_fn).batch(1024)
                history = server_model.evaluate(test_ds, return_dict=True)
                with global_summary_writer[site].as_default():
                    loss = tf.squeeze(history["loss"])
                    dice_coef = tf.squeeze(history["dice_metric"])
                    tf.summary.scalar('loss', loss, step=rnd)
                    tf.summary.scalar('dice_metric', dice_coef, step=rnd)

                    mean_client_loss = mean_client_loss + loss / k
                    mean_client_dice_coef = mean_client_dice_coef + dice_coef / k
            with global_summary_writer_mean.as_default():
                print("Average loss: ", mean_client_loss, " Average dice coef:", mean_client_dice_coef)
                tf.summary.scalar('loss', mean_client_loss, step=rnd)
                tf.summary.scalar('dice_metric', mean_client_dice_coef, step=rnd)

            list_of_clients = [i for i in range(0, total_clients)]
            sampled_clients = np.random.choice(
                list_of_clients,
                size=k,
                replace=False)
            print("Selected clients for the next round ", sampled_clients)

            # variables to store updates
            delta_w_global_trainable = tf.nest.map_structure(lambda a, b: a - b,
                                                             server_model.trainable_weights,
                                                             server_model.trainable_weights)
            if norm in ["batch"]:
                delta_w_with_batch = tf.nest.map_structure(lambda a, b: a - b,
                                                           server_model.get_weights(),
                                                           server_model.get_weights())

            selected_client_examples = local_n_examples[sampled_clients.tolist()]
            print("Total examples ", np.sum(selected_client_examples))
            print("Local examples selected clients ", selected_client_examples)
            total_examples = np.sum(selected_client_examples)

            for c in range(0, k):
                reset_tensorflow_keras_backend()
                print("Client: ", c)
                local_examples = selected_client_examples[c]
                print("Local examples: ", local_examples)

                # Local training
                print(f"[Client {c}] Local Training ---")
                training_dataset = utility.get_dataset_site(sites[c], 'train')
                training_dataset = training_dataset \
                    .map(utility.element_norm_fn) \
                    .map(utility.rotate_flip_fn) \
                    .shuffle(buffer_size=1024, seed=random_seed) \
                    .batch(local_batch_size)

                opt = keras.optimizers.Adam(learning_rate=lr_client,
                                            weight_decay=l2_weight_decay,
                                            epsilon=1e-08)
                # compile clients' model
                ce_dice_loss = utility.JointLoss()
                dice_coeff = utility.dice_metric

                client_model.compile(opt,
                                     loss=ce_dice_loss,
                                     metrics=[dice_coeff]
                                     )
                client_model.set_weights(server_model.get_weights())

                # memory leak in .fit
                history = client_model.fit(
                    training_dataset,
                    epochs=local_epochs,
                )

                if norm in ["batch"]:
                    delta_w_local_with_batch = tf.nest.map_structure(lambda a, b: a - b,
                                                                     client_model.get_weights(),
                                                                     server_model.get_weights(),
                                                                     )

                    delta_w_with_batch = tf.nest.map_structure(
                        # lambda a, b: a + b * (local_examples / total_examples),
                        lambda a, b: a + b * (1 / k),
                        delta_w_with_batch,
                        delta_w_local_with_batch)
                else:
                    delta_w_local_trainable = tf.nest.map_structure(lambda a, b: a - b,
                                                                    client_model.trainable_variables,
                                                                    server_model.trainable_variables,
                                                                    )
                    delta_w_global_trainable = tf.nest.map_structure(
                        # lambda a, b: a + b * (local_examples / total_examples),
                        lambda a, b: a + b * (1 / k),
                        delta_w_global_trainable,
                        delta_w_local_trainable)

                # clear memory
                del opt
                reset_tensorflow_keras_backend()

            if norm in ["batch"]:
                new_ww = tf.nest.map_structure(lambda a, b: a + b,
                                               server_model.get_weights(),
                                               delta_w_with_batch)
                server_model.set_weights(new_ww)
            else:
                gradients = tf.nest.map_structure(lambda a: -a, delta_w_global_trainable)
                server_model.optimizer.apply_gradients(zip(gradients, server_model.trainable_variables))

            # lr_client *= exp_decay
