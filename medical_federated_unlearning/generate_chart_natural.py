
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
import hydra

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd

import numpy as np

from medical_federated_unlearning import utility
from medical_federated_unlearning.generate_csv_results import compute_yeom_mia
from medical_federated_unlearning.mia_svc import SVC_MIA
from medical_federated_unlearning.prostate_utility import prostate_utility
from medical_federated_unlearning.prostate_utility.unet import UNet
from medical_federated_unlearning.utility import list_clients_to_string

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


#plt.style.use("seaborn-whitegrid")
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



def draw_chart(cid, df, retrained_value, metric="train_acc", y_label="Accuracy",
               dataset="cifar100"):
    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Define line styles and colors
    line_styles = ['-', '--', ':']
    colors = sns.color_palette('tab10', n_colors=2)

    # Plot the first couple of lines
    plt.plot(df['round_recovery'], df[metric], linestyle=line_styles[0],
             color=colors[0])
    # plt.plot(df['round_recovery'], df['yeom_mia'], linestyle=line_styles[1],
    #          color=colors[0])
    # plt.plot(df['round_recovery'], df['mia'], linestyle=line_styles[2],
    #          color=colors[0])

    # Add a straight line
    plt.axhline(y=retrained_value, color=colors[1], linestyle=line_styles[1])
    # plt.axhline(y=retrained_yeom_mia, color='black', linestyle=line_styles[1])
    # plt.axhline(y=retrained_mia, color='black', linestyle=line_styles[2])

    # Customize the plot
    # plt.title("Line Chart with Different Styles")
    plt.xlabel("Round", fontsize=14)
    plt.ylabel(y_label, fontsize=14)

    plt.xticks(
        ticks=range(df['round_recovery'].min(), df['round_recovery'].max() + 1, 5))
    plt.xlim(df['round_recovery'].min(), df['round_recovery'].max())
    # plt.ylim(df['round_recovery'].min(), df['round_recovery'].max())

    legend_lines = [
        mlines.Line2D([], [], linestyle=line_styles[0],
                      color=colors[0],
                      label="Natural"),
        # mlines.Line2D([], [], linestyle=line_styles[1],
        #           color=colors[0], label='MIA [Yeom et al.] (N)'),
        # mlines.Line2D([], [], linestyle=line_styles[2],
        #           color=colors[0], label='MIA [Song et al.] (N)'),
        mlines.Line2D([], [], linestyle=line_styles[1],
                      color=colors[1],
                      label="Retrained"),
        # mlines.Line2D([], [], linestyle=line_styles[1],
        #               color='black',
        #               label='MIA [Yeom et al.] (R)'),
        # mlines.Line2D([], [], linestyle=line_styles[2],
        #               color='black',
        #               label='MIA [Song et al.] (R)'),
    ]

    # Add the legend to the plot
    plt.legend(handles=legend_lines, loc='best', facecolor='white', edgecolor='gray', framealpha=1.0, fontsize=14)

    # plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    base_path = os.path.join("charts", f"natural_{dataset}")
    path_to_save = os.path.join(base_path,
                                f"natural_{metric}_client{cid}.pdf")
    exist = os.path.exists(base_path)

    if not exist:
        os.makedirs(base_path)

    plt.savefig(path_to_save, dpi=300, bbox_inches="tight",
                format="pdf")
    return


def create_load_compile_UNET_model(model_checkpoint_dir):
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
    return model


@hydra.main(config_path="conf", config_name="generate_tables", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #  build base config
    dataset = cfg.dataset
    total_clients = cfg.total_clients
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    model_string = cfg.model
    seed  = cfg.seed
    rounds_to_check = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                       30, 40, 50, 60, 70, 80, 90, 100]
    # rounds_to_check = [1, 2, 3, 4, 5]
    config_dir = os.path.join(f"{dataset}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )

    # ds_test_sites, ds_train_sites, ds_test_union, samples_per_client = utility.get_test_and_train_dataset()
    ds_test_sites, ds_train_sites, ds_test_union, ds_test_mia_balanced, samples_per_client, min_samples_test = utility.get_test_and_train_dataset()

    # test_cardinality = np.sum(samples_per_client)

    # for cid in [0, 1, 2, 3, 4, 5]:
    for cid in [[2]]:
        unlearned_cid_string = list_clients_to_string(cid)
        print(unlearned_cid_string)
        ds_retain, ds_retain_yeom = utility.get_union_retain_dataset(cid,
                                                                     min_samples_test)

        client_train_ds = utility.get_forget_dataset(cid)
        forgetting_ds = client_train_ds

        original_computed = False
        retrained_dict = {}
        entry_list = []
        # for what in ["original", "retrained", "natural"]:
        for what in ["retrained", "natural"]:
            print(f"What: {what}")
            entry_pd = {}
            if what in ["original", "retrained"]:
                what_string = "" if what == "original" else "_retrained"
                client_dir = os.path.join("checkpoints") if what == "original" else os.path.join(f"client{unlearned_cid_string}", "checkpoints")
                exist = os.path.exists(os.path.join("model_checkpoints" + what_string, config_dir, client_dir))
                print(os.path.join("model_checkpoints" + what_string, config_dir, client_dir))
                print(exist)
                if exist:
                    if what in ["retrained"] or not original_computed:
                        last_checkpoint_retrained = find_last_checkpoint(os.path.join("model_checkpoints"+what_string, config_dir, client_dir))

                        model_checkpoint_dir = os.path.join("model_checkpoints"+what_string, config_dir,
                                                            client_dir,
                                                            f"R_{last_checkpoint_retrained}")

                        print(f"-- {what} last saved round: {last_checkpoint_retrained} --")

                        model = create_load_compile_UNET_model(model_checkpoint_dir)

                        # --- MIA Yeom et al.
                        yeom_mia = compute_yeom_mia(model, train_data=ds_retain_yeom,
                                                    forget_data=forgetting_ds,
                                                    model_name=model_string)

                        # --- MIA Efficiency
                        results_mia = SVC_MIA(shadow_train=ds_retain,
                                              shadow_test=ds_test_union,
                                              target_train=forgetting_ds,
                                              target_test=None,
                                              model=model,
                                              model_name=model_string)

                        test_loss, test_acc = model.evaluate(ds_test_union)
                        test_acc = test_acc * 100
                        train_loss, train_acc = model.evaluate(client_train_ds)
                        train_acc = train_acc * 100
                        ua = 100.0 - train_acc
                        mia = results_mia * 100

                        print(f"Test acc: {test_acc} -- Train acc: {train_acc} -- Train loss: {train_loss} -- mia: {mia} -- yeom_mia:{yeom_mia} ")
                        entry_pd["test_acc"] = test_acc
                        entry_pd["train_acc"] = train_acc
                        entry_pd["ua"] = ua
                        entry_pd["mia"] = mia
                        entry_pd["yeom_mia"] = yeom_mia
                        entry_pd["name"] = f"{what}_client{cid}"
                        entry_pd["cid"] = cid
                        entry_pd["algorithm"] = what
                        entry_pd["train_loss"] = train_loss

                        if what in ["retrained"]:
                            retrained_dict["test_acc"] = test_acc
                            retrained_dict["train_acc"] = train_acc
                            retrained_dict["train_loss"] = train_loss
                            retrained_dict["mia"] = mia
                            retrained_dict["yeom_mia"] = yeom_mia

                        if what in ["original"]:
                            original_computed = True
                            entry_pd["round_recovery"] = 0
                            entry_list.append(entry_pd)


            elif what in ["natural"]:
                model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                    config_dir,
                                                    what)

                list_configs = find_configs_in_folder(model_checkpoint_dir)
                print(list_configs)
                print(model_checkpoint_dir)
                for unlearning_config in list_configs:
                    # if rounds_to_check in unlearning_config:
                    if True:
                        model_checkpoint_base_dir = os.path.join(model_checkpoint_dir,
                                                                 unlearning_config,
                                                                 f"client{unlearned_cid_string}",
                                                                 "checkpoints")

                        exist = os.path.exists(model_checkpoint_base_dir)
                        if exist:
                            print(f"-- Resumed config {unlearning_config} client {unlearned_cid_string}--")

                            # for round_recovery in range(1, rounds_to_check):
                            for round_recovery in rounds_to_check:
                                entry_pd = {}
                                r = last_checkpoint_retrained + round_recovery
                                print(
                                    f"-- Resumed config {unlearning_config} client {unlearned_cid_string} round {r}--")
                                model_checkpoint_round_dir = os.path.join(
                                    model_checkpoint_base_dir,
                                    f"R_{r}")
                                exist = os.path.exists(model_checkpoint_round_dir)
                                if not exist:
                                    break


                                model = create_load_compile_UNET_model(model_checkpoint_round_dir)


                                test_loss, test_acc = model.evaluate(ds_test_union)
                                test_acc = test_acc * 100

                                # --- MIA Yeom et al.
                                yeom_mia = compute_yeom_mia(model,
                                                            train_data=ds_retain_yeom,
                                                            forget_data=forgetting_ds,
                                                            model_name=model_string)

                                # --- MIA Efficiency
                                # Should be
                                # ds_retain balanced per site
                                # ds_test balanced per site
                                results_mia = SVC_MIA(shadow_train=ds_retain,
                                                      shadow_test=ds_test_union,
                                                      target_train=forgetting_ds,
                                                      target_test=None,
                                                      model=model,
                                                      model_name=model_string)

                                train_loss, train_acc = model.evaluate(client_train_ds)
                                train_acc = train_acc * 100
                                ua = 100.0 - train_acc
                                mia = results_mia * 100


                                print(
                                    f"Test acc: {test_acc} -- Train acc: {train_acc} -- ua {ua} -- mia: {mia}")
                                entry_pd["algorithm"] = what
                                entry_pd["round_recovery"] = round_recovery
                                entry_pd["name"] = unlearning_config
                                entry_pd["test_acc"] = test_acc
                                entry_pd["train_acc"] = train_acc
                                entry_pd["train_loss"] = train_loss
                                entry_pd["ua"] = ua
                                entry_pd["mia"] = mia
                                entry_pd["yeom_mia"] = yeom_mia
                                entry_pd["cid"] = unlearned_cid_string

                                print(entry_pd)
                                entry_list.append(entry_pd)

                            # create chart
                            df = pd.DataFrame(entry_list)
                            print(df)

                            for metric in ["train_acc", "train_loss", "mia", "yeom_mia", "test_acc"]:
                                retrained_value = retrained_dict[metric]
                                if metric in ["kl_div"]:
                                    y_label = "KL Divergence"
                                elif metric in ["train_loss"]:
                                    y_label = "Loss"
                                else:
                                    y_label = "Accuracy"

                                draw_chart(unlearned_cid_string, df, retrained_value=retrained_value, metric=metric,
                                           y_label=y_label, dataset=dataset)



if __name__ == "__main__":
    main()

