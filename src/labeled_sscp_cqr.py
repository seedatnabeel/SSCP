# %%
# stdlib
import argparse
import logging
import os
import random
import time
import warnings

# third party
import datasets
import numpy as np
import torch
import wandb
import yaml
from cqr import helper  # From https://github.com/yromano/cqr
from nonconformist.nc import QuantileRegErrFunc, RegressorNc
from predictive_models import *
from self_supervised_tasks import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import process_results

warnings.filterwarnings("ignore")

np.warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.INFO)

# Load the WANDB YAML file
with open("../wandb.yaml") as file:
    wandb_data = yaml.load(file, Loader=yaml.FullLoader)


os.environ["WANDB_API_KEY"] = wandb_data["wandb_key"]
wandb_entity = wandb_data["wandb_entity"]


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bike", type=str)
    parser.add_argument("--test-size", default=0.33, type=float)
    parser.add_argument("--labeled-prop", default=0.4, type=float)
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--epochs", default=10, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    project_name = "cqr_labeled"
    logging.info("WANDB init...")
    run = wandb.init(
        project=project_name,
        entity=wandb_entity,
    )

    args = init_arg()

    dataset_name = args.dataset
    epochs = args.epochs
    labeled_prop = args.labeled_prop

    assert dataset_name in [
        "star",
        "bike",
        "bio",
        "concrete",
        "community",
        "facebook_1",
        "facebook_2",
        "blog_data",
    ]

    start = time.time()

    logging.info("Getting dataset...")

    X, y = datasets.GetDataset(dataset_name, "../datasets/")

    cqr_results_list = []

    wandb.log(
        {
            "Dataset": args.dataset,
            "Test Size": args.test_size,
            "Runs": args.runs,
            "Samples": len(y),
            "Epochs": epochs,
            "Labeled Prop": labeled_prop,
        }
    )

    seed = 99

    # set hyperparameters as per original CQR
    nn_learn_func = torch.optim.Adam
    epochs = epochs
    lr = 0.0005
    batch_size = 64
    hidden_size = 64
    dropout = 0.1
    wd = 1e-6
    quantiles_net = [0.1, 0.9]
    coverage_factor = 0.90
    cv_test_ratio = 0.1
    cv_random_state = 1
    # desired miscoverage error
    alpha = 0.1

    # desired quanitile levels
    quantiles = [0.1, 0.9]

    # used to determine the size of test set
    test_ratio = 0.2

    for i in tqdm(range(args.runs)):
        seed += 1  # increment per run so we test different random seeds

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        ##########################################################################
        #
        # Process data
        #
        ##########################################################################

        # divide the dataset into test and train based on the test_ratio parameter
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=seed
        )

        # reshape the data
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        # compute input dimensions
        n_train = x_train.shape[0]
        in_shape = x_train.shape[1]

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half], idx[n_half : 2 * n_half]

        # zero mean and unit variance scaling
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train[idx_train])

        # scale
        x_train = scalerX.transform(x_train)
        x_test = scalerX.transform(x_test)

        # scale the labels
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train) / mean_y_train
        y_test = np.squeeze(y_test) / mean_y_train

        results = {}

        ##########################################################################
        #
        # BASELINE CQR
        #
        ##########################################################################

        print(f"Running CQR - {i}")
        # define quantile neural network model
        quantile_estimator = helper.AllQNet_RegressorAdapter(
            model=None,
            fit_params=None,
            in_shape=in_shape,
            hidden_size=hidden_size,
            quantiles=quantiles_net,
            learn_func=nn_learn_func,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            lr=lr,
            wd=wd,
            test_ratio=cv_test_ratio,
            random_state=cv_random_state,
            use_rearrangement=False,
        )

        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train, y_train, x_test, idx_train, idx_cal, alpha
        )

        # compute metrics
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["CQR"] = {"coverage": coverage_cp_qnet, "avg_length": length_cp_qnet}

        ##########################################################################
        #
        # CQR+VIME (INDEPENDENT TRAINING) - Simple augmentation of inputs
        #
        ##########################################################################

        print("Running CQR+VIME...")

        print("Fitting VIME...")
        vime = Vime_Task(epochs=epochs)

        vime.fit(x_train[idx_train, :])

        # fit a new model to predict residuals - augmented with SSL task
        ssl_train_vime = vime.predict(x_train).reshape(-1, 1)
        ssl_test_vime = vime.predict(x_test).reshape(-1, 1)

        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train_vime)

        ssl_train_sc_vime = scaler_ssl.transform(ssl_train_vime)
        ssl_test_sc_vime = scaler_ssl.transform(ssl_test_vime)

        # AUGMENT THE INPUT WITH THE VIME SSL TASK
        x_train_aug = np.hstack([x_train, ssl_train_sc_vime])
        x_test_aug = np.hstack([x_test, ssl_test_sc_vime])

        # compute input dimensions
        n_train = x_train_aug.shape[0]
        in_shape = x_train_aug.shape[1]

        print("Fitting CQR+VIME model...")
        # define quantile neural network model
        quantile_estimator = helper.AllQNet_RegressorAdapter(
            model=None,
            fit_params=None,
            in_shape=in_shape,
            hidden_size=hidden_size,
            quantiles=quantiles_net,
            learn_func=nn_learn_func,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            lr=lr,
            wd=wd,
            test_ratio=cv_test_ratio,
            random_state=cv_random_state,
            use_rearrangement=False,
        )

        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train_aug, y_train, x_test_aug, idx_train, idx_cal, alpha
        )

        # compute metrics
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["CQR+VIME"] = {"coverage": coverage_cp_qnet, "length": length_cp_qnet}

        ##########################################################################
        #
        # CQR+VIME (SHARED_
        #
        ##########################################################################

        print("Running CQR+VIME (SHARED)...")

        # define a predictive model
        pred_model = predictive_model(
            input_shape=x_train[idx_train, :].shape[1],
            epochs=epochs,
            batch_size=batch_size,
        )
        pred_model.fit(x_train[idx_train, :], y_train[idx_train])

        # Fit VIME using a shared representation from the task predictor
        vime = Vime_Task(epochs=epochs)
        vime.fit(pred_model.extract_encoding(x_train[idx_train, :]))

        ssl_train_vime = vime.predict(pred_model.extract_encoding(x_train)).reshape(
            -1, 1
        )
        ssl_test_vime = vime.predict(pred_model.extract_encoding(x_test)).reshape(-1, 1)

        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train_vime)

        ssl_train_sc_vime = scaler_ssl.transform(ssl_train_vime)
        ssl_test_sc_vime = scaler_ssl.transform(ssl_test_vime)

        # AUGMENT THE INPUT WITH THE VIME SSL TASK
        x_train_aug = np.hstack([x_train, ssl_train_sc_vime])
        x_test_aug = np.hstack([x_test, ssl_test_sc_vime])

        # compute input dimensions
        n_train = x_train_aug.shape[0]
        in_shape = x_train_aug.shape[1]

        print("Fitting CQR+VIME (SHARED)...")
        # define quantile neural network model
        quantile_estimator = helper.AllQNet_RegressorAdapter(
            model=None,
            fit_params=None,
            in_shape=in_shape,
            hidden_size=hidden_size,
            quantiles=quantiles_net,
            learn_func=nn_learn_func,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            lr=lr,
            wd=wd,
            test_ratio=cv_test_ratio,
            random_state=cv_random_state,
            use_rearrangement=False,
        )

        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train_aug, y_train, x_test_aug, idx_train, idx_cal, alpha
        )

        # compute metrics
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["CQR+VIME+SHARED"] = {
            "coverage": coverage_cp_qnet,
            "length": length_cp_qnet,
        }

        logging.info("DONE")
        cqr_results_list.append(results)

        seed += 1

    final_cqr = process_results(cqr_results_list)
    print(f"CQR: {final_cqr}")

    wandb.log({"CQR": final_cqr})

    end = time.time()
    print(f"TIME TAKEN = {end - start}")
