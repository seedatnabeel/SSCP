# stdlib
import argparse
import logging
import os
import random
import warnings

# third party
import datasets
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from cqr import helper  # From https://github.com/yromano/cqr
from nonconformist.nc import QuantileRegErrFunc, RegressorNc
from predictive_models import *
from self_supervised_tasks import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import set_random_seed
from tqdm import tqdm

warnings.filterwarnings("ignore")

np.warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.INFO)

# Load the WANDB YAML file
with open("../wandb.yaml") as file:
    wandb_data = yaml.load(file, Loader=yaml.FullLoader)

os.environ["WANDB_API_KEY"] = wandb_data["wandb_key"]
wandb_entity = wandb_data["wandb_entity"]


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(seed)


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bike", type=str)
    parser.add_argument("--test-size", default=0.33, type=float)
    parser.add_argument("--labeled-prop", default=0.4, type=float)
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=10, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    project_name = "cqr_unalabeled"
    logging.info("WANDB init...")

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

    logging.info("Getting dataset...")

    X, y = datasets.GetDataset(dataset_name, "../datasets/")

    cqr_results_list = []

    seed = int(args.seed)

    # set hyperparameters as per original CQR paper
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

        logging.info("WANDB init...")
        run = wandb.init(
            project=project_name,
            entity=wandb_entity,
        )

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

        random_state_train_test = seed
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

        # divide into labeled and unlabeled
        X_unlabeled, X_labeled, _, y_labeled = train_test_split(
            X, y, test_size=labeled_prop, random_state=random_state_train_test
        )  # labeled vs unlabeled

        # divide into train, test,
        x_train, x_test, y_train, y_test = train_test_split(
            X_labeled,
            y_labeled,
            test_size=test_ratio,
            random_state=random_state_train_test,
        )  # train_full vs test

        # reshape the data
        x_train = np.asarray(x_train)
        x_unlabeled = np.asarray(X_unlabeled)
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
        x_unlabeled = scalerX.transform(x_unlabeled)
        x_test = scalerX.transform(x_test)

        # scale the labels by dividing each by the mean absolute response
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train) / mean_y_train
        y_test = np.squeeze(y_test) / mean_y_train

        results = {}

        ###########################################################################################################
        #
        # CQR
        #
        ###########################################################################################################
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

        # define a CQR object, computes the absolute residual error of points
        # located outside the estimated quantile neural network band
        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train, y_train, x_test, idx_train, idx_cal, alpha
        )

        # compute and print average coverage and average length
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["CQR"] = {"coverage": coverage_cp_qnet, "length": length_cp_qnet}

        ###########################################################################################################
        #
        # SSCP + VIME (LABELED + UNLABELED) - INDEPENDENT TRAINING
        #
        ###########################################################################################################

        print(f"Running CQR+VIME - {i}")
        print(f"Fitting VIME")

        # combine labeled and unlabeled data
        ssl_train_unlabeled = np.vstack([x_unlabeled, x_train[idx_train, :]])

        # fit VIME
        vime = Vime_Task(epochs=epochs, seed=seed)
        vime.fit(ssl_train_unlabeled)

        # predict using VIME
        ssl_train_vime = vime.predict(x_train).reshape(-1, 1)
        ssl_test_vime = vime.predict(x_test).reshape(-1, 1)

        # scale SSL outputs
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train_vime)
        ssl_train_sc_vime = scaler_ssl.transform(ssl_train_vime)
        ssl_test_sc_vime = scaler_ssl.transform(ssl_test_vime)

        # augment the CQR data with SSL data
        x_train_aug = np.hstack([x_train, ssl_train_sc_vime])
        x_test_aug = np.hstack([x_test, ssl_test_sc_vime])

        # compute input dimensions
        n_train = x_train_aug.shape[0]
        in_shape = x_train_aug.shape[1]

        print(f"Fitting CQR+VIME...")
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

        # define a CQR object, computes the absolute residual error of points
        # located outside the estimated quantile neural network band
        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train_aug, y_train, x_test_aug, idx_train, idx_cal, alpha
        )

        # compute and print average coverage and average length
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["VIME+UNLAB"] = {"coverage": coverage_cp_qnet, "length": length_cp_qnet}

        ###########################################################################################################
        #
        # SSCP + VIME (LABELED + UNLABELED) - SHARED TRAINING
        #
        ###########################################################################################################

        print(f"Running CQR+VIME (SHARED) - {i}")
        print(f"Fitting VIME...")

        # fit a predictive model
        pred_model = predictive_model(
            input_shape=x_train[idx_train, :].shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        pred_model.fit(x_train[idx_train, :], y_train[idx_train])

        # use the predictive model output as embedding for VIME
        vime = Vime_Task(epochs=epochs, seed=seed)
        vime.fit(pred_model.extract_encoding(ssl_train_unlabeled))

        # predict using VIME
        ssl_train_vime = vime.predict(pred_model.extract_encoding(x_train)).reshape(
            -1, 1
        )
        ssl_test_vime = vime.predict(pred_model.extract_encoding(x_test)).reshape(-1, 1)

        # scale SSL outputs
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train_vime)

        ssl_train_sc_vime = scaler_ssl.transform(ssl_train_vime)
        ssl_test_sc_vime = scaler_ssl.transform(ssl_test_vime)

        # augment the CQR data with SSL data
        x_train_aug = np.hstack([x_train, ssl_train_sc_vime])
        x_test_aug = np.hstack([x_test, ssl_test_sc_vime])

        # compute input dimensions
        n_train = x_train_aug.shape[0]
        in_shape = x_train_aug.shape[1]

        print(f"Fitting CQR+VIME (SHARED")
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

        # define a CQR object, computes the absolute residual error of points
        # located outside the estimated quantile neural network band
        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train_aug, y_train, x_test_aug, idx_train, idx_cal, alpha
        )

        # compute and print average coverage and average length
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["VIME+SHARED+UNLAB"] = {
            "coverage": coverage_cp_qnet,
            "length": length_cp_qnet,
        }

        ###########################################################################################################
        #
        # SSCP + VIME (LABELED ONLY) - INDEPENDENT TRAINING
        #
        ###########################################################################################################

        print(f"Running CQR+VIME - {i}")
        print(f"Fitting VIME")
        vime = Vime_Task(epochs=epochs, seed=seed)
        vime.fit(x_train[idx_train, :])

        # predict using VIME
        ssl_train_vime = vime.predict(x_train).reshape(-1, 1)
        ssl_test_vime = vime.predict(x_test).reshape(-1, 1)

        # scale SSL outputs
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train_vime)

        ssl_train_sc_vime = scaler_ssl.transform(ssl_train_vime)
        ssl_test_sc_vime = scaler_ssl.transform(ssl_test_vime)

        # augment the CQR data with SSL data
        x_train_aug = np.hstack([x_train, ssl_train_sc_vime])
        x_test_aug = np.hstack([x_test, ssl_test_sc_vime])

        # compute input dimensions
        n_train = x_train_aug.shape[0]
        in_shape = x_train_aug.shape[1]

        print(f"Fitting CQR+VIME")
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

        # define a CQR object, computes the absolute residual error of points
        # located outside the estimated quantile neural network band
        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train_aug, y_train, x_test_aug, idx_train, idx_cal, alpha
        )

        # compute and print average coverage and average length
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["VIME+LAB"] = {"coverage": coverage_cp_qnet, "length": length_cp_qnet}

        ###########################################################################################################
        #
        # SSCP + VIME (LABELED ONLY) - SHARED TRAINING
        #
        ###########################################################################################################

        print(f"Running CQR+VIME (SHARED) - {i}")
        print(f"Fitting VIME")

        # fit a predictive model
        pred_model = predictive_model(
            input_shape=x_train[idx_train, :].shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        pred_model.fit(x_train[idx_train, :], y_train[idx_train])

        # use the predictive model output as embedding for VIME
        vime = Vime_Task(epochs=epochs, seed=seed)
        vime.fit(pred_model.extract_encoding(x_train[idx_train, :]))

        # predict using VIME
        ssl_train_vime = vime.predict(pred_model.extract_encoding(x_train)).reshape(
            -1, 1
        )
        ssl_test_vime = vime.predict(pred_model.extract_encoding(x_test)).reshape(-1, 1)

        # scale SSL outputs
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train_vime)
        ssl_train_sc_vime = scaler_ssl.transform(ssl_train_vime)
        ssl_test_sc_vime = scaler_ssl.transform(ssl_test_vime)

        # augment the CQR data with SSL data
        x_train_aug = np.hstack([x_train, ssl_train_sc_vime])
        x_test_aug = np.hstack([x_test, ssl_test_sc_vime])

        # compute input dimensions
        n_train = x_train_aug.shape[0]
        in_shape = x_train_aug.shape[1]

        print(f"Fitting CQR+VIME (SHARED")
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

        # define a CQR object, computes the absolute residual error of points
        # located outside the estimated quantile neural network band
        nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

        # run CQR procedure
        y_lower, y_upper = helper.run_icp(
            nc, x_train_aug, y_train, x_test_aug, idx_train, idx_cal, alpha
        )

        # compute and print average coverage and average length
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(
            y_test, y_lower, y_upper, alpha, "CQR Neural Net"
        )

        results["VIME+SHARED+LAB"] = {
            "coverage": coverage_cp_qnet,
            "length": length_cp_qnet,
        }

        logging.info("DONE")

        ###########################################################################################################
        #
        # LOG RESULTS
        #
        ###########################################################################################################

        wandb.log({"cqr": results})

        wandb.finish()
        cqr_results_list.append(results)

        seed += 1
