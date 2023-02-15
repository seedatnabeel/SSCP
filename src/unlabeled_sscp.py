# stdlib
import argparse
import logging
import os
import time
import warnings

# third party
import datasets
import numpy as np
import wandb
import yaml
from predictive_models import *
from self_supervised_tasks import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import set_random_seed
from tqdm import tqdm
from utils import compute_deficet, compute_excess, compute_interval_metrics

warnings.filterwarnings("ignore")

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
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=10, type=int)

    return parser.parse_args()


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


if __name__ == "__main__":

    project_name = "crf_unlabeled"

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

    crf_results_list = []

    test_prop = 0.2
    cal_prop = 0.2

    seed = int(args.seed)
    batch_size = 128

    for i in tqdm(range(args.runs)):

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
        seed += 10  # increment per run so we test different random seeds

        seed_everything(seed)

        ##########################################################################
        #
        # Process data
        #
        ##########################################################################

        # divide into labeled and unlabeled
        X_unlabeled, X_labeled, _, y_labeled = train_test_split(
            X, y, test_size=labeled_prop, random_state=seed
        )  # labeled vs unlabeled

        # divide into train, test, cal
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_labeled, y_labeled, test_size=test_prop, random_state=seed
        )  # train_full vs test
        X_train, X_residual, y_train, y_residual = train_test_split(
            X_train_full, y_train_full, test_size=cal_prop, random_state=seed
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=cal_prop, random_state=seed
        )

        # scale data
        scaler_feats = StandardScaler()
        scaler_feats.fit(X_train)

        X_train_sc = scaler_feats.transform(X_train)
        X_unlabeled_sc = scaler_feats.transform(X_unlabeled)
        X_residual_sc = scaler_feats.transform(X_residual)
        X_cal_sc = scaler_feats.transform(X_cal)
        X_test_sc = scaler_feats.transform(X_test)

        scaler_labels = StandardScaler()
        scaler_labels.fit(y_train.reshape(-1, 1))

        # scale labels
        mean_ytrain = np.mean(np.abs(y_train))
        y_train_sc = np.squeeze(y_train) / mean_ytrain
        y_cal_sc = np.squeeze(y_cal) / mean_ytrain
        y_residual_sc = np.squeeze(y_residual) / mean_ytrain
        y_test_sc = np.squeeze(y_test) / mean_ytrain

        y_train_sc = y_train_sc.reshape(-1, 1)
        y_cal_sc = y_cal_sc.reshape(-1, 1)
        y_residual_sc = y_residual_sc.reshape(-1, 1)
        y_test_sc = y_test_sc.reshape(-1, 1)

        # Train base predictive model
        pred_model = predictive_model(
            input_shape=X_train_sc.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        pred_model.fit(X_train_sc, y_train_sc)

        # combine unlabeled and labeled training data
        ssl_train_data = np.vstack([X_unlabeled_sc, X_train_sc])

        # keep just labeled data
        ssl_train_only = X_train_sc

        # Train VIME on both labeled and unlabeled data
        vime = Vime_Task(epochs=epochs, seed=seed)
        vime.fit(pred_model.extract_encoding(ssl_train_data))

        # Train VIME on just labeled data
        vime_train = Vime_Task(epochs=epochs, seed=seed)
        vime_train.fit(pred_model.extract_encoding(ssl_train_only))

        results = {}

        ###########################################################################################################
        # Make predictions with the main model
        # Predict on the calibration set
        # calculate q_yhat for the calibration set

        alpha = 0.1

        # predict on calibration
        ypred_cal = pred_model.predict(X_cal_sc).reshape(-1, 1)

        # predict on test
        ypred_test = pred_model.predict(X_test_sc).reshape(-1, 1)

        N = len(y_cal_sc)
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal),
            np.ceil((N + 1) * (1 - alpha)) / N,
            interpolation="lower",
        )

        main_model_errors = np.abs(ypred_test - y_test_sc)

        ###########################################################################################################
        #
        # CRF
        #
        ###########################################################################################################

        logging.info("Running conformalized residual fitting")
        # calculate residuals
        res_y = np.abs(y_residual_sc - pred_model.predict(X_residual_sc).reshape(-1, 1))

        # train residual model
        model_r = predictive_model(
            input_shape=X_residual_sc.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model_r.fit(X_residual_sc, res_y)

        N = len(y_cal_sc)
        # calculate q_yhat for the calibration set
        res_ypred_cal = model_r.predict(X_cal_sc).reshape(-1, 1)
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal) / res_ypred_cal,
            np.ceil((N + 1) * (1 - alpha)) / N,
            interpolation="lower",
        )

        # predict with 1-alpha confidence - scaled by residual predictions
        res_ypred_test = model_r.predict(X_test_sc).reshape(-1, 1)
        lower_bound = ypred_test - (q_yhat * res_ypred_test.reshape(-1, 1))
        upper_bound = ypred_test + (q_yhat * res_ypred_test.reshape(-1, 1))

        residual_model_errors = np.abs(ypred_test - y_test_sc)

        _, avg_length = compute_interval_metrics(lower_bound, upper_bound, y_test_sc)
        avg_excess, proportion_excess = compute_excess(
            lower_bound, upper_bound, y_test_sc
        )
        avg_deficet, proportion_deficet = compute_deficet(
            lower_bound, upper_bound, y_test_sc
        )

        results["CRF"] = {
            "avg_length": avg_length,
            "avg_excess": avg_excess,
            "avg_deficet": avg_deficet,
        }

        ###########################################################################################################
        #
        # SSCP + VIME (LABELED + UNLABELED)
        #
        ###########################################################################################################

        logging.info("VIME SSL Running SSL conformalized residual fitting")

        # make predictions with the vime model
        ssl_train = vime.predict(pred_model.extract_encoding(ssl_train_data)).reshape(
            -1, 1
        )
        ssl_cal = vime.predict(pred_model.extract_encoding(X_cal_sc)).reshape(-1, 1)
        ssl_residual = vime.predict(pred_model.extract_encoding(X_residual_sc)).reshape(
            -1, 1
        )
        ssl_test = vime.predict(pred_model.extract_encoding(X_test_sc)).reshape(-1, 1)

        # scale the SSL
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train)
        ssl_train_sc = scaler_ssl.transform(ssl_train)
        ssl_cal_sc = scaler_ssl.transform(ssl_cal)
        ssl_residual_sc = scaler_ssl.transform(ssl_residual)
        ssl_test_sc = scaler_ssl.transform(ssl_test)

        res_y = np.abs(y_residual_sc - pred_model.predict(X_residual_sc).reshape(-1, 1))

        # fit a new model to predict residuals - augmented with SSL task
        X_train_r = np.hstack([X_residual_sc, ssl_residual_sc])
        model_r = predictive_model(
            input_shape=X_train_r.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model_r.fit(X_train_r, res_y)

        # calculate q_yhat for the calibration set
        X_cal_r = np.hstack([X_cal_sc, ssl_cal_sc])
        res_ypred_cal = model_r.predict(X_cal_r).reshape(-1, 1)
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal) / res_ypred_cal,
            np.ceil((N + 1) * (1 - alpha)) / N,
            interpolation="lower",
        )

        # predict with 1-alpha confidence - scaled by residual predictions
        X_test_r = np.hstack([X_test_sc, ssl_test_sc])
        res_ypred_test = model_r.predict(X_test_r).reshape(-1, 1)
        lower_bound = ypred_test - (q_yhat * res_ypred_test.reshape(-1, 1))
        upper_bound = ypred_test + (q_yhat * res_ypred_test.reshape(-1, 1))

        residual_model_errors = np.abs(ypred_test - y_test_sc)

        _, avg_length = compute_interval_metrics(lower_bound, upper_bound, y_test_sc)
        avg_excess, proportion_excess = compute_excess(
            lower_bound, upper_bound, y_test_sc
        )
        avg_deficet, proportion_deficet = compute_deficet(
            lower_bound, upper_bound, y_test_sc
        )

        results["CRF+VIME"] = {
            "avg_length": avg_length,
            "avg_excess": avg_excess,
            "avg_deficet": avg_deficet,
        }

        ###########################################################################################################
        #
        # SSCP + VIME (LABELED ONLY)
        #
        ###########################################################################################################
        logging.info("VIME SSL Running SSL conformalized residual fitting")

        # make predictions with the vime model
        ssl_train = vime_train.predict(
            pred_model.extract_encoding(ssl_train_only)
        ).reshape(-1, 1)
        ssl_cal = vime_train.predict(pred_model.extract_encoding(X_cal_sc)).reshape(
            -1, 1
        )
        ssl_residual = vime_train.predict(
            pred_model.extract_encoding(X_residual_sc)
        ).reshape(-1, 1)
        ssl_test = vime_train.predict(pred_model.extract_encoding(X_test_sc)).reshape(
            -1, 1
        )

        # scale the SSL
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train)
        ssl_train_sc = scaler_ssl.transform(ssl_train)
        ssl_cal_sc = scaler_ssl.transform(ssl_cal)
        ssl_residual_sc = scaler_ssl.transform(ssl_residual)
        ssl_test_sc = scaler_ssl.transform(ssl_test)

        res_y = np.abs(y_residual_sc - pred_model.predict(X_residual_sc).reshape(-1, 1))

        # fit a new model to predict residuals - augmented with SSL task
        X_train_r = np.hstack([X_residual_sc, ssl_residual_sc])
        model_r = predictive_model(
            input_shape=X_train_r.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model_r.fit(X_train_r, res_y)

        # calculate q_yhat for the calibration set
        X_cal_r = np.hstack([X_cal_sc, ssl_cal_sc])
        res_ypred_cal = model_r.predict(X_cal_r).reshape(-1, 1)
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal) / res_ypred_cal,
            np.ceil((N + 1) * (1 - alpha)) / N,
            interpolation="lower",
        )

        # predict with 1-alpha confidence - scaled by residual predictions
        X_test_r = np.hstack([X_test_sc, ssl_test_sc])
        res_ypred_test = model_r.predict(X_test_r).reshape(-1, 1)
        lower_bound = ypred_test - (q_yhat * res_ypred_test.reshape(-1, 1))
        upper_bound = ypred_test + (q_yhat * res_ypred_test.reshape(-1, 1))

        residual_model_errors = np.abs(ypred_test - y_test_sc)

        _, avg_length = compute_interval_metrics(lower_bound, upper_bound, y_test_sc)
        avg_excess, proportion_excess = compute_excess(
            lower_bound, upper_bound, y_test_sc
        )
        avg_deficet, proportion_deficet = compute_deficet(
            lower_bound, upper_bound, y_test_sc
        )

        results["CRF+VIME_TRAIN"] = {
            "avg_length": avg_length,
            "avg_excess": avg_excess,
            "avg_deficet": avg_deficet,
        }

        ###########################################################################################################
        #
        # LOG RESULTS
        #
        ###########################################################################################################
        wandb.log({"CRF": results})

        wandb.finish()
        crf_results_list.append(results)

        seed += 1
