# IMPORTS
import argparse
import logging
import os
import time
import warnings

import numpy as np
import wandb
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import set_random_seed
from tqdm import tqdm

import datasets
from predictive_models import *
from self_supervised_tasks import *
from utils import (
    compute_interval_metrics,
    compute_deficet,
    compute_excess,
    process_results,
    write_to_file,
)

logging.getLogger().setLevel(logging.INFO)


warnings.filterwarnings("ignore")

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


logging.getLogger().setLevel(logging.INFO)

import yaml
# Load the WANDB YAML file
with open('../wandb.yaml') as file:
    wandb_data = yaml.load(file, Loader=yaml.FullLoader)

os.environ["WANDB_API_KEY"] = wandb_data['wandb_key'] 
wandb_entity = wandb_data['wandb_entity'] 


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bike", type=str)
    parser.add_argument("--test-size", default=0.33, type=float)
    parser.add_argument("--labeled-prop", default=0.4, type=float)
    parser.add_argument("--runs", default=3, type=int)
    parser.add_argument("--epochs", default=10, type=int)

    return parser.parse_args()

# If you want to write the different intervals to a file for post-hoc analysis
save_intervals = False 

if __name__ == "__main__":
    project_name = "crf_labeled"
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
        "facebook_2",
        "blog_data",
    ]

    start = time.time()

    logging.info(f"Getting dataset {dataset_name}...")

    X, y, seed = datasets.GetDataset(dataset_name, "../datasets/", seeded=True)

    crf_results_list = []
    batch_size = 128
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


    start_seed = seed

    log_res = False

    for i in tqdm(range(args.runs)):
        seed += 1 #increment per run so we test different random seeds
        seed_everything(seed)

        alpha = 0.1

        # Preprocess data
        (
            X_train_sc,
            X_residual_sc,
            X_cal_sc,
            X_test_sc,
            y_train_sc,
            y_residual_sc,
            y_cal_sc,
            y_test_sc,
        ) = datasets.process_data(X=X, y=y, seed=seed)

        # Fit the predictive model on the proper training set
        pred_model = predictive_model(
            input_shape=X_train_sc.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        pred_model.fit(X_train_sc, y_train_sc)

        #######################################################
        #
        # Self-supervised learning
        #
        ######################################################

        # Fit the autoencoder
        ssl_task = AutoEncoder(
            input_shape=pred_model.extract_encoding(X_train_sc).shape[1],
            reconstruct_shape=X_train_sc.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        ssl_task.fit(x_train=X_train_sc, pred_model=pred_model)

        # Fit VIME
        vime = Vime_Task(epochs=epochs, seed=seed)
        vime.fit(pred_model.extract_encoding(X_train_sc))

        results = {}

        ###########################################################################################################
        # Make predictions with the main model
        # Predict on the calibration set
        ypred_cal = pred_model.predict(X_cal_sc).reshape(-1, 1)

        # Predict on the test set
        ypred_test = pred_model.predict(X_test_sc).reshape(-1, 1)

        # Compute the main model errors
        main_model_errors = np.abs(ypred_test - y_test_sc)

        N = len(y_cal_sc)

        ###########################################################################################################

        ###########################################################################################################
        #
        # CRF
        #
        ###########################################################################################################

        # RUN CONFORMALIZED RESIDUAL FITTING  - CRF

        logging.info("Running Conformalized Residual Fitting")

        # calculate residuals
        res_y = np.abs(y_residual_sc - pred_model.predict(X_residual_sc).reshape(-1, 1))

        if log_res:
            res_y += 0.00001  # prevent log(0)
            res_y = np.log(res_y)


        # fir residual (normalization) model
        model_r = predictive_model(
            input_shape=X_residual_sc.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model_r.fit(X_residual_sc, res_y)

        # calculate q_yhat for the calibration set
        res_ypred_cal = model_r.predict(X_cal_sc).reshape(-1, 1)

        if log_res:
            res_ypred_cal = np.exp(res_ypred_cal)

        # compute critical nonconformity value
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal) / res_ypred_cal,
            np.ceil((N + 1) * (1 - alpha)) / N,
        )

        res_ypred_test = model_r.predict(X_test_sc).reshape(-1, 1)

        if log_res:
            res_ypred_test = np.exp(res_ypred_test)

        # predict with 1-alpha confidence - scaled by residual predictions
        lower_bound = ypred_test - (q_yhat * res_ypred_test.reshape(-1, 1))
        upper_bound = ypred_test + (q_yhat * res_ypred_test.reshape(-1, 1))

        # COMPUTE METRICS
        intervals = np.array(abs(upper_bound - lower_bound))

        if save_intervals:
             write_to_file(intervals, f"../dists/{dataset_name}_{i}_crf.p")

        residual_model_errors = np.abs(ypred_test - y_test_sc)

        _, avg_length = compute_interval_metrics(lower_bound, upper_bound, y_test_sc)
        avg_excess, proportion_excess = compute_excess(
            lower_bound, upper_bound, y_test_sc
        )
        avg_deficet, proportion_deficet = compute_deficet(
            lower_bound, upper_bound, y_test_sc
        )

        # Log results
        results["CRF"] = {
            "avg_length": avg_length,
            "avg_excess": avg_excess,
            "avg_deficet": avg_deficet, 
        }

        ###########################################################################################################

        ###########################################################################################################
        #
        # SSCP + VIME
        #
        ###########################################################################################################

        # RUN CRF + SSL USING VIME

        # fit a new model to predict residuals - augmented with SSL task
        logging.info("VIME SSL Running SSL conformalized residual fitting")

        # make predictions with the vime
        ssl_train = vime.predict(pred_model.extract_encoding(X_train_sc)).reshape(-1, 1)
        ssl_residual = vime.predict(pred_model.extract_encoding(X_residual_sc)).reshape(
            -1, 1
        )
        ssl_cal = vime.predict(pred_model.extract_encoding(X_cal_sc)).reshape(-1, 1)
        ssl_test = vime.predict(pred_model.extract_encoding(X_test_sc)).reshape(-1, 1)

        # scale the SSL outputs
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train)
        ssl_train_sc = scaler_ssl.transform(ssl_train)
        ssl_residual_sc = scaler_ssl.transform(ssl_residual)
        ssl_cal_sc = scaler_ssl.transform(ssl_cal)
        ssl_test_sc = scaler_ssl.transform(ssl_test)

        res_y = np.abs(y_residual_sc - pred_model.predict(X_residual_sc).reshape(-1, 1))

        if log_res:
            res_y += 0.00001  # prevent log(0)
            res_y = np.log(res_y)

        # Augment the input data for residual model (normalization model) with the SSL outputs
        X_train_r = np.hstack([X_residual_sc, ssl_residual_sc])

        # Fit residual/normalization model
        model_r = predictive_model(
            input_shape=X_train_r.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model_r.fit(X_train_r, res_y)

        # calculate q_yhat for the calibration set
        X_cal_r = np.hstack([X_cal_sc, ssl_cal_sc])

        # predict residuals for calibration set
        res_ypred_cal = model_r.predict(X_cal_r).reshape(-1, 1)

        if log_res:
            res_ypred_cal = np.exp(res_ypred_cal)

        # compute critical nonconformity value
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal) / res_ypred_cal,
            np.ceil((N + 1) * (1 - alpha)) / N,
        )

        X_test_r = np.hstack([X_test_sc, ssl_test_sc])
        # predict residuals for test set
        res_ypred_test = model_r.predict(X_test_r).reshape(-1, 1)

        if log_res:
            res_ypred_test = np.exp(res_ypred_test)

        # predict with 1-alpha confidence - scaled by residual predictions
        lower_bound = ypred_test - (q_yhat * res_ypred_test.reshape(-1, 1))
        upper_bound = ypred_test + (q_yhat * res_ypred_test.reshape(-1, 1))

        residual_model_errors = np.abs(ypred_test - y_test_sc)

        # COMPUTE METRICS
        intervals = np.array(abs(upper_bound - lower_bound))
        if save_intervals:
            write_to_file(intervals, f"../dists/{dataset_name}_{i}_vime.p")

        _, avg_length = compute_interval_metrics(lower_bound, upper_bound, y_test_sc)
        avg_excess, proportion_excess = compute_excess(
            lower_bound, upper_bound, y_test_sc
        )
        avg_deficet, proportion_deficet = compute_deficet(
            lower_bound, upper_bound, y_test_sc
        )

        # Log results
        results["CRF+VIME"] = {
            "avg_length": avg_length,
            "avg_excess": avg_excess,
            "avg_deficet": avg_deficet, 
        }

        ###########################################################################################################
        #
        # SSCP + AE
        #
        ###########################################################################################################

        # RUN CRF + SSL USING AUTOENCODER

        # fit a new model to predict residuals - augmented with SSL task
        logging.info("AE SSL Running SSL conformalized residual fitting")

        # make predictions with the autoencoder
        ssl_train = ssl_task.predict(X_train_sc).reshape(-1, 1)
        ssl_residual = ssl_task.predict(X_residual_sc).reshape(-1, 1)
        ssl_cal = ssl_task.predict(X_cal_sc).reshape(-1, 1)
        ssl_test = ssl_task.predict(X_test_sc).reshape(-1, 1)

        # scale the SSL outputs
        scaler_ssl = StandardScaler()
        scaler_ssl.fit(ssl_train)
        ssl_train_sc = scaler_ssl.transform(ssl_train)
        ssl_residual_sc = scaler_ssl.transform(ssl_residual)
        ssl_cal_sc = scaler_ssl.transform(ssl_cal)
        ssl_test_sc = scaler_ssl.transform(ssl_test)

        res_y = np.abs(y_residual_sc - pred_model.predict(X_residual_sc).reshape(-1, 1))

        if log_res:
            res_y += 0.00001  # prevent log(0)
            res_y = np.log(res_y)

        # Augment the input data for residual model (normalization model) with the SSL outputs
        X_train_r = np.hstack([X_residual_sc, ssl_residual_sc])

        # Fit residual/normalization model
        model_r = predictive_model(
            input_shape=X_train_r.shape[1],
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
        )
        model_r.fit(X_train_r, res_y)

        # calculate q_yhat for the calibration set
        X_cal_r = np.hstack([X_cal_sc, ssl_cal_sc])

        # predict residuals for calibration set
        res_ypred_cal = model_r.predict(X_cal_r).reshape(-1, 1)

        if log_res:
            res_ypred_cal = np.exp(res_ypred_cal)

        # compute critical nonconformity value
        q_yhat = np.quantile(
            np.abs(y_cal_sc - ypred_cal) / res_ypred_cal,
            np.ceil((N + 1) * (1 - alpha)) / N,
        )

        X_test_r = np.hstack([X_test_sc, ssl_test_sc])
        # predict residuals for test set
        res_ypred_test = model_r.predict(X_test_r).reshape(-1, 1)

        if log_res:
            res_ypred_test = np.exp(res_ypred_test)

        # predict with 1-alpha confidence - scaled by residual predictions
        lower_bound = ypred_test - (q_yhat * res_ypred_test.reshape(-1, 1))
        upper_bound = ypred_test + (q_yhat * res_ypred_test.reshape(-1, 1))

        # COMPUTE METRICS
        residual_model_errors = np.abs(ypred_test - y_test_sc)

        _, avg_length = compute_interval_metrics(lower_bound, upper_bound, y_test_sc)
        avg_excess, proportion_excess = compute_excess(
            lower_bound, upper_bound, y_test_sc
        )
        avg_deficet, proportion_deficet = compute_deficet(
            lower_bound, upper_bound, y_test_sc
        )

        results["CRF+AE"] = {
            "avg_length": avg_length,
            "avg_excess": avg_excess,
            "avg_deficet": avg_deficet, 
        }


        crf_results_list.append(results)

        seed += 1

    # Process results
    final_crf = process_results(crf_results_list)
    print(f"CRF: {final_crf}")

    # Log to wandb
    wandb.log({"CRF": final_crf})

    end = time.time()
    print(f"TIME TAKEN = {end - start}")
