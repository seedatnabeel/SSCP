# third party
# stdlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def GetDataset(name, base_path, seeded=False):
    """Load a dataset

    # Code adapted from https://github.com/yromano/cqr

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path to datasets"

    Returns
    -------
    X : features (n X p)
    y : labels (n)

    """

    if name == "star":
        df = pd.read_csv(base_path + "STAR.csv")
        df.loc[df["gender"] == "female", "gender"] = 0
        df.loc[df["gender"] == "male", "gender"] = 1

        df.loc[df["ethnicity"] == "cauc", "ethnicity"] = 0
        df.loc[df["ethnicity"] == "afam", "ethnicity"] = 1
        df.loc[df["ethnicity"] == "asian", "ethnicity"] = 2
        df.loc[df["ethnicity"] == "hispanic", "ethnicity"] = 3
        df.loc[df["ethnicity"] == "amindian", "ethnicity"] = 4
        df.loc[df["ethnicity"] == "other", "ethnicity"] = 5

        df.loc[df["stark"] == "regular", "stark"] = 0
        df.loc[df["stark"] == "small", "stark"] = 1
        df.loc[df["stark"] == "regular+aide", "stark"] = 2

        df.loc[df["star1"] == "regular", "star1"] = 0
        df.loc[df["star1"] == "small", "star1"] = 1
        df.loc[df["star1"] == "regular+aide", "star1"] = 2

        df.loc[df["star2"] == "regular", "star2"] = 0
        df.loc[df["star2"] == "small", "star2"] = 1
        df.loc[df["star2"] == "regular+aide", "star2"] = 2

        df.loc[df["star3"] == "regular", "star3"] = 0
        df.loc[df["star3"] == "small", "star3"] = 1
        df.loc[df["star3"] == "regular+aide", "star3"] = 2

        df.loc[df["lunchk"] == "free", "lunchk"] = 0
        df.loc[df["lunchk"] == "non-free", "lunchk"] = 1

        df.loc[df["lunch1"] == "free", "lunch1"] = 0
        df.loc[df["lunch1"] == "non-free", "lunch1"] = 1

        df.loc[df["lunch2"] == "free", "lunch2"] = 0
        df.loc[df["lunch2"] == "non-free", "lunch2"] = 1

        df.loc[df["lunch3"] == "free", "lunch3"] = 0
        df.loc[df["lunch3"] == "non-free", "lunch3"] = 1

        df.loc[df["schoolk"] == "inner-city", "schoolk"] = 0
        df.loc[df["schoolk"] == "suburban", "schoolk"] = 1
        df.loc[df["schoolk"] == "rural", "schoolk"] = 2
        df.loc[df["schoolk"] == "urban", "schoolk"] = 3

        df.loc[df["school1"] == "inner-city", "school1"] = 0
        df.loc[df["school1"] == "suburban", "school1"] = 1
        df.loc[df["school1"] == "rural", "school1"] = 2
        df.loc[df["school1"] == "urban", "school1"] = 3

        df.loc[df["school2"] == "inner-city", "school2"] = 0
        df.loc[df["school2"] == "suburban", "school2"] = 1
        df.loc[df["school2"] == "rural", "school2"] = 2
        df.loc[df["school2"] == "urban", "school2"] = 3

        df.loc[df["school3"] == "inner-city", "school3"] = 0
        df.loc[df["school3"] == "suburban", "school3"] = 1
        df.loc[df["school3"] == "rural", "school3"] = 2
        df.loc[df["school3"] == "urban", "school3"] = 3

        df.loc[df["degreek"] == "bachelor", "degreek"] = 0
        df.loc[df["degreek"] == "master", "degreek"] = 1
        df.loc[df["degreek"] == "specialist", "degreek"] = 2
        df.loc[df["degreek"] == "master+", "degreek"] = 3

        df.loc[df["degree1"] == "bachelor", "degree1"] = 0
        df.loc[df["degree1"] == "master", "degree1"] = 1
        df.loc[df["degree1"] == "specialist", "degree1"] = 2
        df.loc[df["degree1"] == "phd", "degree1"] = 3

        df.loc[df["degree2"] == "bachelor", "degree2"] = 0
        df.loc[df["degree2"] == "master", "degree2"] = 1
        df.loc[df["degree2"] == "specialist", "degree2"] = 2
        df.loc[df["degree2"] == "phd", "degree2"] = 3

        df.loc[df["degree3"] == "bachelor", "degree3"] = 0
        df.loc[df["degree3"] == "master", "degree3"] = 1
        df.loc[df["degree3"] == "specialist", "degree3"] = 2
        df.loc[df["degree3"] == "phd", "degree3"] = 3

        df.loc[df["ladderk"] == "level1", "ladderk"] = 0
        df.loc[df["ladderk"] == "level2", "ladderk"] = 1
        df.loc[df["ladderk"] == "level3", "ladderk"] = 2
        df.loc[df["ladderk"] == "apprentice", "ladderk"] = 3
        df.loc[df["ladderk"] == "probation", "ladderk"] = 4
        df.loc[df["ladderk"] == "pending", "ladderk"] = 5
        df.loc[df["ladderk"] == "notladder", "ladderk"] = 6

        df.loc[df["ladder1"] == "level1", "ladder1"] = 0
        df.loc[df["ladder1"] == "level2", "ladder1"] = 1
        df.loc[df["ladder1"] == "level3", "ladder1"] = 2
        df.loc[df["ladder1"] == "apprentice", "ladder1"] = 3
        df.loc[df["ladder1"] == "probation", "ladder1"] = 4
        df.loc[df["ladder1"] == "noladder", "ladder1"] = 5
        df.loc[df["ladder1"] == "notladder", "ladder1"] = 6

        df.loc[df["ladder2"] == "level1", "ladder2"] = 0
        df.loc[df["ladder2"] == "level2", "ladder2"] = 1
        df.loc[df["ladder2"] == "level3", "ladder2"] = 2
        df.loc[df["ladder2"] == "apprentice", "ladder2"] = 3
        df.loc[df["ladder2"] == "probation", "ladder2"] = 4
        df.loc[df["ladder2"] == "noladder", "ladder2"] = 5
        df.loc[df["ladder2"] == "notladder", "ladder2"] = 6

        df.loc[df["ladder3"] == "level1", "ladder3"] = 0
        df.loc[df["ladder3"] == "level2", "ladder3"] = 1
        df.loc[df["ladder3"] == "level3", "ladder3"] = 2
        df.loc[df["ladder3"] == "apprentice", "ladder3"] = 3
        df.loc[df["ladder3"] == "probation", "ladder3"] = 4
        df.loc[df["ladder3"] == "noladder", "ladder3"] = 5
        df.loc[df["ladder3"] == "notladder", "ladder3"] = 6

        df.loc[df["tethnicityk"] == "cauc", "tethnicityk"] = 0
        df.loc[df["tethnicityk"] == "afam", "tethnicityk"] = 1

        df.loc[df["tethnicity1"] == "cauc", "tethnicity1"] = 0
        df.loc[df["tethnicity1"] == "afam", "tethnicity1"] = 1

        df.loc[df["tethnicity2"] == "cauc", "tethnicity2"] = 0
        df.loc[df["tethnicity2"] == "afam", "tethnicity2"] = 1

        df.loc[df["tethnicity3"] == "cauc", "tethnicity3"] = 0
        df.loc[df["tethnicity3"] == "afam", "tethnicity3"] = 1
        df.loc[df["tethnicity3"] == "asian", "tethnicity3"] = 2

        df = df.dropna()

        grade = df["readk"] + df["read1"] + df["read2"] + df["read3"]
        grade += df["mathk"] + df["math1"] + df["math2"] + df["math3"]

        names = df.columns
        target_names = names[8:16]
        data_names = np.concatenate((names[0:8], names[17:]))
        seed=42
        X = df.loc[:, data_names].values
        y = grade.values

    if name == "facebook_1":
        df = pd.read_csv(base_path + "facebook/Features_Variant_1.csv")
        seed=42
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values

    if name == "facebook_2":
        df = pd.read_csv(base_path + "facebook/Features_Variant_2.csv")
        seed=42
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values


    if name == "blog_data":
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + "blogData_train.csv", header=None)
        seed=0
        X = df.iloc[:, 0:280].values
        y = df.iloc[:, -1].values
        

    if name == "concrete":
        dataset = np.loadtxt(
            open(base_path + "Concrete_Data.csv", "rb"), delimiter=",", skiprows=1
        )
        seed=0
        X = dataset[:, :-1]
        y = dataset[:, -1:]

    if name == "bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df = pd.read_csv(base_path + "bike_train.csv")

        # # seperating season as per values. this is bcoz this will enhance features.
        season = pd.get_dummies(df["season"], prefix="season")
        df = pd.concat([df, season], axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather = pd.get_dummies(df["weather"], prefix="weather")
        df = pd.concat([df, weather], axis=1)

        # # # now can drop weather and season.
        df.drop(["season", "weather"], inplace=True, axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df["year"] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df["year"] = df["year"].map({2011: 0, 2012: 1})

        df.drop("datetime", axis=1, inplace=True)
        df.drop(["casual", "registered"], axis=1, inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        seed=0
        X = df.drop("count", axis=1).values
        y = df["count"].values

    if name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(
            base_path + "communities_attributes.csv", delim_whitespace=True
        )
        data = pd.read_csv(base_path + "communities.data", names=attrib["attributes"])
        data = data.drop(
            columns=["state", "county", "community", "communityname", "fold"], axis=1
        )

        data = data.replace("?", np.nan)

        # Impute mean values for samples with missing values
        from sklearn.preprocessing import Imputer

        imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

        imputer = imputer.fit(data[["OtherPerCap"]])
        data[["OtherPerCap"]] = imputer.transform(data[["OtherPerCap"]])
        data = data.dropna(axis=1)
        seed=0
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if seeded:
        return X, y, seed
    else:
        return X, y

    


def process_data(
    X, y, seed
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    > We split the data into train, residual, and test sets. We then scale the features and labels

    Args:
      X: the full dataset
      y: the labels
      seed: the random seed used to split the data into train, residual, and test sets
    """

    # train_full vs test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # train vs residual
    X_train, X_residual, y_train, y_residual = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    # train vs cal
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed
    )

    print(X_train.shape, X_residual.shape, X_cal.shape, X_test.shape, X.shape)
    
    # scale the features and labels
    scaler_feats = StandardScaler()
    scaler_feats.fit(X_train)

    # transform the features
    X_train_sc = scaler_feats.transform(X_train)
    X_residual_sc = scaler_feats.transform(X_residual)
    X_cal_sc = scaler_feats.transform(X_cal)
    X_test_sc = scaler_feats.transform(X_test)

    # scale the labels
    scaler_labels = StandardScaler()
    scaler_labels.fit(y_train.reshape(-1, 1))

    # transform the labels
    mean_ytrain = np.mean(np.abs(y_train))
    y_train_sc = np.squeeze(y_train) / mean_ytrain
    y_residual_sc = np.squeeze(y_residual) / mean_ytrain
    y_cal_sc = np.squeeze(y_cal) / mean_ytrain
    y_test_sc = np.squeeze(y_test) / mean_ytrain

    # reshape the labels
    y_train_sc = y_train_sc.reshape(-1, 1)
    y_residual_sc = y_residual_sc.reshape(-1, 1)
    y_cal_sc = y_cal_sc.reshape(-1, 1)
    y_test_sc = y_test_sc.reshape(-1, 1)

    return (
        X_train_sc,
        X_residual_sc,
        X_cal_sc,
        X_test_sc,
        y_train_sc,
        y_residual_sc,
        y_cal_sc,
        y_test_sc,
    )
