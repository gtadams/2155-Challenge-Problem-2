import os
os.environ.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})

from utils_public import *
from feature_engineer import *
import numpy as np
import pandas as pd
import time
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.base import clone

from autogluon.tabular import TabularPredictor
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def advisor_train(grids, ratings, FE_fn, advisor):
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor]) #gets subset of the dataset rated by advisor
    X_train, X_val, y_train, y_val = train_test_split(
        grids_subset, ratings_subset, test_size=0.2, random_state=42, shuffle=True
    )

    # Feature engineering
    # Ensure column names are strings matching expected format
    n_features = FE_fn(X_train).shape[1]
    col_names = [str(i) for i in range(n_features)]
    grids_fa_train = pd.DataFrame(FE_fn(X_train), columns=col_names).astype(np.float32)
    grids_fa_val = pd.DataFrame(FE_fn(X_val), columns=col_names).astype(np.float32)

    # Build TabularPredictor-compatible DataFrames
    all_train = grids_fa_train.copy()
    all_train["label"] = y_train.astype(np.float32)
    all_val = grids_fa_val.copy()
    all_val["label"] = y_val.astype(np.float32)

    # Train with validation as tuning_data
    predictor = TabularPredictor(
        label='label',
        problem_type='regression',
        verbosity=4,
        eval_metric='r2'
    )
    predictor.fit(
        train_data=all_train,
        tuning_data=all_val,
        presets='medium',
        time_limit=60,
        num_bag_folds=5,
        use_bag_holdout=True
    )
    predictor.delete_models(models_to_keep='best')
    predictor.fit_summary()

    # Compute and display feature importance
    fi = predictor.feature_importance(all_val)
    if isinstance(fi, pd.Series):
        fi_sorted = fi.sort_values(ascending=False)
    else:
        # If fi is a DataFrame, sort by the 'importance' column
        fi_sorted = fi.sort_values(by='importance', ascending=False)['importance']
    print("\nTop 30 Feature Importances:")
    print(fi_sorted.head(30))
    print(f"\nTotal features: {len(fi_sorted)}")
    print(f"Features with importance > 0: {(fi_sorted > 0).sum()}")

    # Identify features to drop: feature_id >= 50 AND low importance
    # Define low importance threshold (e.g., zero or negative)
    low_importance_threshold = 0.003
    features_to_drop = []
    for feat_name, importance in fi_sorted.items():
        try:
            feat_id = int(feat_name)
            if feat_id >= 50 and importance <= low_importance_threshold:
                features_to_drop.append(feat_name)
        except ValueError:
            # Skip non-integer feature names
            continue

    print(f"\nFeatures to drop (ID >= 50 and importance <= {low_importance_threshold}): {len(features_to_drop)}")
    print(features_to_drop[:20] if len(features_to_drop) > 20 else features_to_drop)

    # Create filtered datasets to reduce feature count
    all_train_filtered = all_train.drop(columns=features_to_drop, errors='ignore')
    all_val_filtered = all_val.drop(columns=features_to_drop, errors='ignore')

    print(f"\nOriginal features: {all_train.shape[1] - 1}")  # -1 for label
    print(f"Filtered features: {all_train_filtered.shape[1] - 1}")

    # Re-train with reduced features
    predictor = TabularPredictor(
        label='label',
        problem_type='regression',
        verbosity=4,
        eval_metric='r2'
    )
    predictor.fit(
        train_data=all_train_filtered,
        tuning_data=all_val_filtered,
        presets='medium',
        time_limit=60,
        num_bag_folds=5,
        use_bag_holdout=True
    )
    predictor.delete_models(models_to_keep='best')
    predictor.fit_summary()

    return predictor

def advisor_train_cnn(grids, ratings, FE_fn, advisor):
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor]) #gets subset of the dataset rated by advisor
    X_train, X_val, y_train, y_val = train_test_split(
        grids_subset, ratings_subset, test_size=0.2, random_state=42, shuffle=True
    )

    # Feature engineering
    # Ensure column names are strings matching expected format
    grids_train = pd.DataFrame(X_train).astype(np.float32)
    grids_val = pd.DataFrame(X_val).astype(np.float32)

    # Build TabularPredictor-compatible DataFrames
    all_train = grids_train.copy()
    all_train["label"] = y_train.astype(np.float32)
    all_val = grids_val.copy()
    all_val["label"] = y_val.astype(np.float32)

    # Train with validation as tuning_data
    # Use a small Keras CNN wrapped to mimic the TabularPredictor fit/predict interface

    class CNNRegressorWrapper:
        def __init__(self, epochs=30, batch_size=32, verbose=2):
            self.epochs = epochs
            self.batch_size = batch_size
            self.verbose = verbose
            self.model = None
            self._input_shape = None

        def _prepare_array(self, df):
            # Accept DataFrame (with 'label') or numpy arrays
            if isinstance(df, pd.DataFrame):
                X = df.drop(columns=['label']).values.astype(np.float32)
                y = df['label'].values.astype(np.float32)
            else:
                # assume tuple/list (X, y) or numpy arrays; keep simple
                X = np.asarray(df[0], dtype=np.float32) if isinstance(df, (list, tuple)) else np.asarray(df, dtype=np.float32)
                y = np.asarray(df[1], dtype=np.float32) if isinstance(df, (list, tuple)) else None

            n_features = X.shape[1]
            side = int(np.sqrt(n_features))
            if side * side == n_features:
                Xr = X.reshape((-1, side, side, 1))
                input_shape = (side, side, 1)
            else:
                # fallback: treat as "tall" image
                Xr = X.reshape((-1, n_features, 1, 1))
                input_shape = (n_features, 1, 1)
            return Xr, y, input_shape

        def _build_model(self, input_shape):
            inp = layers.Input(shape=input_shape)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Flatten()(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dense(32, activation='relu')(x)
            out = layers.Dense(1, activation='linear')(x)
            model = models.Model(inputs=inp, outputs=out)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model

        def fit(self, train_data, tuning_data=None, presets=None, time_limit=None, num_bag_folds=None, use_bag_holdout=None):
            X_train, y_train, input_shape = self._prepare_array(train_data)
            self._input_shape = input_shape
            if tuning_data is not None:
                X_val, y_val, _ = self._prepare_array(tuning_data)
            else:
                X_val, y_val = None, None

            self.model = self._build_model(input_shape)
            cb = [callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
            self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=cb,
                verbose=self.verbose
            )
            return self

        def predict(self, X):
            # Accept DataFrame or numpy array
            if isinstance(X, pd.DataFrame):
                X = X.values.astype(np.float32)
            else:
                X = np.asarray(X, dtype=np.float32)
            n_features = X.shape[1]
            side = int(np.sqrt(n_features))
            if side * side == n_features:
                Xr = X.reshape((-1, side, side, 1))
            else:
                Xr = X.reshape((-1, n_features, 1, 1))
            preds = self.model.predict(Xr)
            return preds.squeeze()

        # no-op compatibility methods
        def delete_models(self, models_to_keep='best'):
            return None

        def fit_summary(self):
            if self.model is not None:
                self.model.summary()

    # instantiate wrapper (tune epochs/batch_size as needed)
    predictor = CNNRegressorWrapper(epochs=30, batch_size=32, verbose=2)
    predictor.fit(
        train_data=all_train,
        tuning_data=all_val,
        presets='medium',
        time_limit=60,
        num_bag_folds=5,
        use_bag_holdout=True
    )
    predictor.delete_models(models_to_keep='best')
    predictor.fit_summary()

    

    return predictor