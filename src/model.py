"""
model.py
--------
Responsible for preprocessing, training, evaluating, and predicting
luxury watch prices using an XGBoost pipeline.

Usage:
    from src.model import WatchPriceModel

    # Training
    model = WatchPriceModel()
    model.fit(df)
    model.save("models/watch_price_model.joblib")

    # Prediction (e.g. in Streamlit) — all nine arguments are required
    model = WatchPriceModel()
    model.load("models/watch_price_model.joblib")
    price = model.predict(
        brand="Rolex", model_name="Daytona", case_material="Steel",
        condition="Very good", movement="Automatic", bracelet_material="Steel",
        sex="Men's watch/Unisex", size=40, yop=2018,
    )
"""

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

import category_encoders as ce
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class WatchPriceModel:
    """
    Full modelling pipeline for luxury watch price prediction.

    The pipeline handles:
        - Log transformation of the target variable (price)
        - Filling missing categoricals with 'Unknown'
        - Native XGBoost categorical handling for low-cardinality columns
        - Target encoding for high-cardinality columns (brand, model),
          fitted inside cross-validation folds to prevent data leakage
        - Hyperparameter tuning via RandomizedSearchCV
        - Model persistence via joblib
    """

    # Low-cardinality categoricals handled natively by XGBoost
    _LOW_CARD_COLS = [
        "movement", "case_material", "bracelet_material", "condition", "sex"
    ]

    # High-cardinality categoricals encoded via TargetEncoder
    _HIGH_CARD_COLS = ["brand", "model"]

    # Categorical columns that receive 'Unknown' for missing values
    _CATEGORICAL_COLS = _LOW_CARD_COLS + _HIGH_CARD_COLS

    # Plausible numeric input ranges for predict(). These mirror the
    # validation bounds enforced by WatchDataCleaner on the training data
    # (_clean_size: 20-60 mm, _clean_yop: 1500 to current year). Values
    # outside these ranges were treated as missing during training, so
    # accepting them at inference time would create a distribution shift.
    _SIZE_RANGE_MM = (20, 60)
    _YOP_MIN = 1500

    def __init__(self):
        """Initialize an untrained model with empty pipeline and data slots."""
        self._pipeline: Pipeline | None = None
        self._ridge_pipeline: Pipeline | None = None
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None
        self._watch_data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, test_size: float = 0.2,
            n_iter: int = 25, random_state: int = 42) -> None:
        """
        Preprocess the data and train the XGBoost pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Cleaned DataFrame from WatchDataCleaner.clean(). Must contain
            all feature columns and a strictly positive, non-missing
            'price' column.
        test_size : float
            Fraction of data held out for final evaluation. Must be
            strictly between 0 and 1. Default 0.2.
        n_iter : int
            Number of hyperparameter combinations passed to
            RandomizedSearchCV. Each combination is evaluated with 5-fold
            cross-validation, so the total number of model fits is
            n_iter * 5, plus one final refit on the full training set
            (125 fits at the default). Must be at least 1. Default 25.
        random_state : int
            Random seed for reproducibility. Default 42.

        Raises
        ------
        KeyError
            If a required column is missing from data.
        ValueError
            If test_size or n_iter is out of range, or if 'price'
            contains missing or non-positive values.
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError(
                f"test_size must be strictly between 0 and 1, got {test_size!r}."
            )
        if not isinstance(n_iter, int) or isinstance(n_iter, bool) or n_iter < 1:
            raise ValueError(
                f"n_iter must be a positive integer, got {n_iter!r}."
            )

        required = self._CATEGORICAL_COLS + ["size", "yop", "price"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise KeyError(
                f"Input data is missing expected columns: {missing}. "
                "Run WatchDataCleaner.clean() first."
            )
        if data["price"].isna().any() or (data["price"] <= 0).any():
            raise ValueError(
                "Column 'price' must be strictly positive and non-missing "
                "for the log transform. Run WatchDataCleaner.clean() first."
            )

        self._watch_data = data.copy()
        prepared = self._preprocess(data)
        self._split(prepared, test_size, random_state)
        self._train(n_iter, random_state)

    def evaluate(self) -> dict:
        """
        Evaluate the trained model against both baselines on the test set.

        Returns
        -------
        dict
            Dictionary containing MAE, MAPE, and RMSE for all three models.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        y_dollars = np.exp(self._y_test)

        # --- Dummy baseline ---
        dummy = DummyRegressor(strategy="median")
        dummy.fit(self._X_train, self._y_train)
        dummy_preds = np.exp(dummy.predict(self._X_test))

        # --- Ridge baseline ---
        ridge_preds = np.exp(self._ridge_pipeline.predict(self._X_test))

        # --- XGBoost ---
        xgb_preds = np.exp(self._pipeline.predict(self._X_test))

        results = {
            "dummy": {
                "MAE":  mean_absolute_error(y_dollars, dummy_preds),
                "MAPE": mean_absolute_percentage_error(y_dollars, dummy_preds) * 100,
            },
            "ridge": {
                "MAE":  mean_absolute_error(y_dollars, ridge_preds),
                "MAPE": mean_absolute_percentage_error(y_dollars, ridge_preds) * 100,
            },
            "xgboost": {
                "MAE":  mean_absolute_error(y_dollars, xgb_preds),
                "MAPE": mean_absolute_percentage_error(y_dollars, xgb_preds) * 100,
                "RMSE": np.sqrt(mean_squared_error(y_dollars, xgb_preds)),
            },
        }

        self._print_evaluation(results)
        return results

    def plot_model_comparison(self, results: dict) -> None:
        """
        Bar chart comparing MAPE across Dummy, Ridge, and XGBoost.

        Parameters
        ----------
        results : dict
            Output from evaluate().
        """
        model_names = [
            "Dummy Baseline\n(Guess Median)",
            "Ridge Baseline\n(Linear)",
            "Final Tuned\nXGBoost Model",
        ]
        mapes = [
            results["dummy"]["MAPE"],
            results["ridge"]["MAPE"],
            results["xgboost"]["MAPE"],
        ]
        colors = ["#a6cee3", "#5cb8d1", "#1f78b4"]

        plt.figure(figsize=(10, 5))
        bars = plt.barh(model_names, mapes, color=colors)
        plt.xlabel("Mean Absolute Percentage Error (%) - Lower is Better",
                   fontsize=12)
        plt.title("Model Performance Comparison (MAPE)", fontsize=14)
        plt.gca().invert_yaxis()

        for bar in bars:
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%",
                va="center", ha="left", fontsize=12, fontweight="bold",
            )

        plt.tight_layout()
        plt.show()

    def plot_shap(self, instance_index: int = 0) -> None:
        """
        SHAP waterfall plot for a single test set observation.

        Parameters
        ----------
        instance_index : int
            Index of the observation in X_test to explain. Default 0.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        fitted_encoder = self._pipeline.named_steps["target_encoder"]
        fitted_xgb = self._pipeline.named_steps["xgb"]

        X_encoded = fitted_encoder.transform(self._X_test)
        for col in self._LOW_CARD_COLS:
            X_encoded[col] = X_encoded[col].astype("category")

        explainer = shap.TreeExplainer(fitted_xgb)
        shap_values = explainer(X_encoded)

        shap.plots.waterfall(shap_values[instance_index], max_display=10)

        actual = np.exp(self._y_test.iloc[instance_index])
        predicted = np.exp(self._pipeline.predict(
            self._X_test.iloc[[instance_index]]
        )[0])

        print(f"\nActual Price:    ${actual:,.2f}")
        print(f"Predicted Price: ${predicted:,.2f}")

    def plot_feature_importance(self) -> None:
        """Bar chart of XGBoost feature importances."""
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        fitted_xgb = self._pipeline.named_steps["xgb"]
        importance_df = (
            pd.DataFrame({
                "Feature": self._X_train.columns,
                "Importance": fitted_xgb.feature_importances_,
            })
            .sort_values("Importance", ascending=True)
        )

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"],
                 color="#ff77b4")
        plt.xlabel("Importance Score (Relative Weight)")
        plt.title("XGBoost Feature Importances: What Drives the Price?")
        plt.tight_layout()
        plt.show()

    def predict(self, brand: str, model_name: str, case_material: str,
                condition: str, movement: str, bracelet_material: str,
                sex: str, size: float | None, yop: int | None) -> float:
        """
        Predict the market price for a single watch.

        All categorical inputs are validated against the dataset the model
        was fitted on — use get_valid_options() to obtain the allowed
        values. Unknown values raise a ValueError instead of silently
        degrading the prediction. The literal value "Unknown" is always
        accepted: it is the placeholder the model itself uses for missing
        categoricals during training. In the rare case that a valid value
        occurs only in the held-out test split (and was therefore never
        seen by the model), it is treated as missing.

        Parameters
        ----------
        brand : str
            Watch brand, e.g. "Rolex". Unseen brands would fall back to
            the target encoder's global prior mean; they are rejected here.
        model_name : str
            Watch model, e.g. "Daytona". Same fallback/rejection as brand.
        case_material : str
            Case material, e.g. "Steel".
        condition : str
            Condition category, e.g. "Very good".
        movement : str
            Movement type, e.g. "Automatic".
        bracelet_material : str
            Bracelet material, e.g. "Steel".
        sex : str
            Target wearer category from the listing.
        size : float or None
            Case diameter in mm, between 20 and 60, or None if unknown
            (treated as missing by the model).
        yop : int or None
            Year of production, between 1500 and the current year, or
            None if unknown (treated as missing by the model).

        Returns
        -------
        float
            Estimated market price in USD.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded.
        ValueError
            If a categorical value is unknown or a numeric value lies
            outside its plausible range.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained or loaded. "
                               "Call fit() or load() first.")

        self._validate_predict_inputs(
            categorical_inputs={
                "brand":             brand,
                "model":             model_name,
                "case_material":     case_material,
                "condition":         condition,
                "movement":          movement,
                "bracelet_material": bracelet_material,
                "sex":               sex,
            },
            size=size,
            yop=yop,
        )

        user_data = pd.DataFrame({
            "brand":             [brand],
            "model":             [model_name],
            "case_material":     [case_material],
            "condition":         [condition],
            "size":              pd.array([size], dtype="Float64"),
            "movement":          [movement],
            "yop":               pd.array([yop], dtype="Int64"),
            "bracelet_material": [bracelet_material],
            "sex":               [sex],
        })

        # Anchor the category codes to the *training* categories. A plain
        # astype("category") on a single row would always yield code 0,
        # which XGBoost would silently misinterpret. Values not present in
        # the training split become NaN and are handled as missing.
        for col in self._LOW_CARD_COLS:
            user_data[col] = pd.Categorical(
                user_data[col],
                categories=self._X_train[col].cat.categories,
            )

        user_data = user_data[self._X_train.columns]
        log_price = self._pipeline.predict(user_data)[0]
        return float(np.exp(log_price))

    def get_valid_options(self) -> dict:
        """
        Return valid input options for the Streamlit UI, derived from
        the training data.

        Returns
        -------
        dict
            Keys are column names, values are sorted lists of valid options.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded.
        ValueError
            If 'size' or 'yop' contains no valid values, so no input
            range can be derived for the UI.
        """
        if self._watch_data is None:
            raise RuntimeError("Model not trained or loaded. "
                               "Call fit() or load() first.")

        size_values = self._watch_data["size"].dropna()
        yop_values = self._watch_data["yop"].dropna()
        if size_values.empty or yop_values.empty:
            raise ValueError(
                "Cannot derive size/yop input ranges: the column contains "
                "no valid (non-missing) values."
            )

        return {
            "brands":    self._watch_data["brand"].value_counts().index.tolist(),
            "models":    self._watch_data["model"].value_counts().index.tolist(),
            "cases":     self._watch_data["case_material"].value_counts().index.tolist(),
            "conditions": self._watch_data["condition"].value_counts().index.tolist(),
            "movements": self._watch_data["movement"].value_counts().index.tolist(),
            "bracelets": self._watch_data["bracelet_material"].value_counts().index.tolist(),
            "sexes":     self._watch_data["sex"].value_counts().index.tolist(),
            "size_range": (
                int(size_values.min()),
                int(size_values.max()),
            ),
            "yop_range": (
                int(yop_values.min()),
                int(yop_values.max()),
            ),
        }

    def save(self, filepath: str) -> None:
        """
        Save the trained pipeline and training data to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model, e.g. 'models/watch_price_model.joblib'.
        """
        if self._pipeline is None:
            raise RuntimeError("Nothing to save. Call fit() first.")

        joblib.dump({
            "pipeline":        self._pipeline,
            "ridge_pipeline":  self._ridge_pipeline,
            "X_train":         self._X_train,
            "y_train":         self._y_train,
            "X_test":          self._X_test,
            "y_test":          self._y_test,
            "watch_data":      self._watch_data,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load a previously saved pipeline from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        """
        checkpoint = joblib.load(filepath)
        self._pipeline        = checkpoint["pipeline"]
        self._ridge_pipeline  = checkpoint["ridge_pipeline"]
        self._X_train         = checkpoint["X_train"]
        self._y_train         = checkpoint["y_train"]
        self._X_test          = checkpoint["X_test"]
        self._y_test          = checkpoint["y_test"]
        self._watch_data      = checkpoint["watch_data"]
        print(f"Model loaded from {filepath}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation and fill missing categoricals."""
        df = data.copy()
        df["log_price"] = np.log(df["price"])
        df.drop(columns=["price"], inplace=True)
        df[self._CATEGORICAL_COLS] = df[self._CATEGORICAL_COLS].fillna("Unknown")
        return df

    def _validate_predict_inputs(self, categorical_inputs: dict,
                                 size: float | None,
                                 yop: int | None) -> None:
        """
        Validate one set of prediction inputs against the fitted dataset.

        Categorical values are checked against the same data source that
        get_valid_options() exposes to the UI, so every value offered
        there passes validation. Numeric values are checked against the
        plausibility bounds used during data cleaning.

        Parameters
        ----------
        categorical_inputs : dict
            Mapping of feature column name to the user-supplied value.
        size : float or None
            Case diameter in mm.
        yop : int or None
            Year of production.

        Raises
        ------
        ValueError
            If a categorical value does not occur in the fitted dataset,
            or a numeric value lies outside its plausible range.
        """
        for col, value in categorical_inputs.items():
            if value == "Unknown":
                # "Unknown" is the model's own placeholder for missing
                # categoricals (see _preprocess) and is always accepted.
                continue
            known = self._watch_data[col].dropna().unique()
            if value not in known:
                raise ValueError(
                    f"Unknown value {value!r} for '{col}'. "
                    "Use get_valid_options() to list the known values."
                )

        size_min, size_max = self._SIZE_RANGE_MM
        if size is not None and not size_min <= size <= size_max:
            raise ValueError(
                f"size must be between {size_min} and {size_max} mm "
                f"(or None), got {size!r}."
            )

        current_year = datetime.date.today().year
        if yop is not None and not self._YOP_MIN <= yop <= current_year:
            raise ValueError(
                f"yop must be between {self._YOP_MIN} and {current_year} "
                f"(or None), got {yop!r}."
            )

    def _split(self, data: pd.DataFrame, test_size: float,
               random_state: int) -> None:
        """Split into train/test and apply category dtypes."""
        X = data.drop(columns=["log_price"])
        y = data["log_price"]

        self._X_train, self._X_test, self._y_train, self._y_test = (
            train_test_split(X, y, test_size=test_size,
                             random_state=random_state)
        )

        for col in self._LOW_CARD_COLS:
            self._X_train[col] = self._X_train[col].astype("category")
            # Anchor the test set to the *training* categories so the
            # integer codes XGBoost consumes are identical in both frames.
            # Levels occurring only in the test split become NaN (missing).
            self._X_test[col] = pd.Categorical(
                self._X_test[col],
                categories=self._X_train[col].cat.categories,
            )

        # Fit Ridge baseline here so it shares the same train/test split
        self._fit_ridge_baseline()

    def _fit_ridge_baseline(self) -> None:
        """Fit the Ridge baseline pipeline on the training data."""
        preprocessor = ColumnTransformer(transformers=[
            ("num", SimpleImputer(strategy="median"), ["size", "yop"]),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot",  OneHotEncoder(handle_unknown="ignore",
                                         sparse_output=False)),
            ]), self._CATEGORICAL_COLS),
        ])

        self._ridge_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("ridge", Ridge()),
        ])
        self._ridge_pipeline.fit(self._X_train, self._y_train)

    def _train(self, n_iter: int, random_state: int) -> None:
        """Build and tune the XGBoost pipeline via RandomizedSearchCV."""
        xgb_model = xgb.XGBRegressor(
            enable_categorical=True,
            tree_method="hist",
            random_state=random_state,
            objective="reg:squarederror",
        )

        pipeline = Pipeline([
            ("target_encoder", ce.TargetEncoder(
                cols=self._HIGH_CARD_COLS,
                min_samples_leaf=10,
                smoothing=10,
            )),
            ("xgb", xgb_model),
        ])

        param_distributions = {
            "xgb__learning_rate":    [0.01, 0.05, 0.1, 0.15, 0.2],
            "xgb__max_depth":        [4, 5, 6, 7, 8, 9, 10],
            "xgb__n_estimators":     [100, 300, 500, 750, 1000],
            "xgb__subsample":        [0.7, 0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=5,
            verbose=2,
            random_state=random_state,
            n_jobs=1,
        )

        search.fit(self._X_train, self._y_train)
        self._pipeline = search.best_estimator_
        print("Best parameters found:")
        print(search.best_params_)

    @staticmethod
    def _print_evaluation(results: dict) -> None:
        """Print a formatted evaluation summary."""
        print("--- Baseline Benchmarks ---")
        print(f"Dummy  MAE: ${results['dummy']['MAE']:,.2f}  | "
              f"MAPE: {results['dummy']['MAPE']:.1f}%")
        print(f"Ridge  MAE: ${results['ridge']['MAE']:,.2f}  | "
              f"MAPE: {results['ridge']['MAPE']:.1f}%")
        print("\n" + "=" * 30)
        print("--- Final XGBoost Evaluation ---")
        print("=" * 30)
        print(f"MAE:  ${results['xgboost']['MAE']:,.2f}")
        print(f"MAPE: {results['xgboost']['MAPE']:.1f}%")
        print(f"RMSE: ${results['xgboost']['RMSE']:,.2f}")