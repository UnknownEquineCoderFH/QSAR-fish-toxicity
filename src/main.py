from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple
from pathlib import Path

import polars as pl
import numpy as np
import sklearn.model_selection as sk
import sklearn.linear_model as lm
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import plotly.offline as plt
import plotly.express as px


class Columns(StrEnum):
    INFORMATION_INDICES = "CIC0"
    MATRIX_DESCRIPTORS = "SM1_Dz(Z)"
    AUTOCORRELATION = "GATS1i"
    MOLECULAR_PROPERTIES = "MLOGP"
    QUANTITATIVE_RESPONSE = "LC50"


def query(path: str | Path = "data/qsar_fish_toxicity.csv") -> pl.LazyFrame:
    """
    Describes the dataset query and casts the columns to the correct types.

    Numeric columns are cast to Float32 to save on memory.

    Returns a lazy frame over the data.
    """
    return (
        pl.scan_csv(
            path,
            has_header=True,
            sep=";",
            dtypes={name: pl.Float32 for name in Columns},
        )
        .select(list(Columns))
        .drop_nulls()
    )


def split(
    df: pl.DataFrame,
    ratio: float = 0.1,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return sk.train_test_split(
        df.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
            Columns.MOLECULAR_PROPERTIES,
        ),
        df.select(Columns.QUANTITATIVE_RESPONSE),
        test_size=ratio,
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    """
    Computes the mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    """
    Computes the root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> np.floating:
    """
    Computes the mean absolute percentage error.
    """
    return mean_absolute_percentage_error(y_true, y_pred)


def polynomial_model(
    _from: pl.DataFrame, _to: pl.DataFrame, linear: bool = False
) -> lm.LinearRegression:
    """
    Creates a linear regression model from the given data.
    """
    model = lm.LinearRegression()

    model.fit(
        _from.to_numpy().reshape(-1, 1) if linear else _from.to_numpy(),
        _to.to_numpy(),
    )

    return model


class Errors(NamedTuple):
    mae: np.floating
    rmse: np.floating
    mape: np.floating


def errors(y_true: np.ndarray, y_pred: np.ndarray) -> Errors:
    return Errors(
        mae=mae(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mape=mape(y_true, y_pred),
    )


class Summary(NamedTuple):
    train: Errors
    test: Errors


def summary(
    model: lm.LinearRegression,
    x_train: pl.DataFrame,
    x_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
) -> Summary:
    """
    Computes the errors for the given model.
    """
    return Summary(
        train=errors(
            y_pred=model.predict(x_train.to_numpy()),
            y_true=y_train.to_numpy(),
        ),
        test=errors(
            y_pred=model.predict(x_test.to_numpy()),
            y_true=y_test.to_numpy(),
        ),
    )


def plot(summaries: list[Summary]) -> None:
    """
    Plot how the MAE changes by increasing the complexity of the model.

    Plot two lines one the same graph, one for test (blue) and one for train (orange).
    """

    fig = px.line(
        x=list(range(1, len(summaries) + 1)),
        y=[summary.test.mae for summary in summaries],
        labels={"x": "Model complexity", "y": "MAE"},
        title="MAE by model complexity",
    )

    fig.add_scatter(
        x=list(range(1, len(summaries) + 1)),
        y=[summary.train.mae for summary in summaries],
        mode="lines",
        name="Train",
    )

    fig.add_scatter(
        x=list(range(1, len(summaries) + 1)),
        y=[summary.test.mae for summary in summaries],
        mode="lines",
        name="Test",
    )

    plt.plot(fig)


def main() -> int:
    q = query()

    x_train, x_test, y_train, y_test = split(q.collect())

    # Train a linear regression model that tries to predict the quantitative response based on the information indices.
    model = polynomial_model(
        x_train.select(Columns.INFORMATION_INDICES),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        linear=True,
    )

    one_d_summary = summary(
        model,
        x_train.select(Columns.INFORMATION_INDICES),
        x_test.select(Columns.INFORMATION_INDICES),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        y_test.select(Columns.QUANTITATIVE_RESPONSE),
    )

    print(f"Summary for one dimension: {one_d_summary}")

    # Train a linear regression model that tries to predict the quantitative response based on the information indice and matrix descriptors.
    model = polynomial_model(
        x_train.select(Columns.INFORMATION_INDICES, Columns.MATRIX_DESCRIPTORS),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
    )

    two_d_summary = summary(
        model,
        x_train.select(Columns.INFORMATION_INDICES, Columns.MATRIX_DESCRIPTORS),
        x_test.select(Columns.INFORMATION_INDICES, Columns.MATRIX_DESCRIPTORS),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        y_test.select(Columns.QUANTITATIVE_RESPONSE),
    )

    print(f"Summary for two dimensions: {two_d_summary}")

    # Train a linear regression model that tries to predict the quantitative response based on the information indice, matrix descriptors and autocorrelation.

    model = polynomial_model(
        x_train.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
        ),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
    )

    three_d_summary = summary(
        model,
        x_train.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
        ),
        x_test.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
        ),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        y_test.select(Columns.QUANTITATIVE_RESPONSE),
    )

    print(f"Summary for three dimensions: {three_d_summary}")

    # Train a linear regression model that tries to predict the quantitative response based on the information indice, matrix descriptors, autocorrelation and molecular properties.

    model = polynomial_model(
        x_train.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
            Columns.MOLECULAR_PROPERTIES,
        ),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
    )

    four_d_summary = summary(
        model,
        x_train.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
            Columns.MOLECULAR_PROPERTIES,
        ),
        x_test.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
            Columns.MOLECULAR_PROPERTIES,
        ),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        y_test.select(Columns.QUANTITATIVE_RESPONSE),
    )

    print(f"Summary for four dimensions: {four_d_summary}")

    plot([one_d_summary, two_d_summary, three_d_summary, four_d_summary])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
