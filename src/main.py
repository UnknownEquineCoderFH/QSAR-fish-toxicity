from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple
from pathlib import Path

import polars as pl
import numpy as np
import sklearn.model_selection as sk
import sklearn.linear_model as lm
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
    mean_absolute_percentage_error as mape,
)
import plotly.offline as plt
import plotly.express as px


class Columns(StrEnum):
    """
    The columns in the dataset.
    """

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
    """
    Splits the data into a training and test set.

    The test set is 10% of the data. (by default)

    Returns a tuple of four dataframes, the first two are the training set and the last two are the test set.
    """

    return sk.train_test_split(
        df.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
            Columns.MOLECULAR_PROPERTIES,
        ),
        df.select(Columns.QUANTITATIVE_RESPONSE),
        test_size=ratio,
    )  # type: ignore


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
    """
    Errors computed from a linear regression.
    """

    mae: np.floating
    rmse: np.floating
    mape: np.floating


def errors(real: np.ndarray, predicted: np.ndarray) -> Errors:
    """
    Computes the errors for the given real and predicted values.
    """

    return Errors(
        mae=mae(real, predicted),
        rmse=np.sqrt(mse(real, predicted)),
        mape=mape(real, predicted),
    )


class Summary(NamedTuple):
    """
    A summary of the errors for the train and test set.
    """

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
    Computes the errors for the given model and produces a Summary.
    """
    return Summary(
        train=errors(
            predicted=model.predict(x_train.to_numpy()),
            real=y_train.to_numpy(),
        ),
        test=errors(
            predicted=model.predict(x_test.to_numpy()),
            real=y_test.to_numpy(),
        ),
    )


def plot(summaries: list[Summary]) -> None:
    """
    Plot how the MAE changes by increasing the complexity of the model.

    Plots two lines one the same graph, one for test (green) and one for train (orange).
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
    # Generate the training and test set.
    x_train, x_test, y_train, y_test = split(query().collect())

    # 1-D Linear regression model: information indices to quantitative response.
    model = polynomial_model(
        x_train.select(Columns.INFORMATION_INDICES),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        linear=True,
    )

    # Compute the errors for the train and test set.
    one_d_summary = summary(
        model,
        x_train.select(Columns.INFORMATION_INDICES),
        x_test.select(Columns.INFORMATION_INDICES),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        y_test.select(Columns.QUANTITATIVE_RESPONSE),
    )

    # Log the errors.
    print(f"Summary for one dimension: {one_d_summary}")

    # 2-D Linear regression model: information indices and matrix descriptors to quantitative response.
    model = polynomial_model(
        x_train.select(Columns.INFORMATION_INDICES, Columns.MATRIX_DESCRIPTORS),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
    )

    # Compute the errors for the train and test set.
    two_d_summary = summary(
        model,
        x_train.select(Columns.INFORMATION_INDICES, Columns.MATRIX_DESCRIPTORS),
        x_test.select(Columns.INFORMATION_INDICES, Columns.MATRIX_DESCRIPTORS),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
        y_test.select(Columns.QUANTITATIVE_RESPONSE),
    )

    # Log the errors.
    print(f"Summary for two dimensions: {two_d_summary}")

    # 3-D Linear regression model: information indices, matrix descriptors and autocorrelation
    # to quantitative response.
    model = polynomial_model(
        x_train.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
        ),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
    )

    # Compute the errors for the train and test set.
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

    # Log the errors.
    print(f"Summary for three dimensions: {three_d_summary}")

    # 4-D Linear regression model: information indices, matrix descriptors,
    # autocorrelation and molecular properties to quantitative response.
    model = polynomial_model(
        x_train.select(
            Columns.INFORMATION_INDICES,
            Columns.MATRIX_DESCRIPTORS,
            Columns.AUTOCORRELATION,
            Columns.MOLECULAR_PROPERTIES,
        ),
        y_train.select(Columns.QUANTITATIVE_RESPONSE),
    )

    # Compute the errors for the train and test set.
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

    # Log the errors.
    print(f"Summary for four dimensions: {four_d_summary}")

    # Plot the errors.
    plot([one_d_summary, two_d_summary, three_d_summary, four_d_summary])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
