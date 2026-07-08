from unittest.mock import ANY, Mock

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import ert
from ert.gui.plotting.ert_plots import HistogramPlot
from ert.gui.plotting.plot_api import EnsembleObject
from ert.gui.plotting.utils import PlotConfig, PlotContext


@pytest.fixture(
    params=[
        pytest.param(
            (
                [
                    EnsembleObject(
                        "ensemble_1",
                        "id",
                        False,
                        "experiment_1",
                        started_at="2012-12-10T00:00:00",
                    )
                ],
                [1],
            ),
        ),
        pytest.param(([], []), id="no_ensembles"),
    ]
)
def plot_context(request):
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = request.param[0]
    context.ensembles_color_indexes.return_value = request.param[1]
    context._log_scale = False
    title = "" + f"num_ensembles={len(request.param[0])}"
    context.plotConfig.return_value = PlotConfig(title=title)
    return context


@pytest.fixture(
    params=[
        pytest.param(pd.DataFrame([[0.1], [0.2], [0.3], [0.4], [0.5]]), id="float"),
        pytest.param(pd.DataFrame(), id="empty"),
        pytest.param(
            pd.DataFrame(["cat", "cat", "cat", "dog"] + ["fish"] * 10),
            id="categorical",
        ),
    ]
)
def ensemble_to_data_map(request, plot_context):
    if len(plot_context.ensembles()) == 0 and not request.param.empty:
        # Only test with empty ensemble list once
        pytest.skip()
    if not request.param.empty and request.param[0].dtype == "object":
        # categorial and logscale is nonsensical
        pytest.skip()
    return dict.fromkeys(plot_context.ensembles(), request.param)


@pytest.mark.mpl_image_compare(tolerance=10)
def test_histogram(plot_context: PlotContext, ensemble_to_data_map):
    figure = Figure()
    HistogramPlot().plot(
        figure,
        plot_context,
        ensemble_to_data_map,
        pd.DataFrame(),
        {},
        {},
    )
    return figure


def test_histogram_plot_for_constant_distribution(monkeypatch):
    # test that the histogram plot is called with the correct min and max values
    # when all the parameter values are the same
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = [
        EnsembleObject(
            "ensemble_1", "id", False, "experiment_1", started_at="2012-12-10T00:00:00"
        )
    ]
    context.ensembles_color_indexes.return_value = [1]
    context.log_scale = False
    title = "Histogram with same values"
    context.plotConfig.return_value = PlotConfig(title=title)
    value = 0
    data_map = dict.fromkeys(context.ensembles(), pd.DataFrame([10 * [value]]))
    min_value = value - 0.1
    max_value = value + 0.1
    figure = Figure()
    mock_plot_histogram = Mock()
    monkeypatch.setattr(
        ert.gui.plotting.ert_plots.histogram,
        "_plotHistogram",
        mock_plot_histogram,
    )
    HistogramPlot().plot(
        figure,
        context,
        data_map,
        pd.DataFrame(),
        {},
        {},
    )
    mock_plot_histogram.assert_called_once_with(
        ANY,
        ANY,
        ANY,
        use_log_scale=ANY,
        minimum=min_value,
        maximum=max_value,
    )


def _first_subplot_bars(figure: Figure) -> list[tuple[float, float, float]]:
    axes = figure.axes[0]
    return [
        (patch.get_x(), patch.get_width(), patch.get_height()) for patch in axes.patches
    ]


def _subplot_counts(figure: Figure) -> list[list[float]]:
    return [[patch.get_height() for patch in axes.patches] for axes in figure.axes]


def _plot_context_for(ensembles: list[EnsembleObject]) -> PlotContext:
    context = Mock(spec=PlotContext)
    context.ensembles.return_value = ensembles
    context.ensembles_color_indexes.return_value = list(range(1, len(ensembles) + 1))
    context.log_scale = False
    context.plotConfig.return_value = PlotConfig(title="Histogram")
    return context


def test_that_ensemble_binning_is_independent_of_other_selected_ensembles():
    # Regression test for #13958: the binning of an ensemble's histogram must
    # depend only on that ensemble's own data, not on other selected ensembles.
    small_ensemble = EnsembleObject(
        "small",
        "small-id",
        False,
        "experiment_1",
        started_at="2012-12-10T00:00:00",
    )
    large_ensemble = EnsembleObject(
        "large",
        "large-id",
        False,
        "experiment_1",
        started_at="2012-12-11T00:00:00",
    )

    rng = np.random.default_rng(1234)
    small_data = pd.DataFrame(rng.normal(size=5))
    large_data = pd.DataFrame(rng.normal(size=5000))

    figure_alone = Figure()
    HistogramPlot().plot(
        figure_alone,
        _plot_context_for([small_ensemble]),
        {small_ensemble: small_data},
        pd.DataFrame(),
        {},
        {},
    )

    figure_with_large = Figure()
    HistogramPlot().plot(
        figure_with_large,
        _plot_context_for([small_ensemble, large_ensemble]),
        {small_ensemble: small_data, large_ensemble: large_data},
        pd.DataFrame(),
        {},
        {},
    )

    bars_alone_height = [patch.get_height() for patch in figure_alone.axes[0].patches]
    bars_with_large_height = [
        patch.get_height() for patch in figure_with_large.axes[0].patches
    ]

    assert bars_alone_height == pytest.approx(bars_with_large_height), (
        "small ensemble's bin heights changed when a larger ensemble was selected"
    )

    bars_alone = _first_subplot_bars(figure_alone)
    bars_with_large = _first_subplot_bars(figure_with_large)

    assert len(bars_alone) == len(bars_with_large), (
        "small ensemble's bin count changed when a larger ensemble was selected"
    )
    assert bars_alone == pytest.approx(bars_with_large)
    assert _first_subplot_bars(figure_alone) == pytest.approx(
        _first_subplot_bars(figure_with_large)
    )
