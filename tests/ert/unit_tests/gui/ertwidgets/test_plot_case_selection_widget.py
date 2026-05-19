from PyQt6.QtCore import Qt
from pytestqt.qtbot import QtBot

from ert.gui.tools.plot.plot_api import EnsembleObject
from ert.gui.tools.plot.plot_ensemble_selection_widget import (
    EnsembleSelectionWidget,
    EnsembleSelectListWidget,
)
from tests.ert.ui_tests.gui.conftest import get_child


def test_ensemble_selection_widget_max_min_selection(qtbot: QtBot):
    test_ensemble_names = [
        EnsembleObject(
            name=f"case{i}",
            id="id",
            hidden=False,
            experiment_name="exp",
            started_at="2012-12-10T00:00:00",
        )
        for i in range(10)
    ]
    widget = EnsembleSelectionWidget(test_ensemble_names, 1)
    qtbot.addWidget(widget)
    list_widget = get_child(widget, EnsembleSelectListWidget, "ensemble_selector")

    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_minimum_ensemble_limit()
    )  # initially one selected

    qtbot.mouseClick(
        list_widget.viewport(),
        Qt.MouseButton.LeftButton,
        pos=list_widget.visualItemRect(list_widget.item(0)).center(),
    )  # deselect the only item selected

    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_minimum_ensemble_limit()
    )  # still one selected

    def iterate_and_click_all_items(count):
        for index in count:
            it = list_widget.item(index)
            qtbot.mouseClick(
                list_widget.viewport(),
                Qt.MouseButton.LeftButton,
                pos=list_widget.visualItemRect(it).center(),
            )

    iterate_and_click_all_items(range(list_widget.count()))  # select 'all'
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_maximum_ensemble_limit()
    )

    iterate_and_click_all_items(reversed(range(list_widget.count())))  # deselect 'all'
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_minimum_ensemble_limit()
    )

    # Increase upper limit and verify that we can select more ensembles
    list_widget.set_maximum_ensemble_limit(10)
    iterate_and_click_all_items(range(list_widget.count()))  # select 'all'
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_maximum_ensemble_limit()
    )

    # Increase lower limit and verify that we cannot deselect below the new limit
    list_widget.set_minimum_ensemble_limit(5)
    iterate_and_click_all_items(reversed(range(list_widget.count())))  # deselect 'all'
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_minimum_ensemble_limit()
    )

    # Set lower lim to 0, then clear selection
    list_widget.set_minimum_ensemble_limit(0)
    list_widget.clear_ensemble_selection()
    assert len(widget.get_selected_ensembles()) == 0

    list_widget.select_all_ensembles()
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_maximum_ensemble_limit()
    )

    # Check that resetting to default works as expected
    list_widget.reset_maximum_ensemble_limit_to_default()
    list_widget.reset_minimum_ensemble_limit_to_default()
    iterate_and_click_all_items(reversed(range(list_widget.count())))  # deselect 'all'
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_minimum_ensemble_limit()
    )
    iterate_and_click_all_items(range(list_widget.count()))  # select 'all'
    assert (
        len(widget.get_selected_ensembles()) == list_widget.get_maximum_ensemble_limit()
    )
