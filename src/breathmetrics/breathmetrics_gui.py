"""
BreathMetrics GUI backbone (PyQt6 + matplotlib) - Pylance-friendly typing.

pip install pyqt6 matplotlib numpy

Run:
python breathmetrics_gui.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Callable, Any

import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from breathmetrics.breath_editor import BreathEditor, EventType
from breathmetrics.utils import MISSING_EVENT

# from breathmetrics.breath_editor import BreathEditor, EventType, EditResult


FloatArray = NDArray[np.float64]

# load and verify object
_REQUIRED_ATTRS = (
    "srate",
    "baseline_corrected_respiration",
    "inhale_onsets",
    "exhale_onsets",
)

_COLOR_INHALE = "#2f77ff"
_COLOR_INHALE_PAUSE = "#12b8b0"
_COLOR_EXHALE = "#c9bf2a"
_COLOR_EXHALE_PAUSE = "#f1f000"


def _supports_feature(bm_obj: Any, name: str) -> bool:
    supports = getattr(bm_obj, "supports", None)
    if callable(supports):
        try:
            return bool(supports(name))
        except Exception:
            return False
    return getattr(bm_obj, name, None) is not None


def _validate_bm_obj(bm_obj: Any) -> None:
    if bm_obj is None:
        raise ValueError(
            "bm_obj is None.\n\n"
            "Initialize your BreathMetrics object and run estimate_all_features() first, e.g.:\n"
            "  bm_obj = breathmetrics.Breathe(...)\n"
            "  bm_obj.estimate_all_features(...)\n"
            "Then launch the GUI with:\n"
            "  BreathMetricsMainWindow(bm_obj)"
        )

    missing = [
        a
        for a in _REQUIRED_ATTRS
        if not hasattr(bm_obj, a) or getattr(bm_obj, a) is None
    ]
    if missing:
        raise ValueError(
            "bm_obj does not look like an initialized/estimated BreathMetrics object.\n"
            f"Missing attributes: {missing}\n\n"
            "Did you forget to run estimate_all_features()?"
        )


# ----------------------------
# Data model placeholders
# ----------------------------


@dataclass(frozen=True, slots=True)
class BreathRow:
    breath_no: int
    onset_s: float
    offset_s: float
    status: str = "valid"


@dataclass(slots=True)
class BreathViewState:
    """What the GUI needs to render one breath."""

    breath_index: int
    n_breaths: int
    status: str

    inhale_t: Optional[float] = None
    inhale_pause_t: Optional[float] = None
    exhale_t: Optional[float] = None
    exhale_pause_t: Optional[float] = None

    t: Optional[FloatArray] = None
    y: Optional[FloatArray] = None


# ----------------------------
# Plot widget (matplotlib)
# ----------------------------


class BreathPlotCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        fig = Figure(figsize=(6, 4), tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self._default()
        # --- NEW: interactive dragging state ---
        self._dragging: Optional[EventType] = None
        self._marker_xs: dict[EventType, float] = {}
        self._pick_radius_s: float = 0.15  # seconds (tune)
        self._last_t: Optional[FloatArray] = None
        self._last_y: Optional[FloatArray] = None

        # callback set by MainWindow:
        self.on_move_requested: Optional[Callable[[EventType, float], None]] = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def _default(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("Time (S)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.25)
        self.draw_idle()

    def plot_breath(self, state: BreathViewState) -> None:
        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("Time (S)")
        self.ax.set_ylabel("Amplitude")

        if state.t is None or state.y is None:
            self.ax.set_title("No data")
            self.draw_idle()
            return

        t = state.t
        y = state.y
        self._last_t = t
        self._last_y = y
        self.ax.plot(t, y, linewidth=2)

        def y_at(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            idx = int(np.argmin(np.abs(t - x)))
            return float(y[idx])

        def mark(
            x: Optional[float], yv: Optional[float], label: str, color: str
        ) -> None:
            if x is None or yv is None or (not np.isfinite(x)) or x < 0:
                return
            self.ax.scatter(
                [x],
                [yv],
                s=60,
                zorder=3,
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )
            if label:
                self.ax.text(x, yv, f" {label}", va="center", fontsize=10)

        mark(state.inhale_t, y_at(state.inhale_t), "", _COLOR_INHALE)
        mark(state.inhale_pause_t, y_at(state.inhale_pause_t), "", _COLOR_INHALE_PAUSE)
        mark(state.exhale_t, y_at(state.exhale_t), "", _COLOR_EXHALE)
        mark(state.exhale_pause_t, y_at(state.exhale_pause_t), "", _COLOR_EXHALE_PAUSE)
        # --- store marker x positions for hit testing ---
        self._marker_xs = {}
        if (
            state.inhale_t is not None
            and np.isfinite(state.inhale_t)
            and state.inhale_t >= 0
        ):
            self._marker_xs[EventType.INHALE_ONSET] = float(state.inhale_t)
        if (
            state.exhale_t is not None
            and np.isfinite(state.exhale_t)
            and state.exhale_t >= 0
        ):
            self._marker_xs[EventType.EXHALE_ONSET] = float(state.exhale_t)
        if (
            state.inhale_pause_t is not None
            and np.isfinite(state.inhale_pause_t)
            and state.inhale_pause_t >= 0
        ):
            self._marker_xs[EventType.INHALE_PAUSE_ONSET] = float(state.inhale_pause_t)
        if (
            state.exhale_pause_t is not None
            and np.isfinite(state.exhale_pause_t)
            and state.exhale_pause_t >= 0
        ):
            self._marker_xs[EventType.EXHALE_PAUSE_ONSET] = float(state.exhale_pause_t)

        self.ax.set_title(
            f"Breath {state.breath_index +1}/{state.n_breaths} ({state.status})"
        )

        self.draw_idle()

    def update_markers(
        self, state: BreathViewState, *, keep_window: bool = True
    ) -> None:
        if keep_window and self._last_t is not None and self._last_y is not None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            t = self._last_t
            y = self._last_y
        else:
            t = state.t
            y = state.y
            xlim = None
            ylim = None

        if t is None or y is None:
            self.plot_breath(state)
            return

        self.ax.clear()
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("Time (S)")
        self.ax.set_ylabel("Amplitude")
        self.ax.plot(t, y, linewidth=2)

        def y_at(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            idx = int(np.argmin(np.abs(t - x)))
            return float(y[idx])

        def mark(
            x: Optional[float], yv: Optional[float], label: str, color: str
        ) -> None:
            if x is None or yv is None or (not np.isfinite(x)) or x < 0:
                return
            self.ax.scatter(
                [x],
                [yv],
                s=60,
                zorder=3,
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )
            if label:
                self.ax.text(x, yv, f" {label}", va="center", fontsize=10)

        mark(state.inhale_t, y_at(state.inhale_t), "", _COLOR_INHALE)
        mark(state.inhale_pause_t, y_at(state.inhale_pause_t), "", _COLOR_INHALE_PAUSE)
        mark(state.exhale_t, y_at(state.exhale_t), "", _COLOR_EXHALE)
        mark(state.exhale_pause_t, y_at(state.exhale_pause_t), "", _COLOR_EXHALE_PAUSE)

        self._marker_xs = {}
        if (
            state.inhale_t is not None
            and np.isfinite(state.inhale_t)
            and state.inhale_t >= 0
        ):
            self._marker_xs[EventType.INHALE_ONSET] = float(state.inhale_t)
        if (
            state.exhale_t is not None
            and np.isfinite(state.exhale_t)
            and state.exhale_t >= 0
        ):
            self._marker_xs[EventType.EXHALE_ONSET] = float(state.exhale_t)
        if (
            state.inhale_pause_t is not None
            and np.isfinite(state.inhale_pause_t)
            and state.inhale_pause_t >= 0
        ):
            self._marker_xs[EventType.INHALE_PAUSE_ONSET] = float(state.inhale_pause_t)
        if (
            state.exhale_pause_t is not None
            and np.isfinite(state.exhale_pause_t)
            and state.exhale_pause_t >= 0
        ):
            self._marker_xs[EventType.EXHALE_PAUSE_ONSET] = float(state.exhale_pause_t)

        self.ax.set_title(
            f"Breath {state.breath_index + 1}/{state.n_breaths} ({state.status})"
        )

        if keep_window and xlim is not None and ylim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.draw_idle()

    # --- interactive dragging methods ---
    def _hit_test(self, x: float) -> Optional[EventType]:
        best_ev: Optional[EventType] = None
        best_d: float = float("inf")
        for ev, mx in self._marker_xs.items():
            d = abs(mx - x)
            if d <= self._pick_radius_s and d < best_d:
                best_d = d
                best_ev = ev
        return best_ev

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        ev = self._hit_test(float(event.xdata))
        self._dragging = ev

    def _on_motion(self, event) -> None:
        if self._dragging is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.on_move_requested is not None:
            self.on_move_requested(self._dragging, float(event.xdata))

    def _on_release(self, event) -> None:
        _ = event
        self._dragging = None


# ----------------------------
# Reusable "feature card" widget
# ----------------------------


class FeatureCard(QFrame):
    def __init__(
        self,
        title: str,
        header_color: str,
        show_controls: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("FeatureCard")

        outer = QVBoxLayout(self)
        outer.setSpacing(6)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel(title)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setObjectName("CardHeader")
        header.setStyleSheet(
            f"background:{header_color}; color:black; padding:8px; font-weight:600;"
        )

        self.value_label = QLabel("NaN")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setObjectName("CardValue")

        big = QFont()
        big.setPointSize(22)
        big.setWeight(QFont.Weight.DemiBold)
        self.value_label.setFont(big)
        self.value_label.setMinimumHeight(70)

        outer.addWidget(header)
        outer.addWidget(self.value_label)

        self.create_btn: Optional[QPushButton] = None
        self.remove_btn: Optional[QPushButton] = None

        if show_controls:
            controls = QVBoxLayout()
            controls.setSpacing(6)
            controls.setContentsMargins(18, 0, 18, 12)

            self.create_btn = QPushButton("Create")
            self.remove_btn = QPushButton("Remove")
            self.create_btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self.remove_btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

            controls.addWidget(self.create_btn)
            controls.addWidget(self.remove_btn)
            outer.addLayout(controls)

    def set_value(self, x: Optional[float]) -> None:
        if x is None:
            self.value_label.setText("NaN")
            return
        if (not np.isfinite(x)) or x < 0:
            self.value_label.setText("NaN")
            return
        self.value_label.setText(f"{x:.3f}")


# ----------------------------
# Main window
# ----------------------------


class BreathMetricsMainWindow(QMainWindow):
    def __init__(self, bm_obj) -> None:
        super().__init__()
        _validate_bm_obj(bm_obj)  # validate input object

        self.setWindowTitle("BreathMetrics GUI")
        self.resize(1500, 800)

        self.bm = bm_obj
        self._supports = lambda name: _supports_feature(self.bm, name)
        self._supports_inhale_pause = self._supports("inhale_pause_onsets")
        self._supports_exhale_pause = self._supports("exhale_pause_onsets")
        self._supports_exhale_offsets = self._supports("exhale_offsets")
        self._backup = self._make_backup()
        self.editor = BreathEditor(self.bm)
        self._save_on_close: bool = False

        self.current_idx: int = 0
        self.breath_rows: list[BreathRow] = self._make_rows_from_bm()

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = self._build_left_panel()
        right = self._build_right_panel()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 980])

        self._populate_table()
        self._select_breath(0)

        self.setStyleSheet(
            """
            QWidget {
                color: #111111;
                font-size: 13px;
            }

            QGroupBox {
                font-weight: 700;
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                margin-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px 0 6px;
            }

            QPushButton {
                color: #111111;
                background-color: #f7f7f7;
                padding: 10px 14px;
                border-radius: 8px;
                border: 1px solid #c7c7c7;
            }

            QPushButton:hover {
                background-color: #efefef;
                color: #111111;
            }

            QPushButton:pressed {
                background-color: #e7e7e7;
                color: #111111;
            }

            QTableWidget {
            background-color: white;
            color: #111111;
            gridline-color: #d9d9d9;
            }

            QHeaderView::section {
            background-color: #f0f0f0;
            color: #111111;
            padding: 6px;
            border: 1px solid #d0d0d0;
            }

            QTableWidget::item {
            background-color: white;
            color: #111111;
            }

            QTableWidget::item:selected {
            background-color: #cfe3ff;
            color: #111111;
            }

            QFrame#FeatureCard {
                border: 1px solid #cfcfcf;
                border-radius: 10px;
                background: white;
                color: #111111;
            }

            QLabel#CardValue {
                color: #111111;
            }
            """
        )

    def keyPressEvent(self, a0: Optional[QKeyEvent]) -> None:
        if a0 is None:
            super().keyPressEvent(a0)
            return
        if a0.key() == Qt.Key.Key_Left:
            self._prev_breath()
            a0.accept()
            return
        if a0.key() == Qt.Key.Key_Right:
            self._next_breath()
            a0.accept()
            return
        super().keyPressEvent(a0)

    def _handle_move_requested(self, event: EventType, x_seconds: float) -> None:
        """
        Called by canvas while dragging.
        Since we plot in absolute seconds, conversion is direct.
        """
        i = self.current_idx
        fs = float(self.bm.srate)
        new_sample = int(round(x_seconds * fs))

        res = self.editor.move_event(i, event, new_sample)

        # Update table onset/offset/status for consistency across edits
        self._refresh_row(i)

        # Redraw markers without changing the current view window
        state = self._breath_state_from_bm(i, len(self.breath_rows))
        self.canvas.update_markers(state, keep_window=True)
        self.card_inhale.set_value(state.inhale_t)
        self.card_inhale_pause.set_value(state.inhale_pause_t)
        self.card_exhale.set_value(state.exhale_t)
        self.card_exhale_pause.set_value(state.exhale_pause_t)

        # Optional status feedback
        if res.was_clamped:
            self._stub("Clamped to preserve event order")

    def _refresh_row(self, i: int) -> None:
        """Refresh table row i from bm (onset/offset/status)."""
        fs = float(self.bm.srate)
        onset_s = float(self.bm.inhale_onsets[i]) / fs
        offsets = self._effective_exhale_offsets()
        offset_s = float(offsets[i]) / fs if i < offsets.shape[0] else float("nan")
        if onset_s < 0:
            onset_s = float("nan")
        if offset_s < 0:
            offset_s = float("nan")

        is_valid = getattr(self.bm, "is_valid", None)
        status = "valid"
        if is_valid is not None and not bool(np.asarray(is_valid, dtype=bool)[i]):
            status = "rejected"

        self.breath_rows[i] = BreathRow(
            breath_no=i + 1, onset_s=onset_s, offset_s=offset_s, status=status
        )

        item_on = self.table.item(i, 1)
        item_off = self.table.item(i, 2)
        item_status = self.table.item(i, 3)
        if item_on is not None:
            item_on.setText(f"{onset_s:.4f}")
        if item_off is not None:
            item_off.setText(f"{offset_s:.4f}")
        if item_status is not None:
            item_status.setText(status)

    # ----------------------------
    # UI builders
    # ----------------------------

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        group = QGroupBox("Select A Breath")
        gl = QVBoxLayout(group)
        gl.setSpacing(10)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Breath No.", "Onset", "Offset", "Status"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        vh = self.table.verticalHeader()
        if vh is not None:
            vh.setVisible(True)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.table.cellClicked.connect(self._on_table_clicked)

        gl.addWidget(self.table)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Breath")
        self.next_btn = QPushButton("Next Breath")
        self.prev_btn.clicked.connect(self._prev_breath)
        self.next_btn.clicked.connect(self._next_breath)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        gl.addLayout(nav)

        self.add_note_btn = QPushButton("Add Note To This Breath")
        self.reject_btn = QPushButton("Reject This Breath")
        self.undo_btn = QPushButton("Undo Changes To This Breath")
        self.save_btn = QPushButton("Save All Changes")

        self.add_note_btn.clicked.connect(lambda: self._stub("Add note"))
        self.reject_btn.clicked.connect(self._toggle_reject_current)
        self.undo_btn.clicked.connect(self._undo_current_breath)
        self.save_btn.clicked.connect(self._save_and_close)

        gl.addWidget(self.add_note_btn)
        gl.addWidget(self.reject_btn)
        gl.addWidget(self.undo_btn)
        gl.addSpacing(6)
        gl.addWidget(self.save_btn)

        layout.addWidget(group)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        group = QGroupBox("Plot and Edit This Breath")
        gl = QVBoxLayout(group)
        gl.setSpacing(10)

        self.canvas = BreathPlotCanvas()
        # --- connect canvas dragging to editor ---
        self.canvas.on_move_requested = self._handle_move_requested

        gl.addWidget(self.canvas, stretch=1)

        cards = QGridLayout()
        cards.setHorizontalSpacing(24)
        cards.setVerticalSpacing(10)

        self.card_inhale = FeatureCard("Inhale", _COLOR_INHALE, show_controls=False)
        self.card_inhale_pause = FeatureCard(
            "Inhale Pause",
            _COLOR_INHALE_PAUSE,
            show_controls=self._supports_inhale_pause,
        )
        self.card_exhale = FeatureCard("Exhale", _COLOR_EXHALE, show_controls=False)
        self.card_exhale_pause = FeatureCard(
            "Exhale Pause",
            _COLOR_EXHALE_PAUSE,
            show_controls=self._supports_exhale_pause,
        )

        self.card_inhale_pause.setEnabled(self._supports_inhale_pause)
        self.card_exhale_pause.setEnabled(self._supports_exhale_pause)

        if (
            self.card_inhale_pause.create_btn is not None
            and self._supports_inhale_pause
        ):
            self.card_inhale_pause.create_btn.clicked.connect(self._create_inhale_pause)

        if (
            self.card_inhale_pause.remove_btn is not None
            and self._supports_inhale_pause
        ):
            self.card_inhale_pause.remove_btn.clicked.connect(self._remove_inhale_pause)

        if (
            self.card_exhale_pause.create_btn is not None
            and self._supports_exhale_pause
        ):
            self.card_exhale_pause.create_btn.clicked.connect(self._create_exhale_pause)

        if (
            self.card_exhale_pause.remove_btn is not None
            and self._supports_exhale_pause
        ):
            self.card_exhale_pause.remove_btn.clicked.connect(self._remove_exhale_pause)

        cards.addWidget(self.card_inhale, 0, 0)
        cards.addWidget(self.card_inhale_pause, 0, 1)
        cards.addWidget(self.card_exhale, 0, 2)
        cards.addWidget(self.card_exhale_pause, 0, 3)

        gl.addLayout(cards)
        layout.addWidget(group)
        return panel

    # ----------------------------
    # Table + selection
    # ----------------------------

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.breath_rows))
        for r, row in enumerate(self.breath_rows):
            items = [
                QTableWidgetItem(f"{row.breath_no:d}"),
                QTableWidgetItem(f"{row.onset_s:.4f}"),
                QTableWidgetItem(f"{row.offset_s:.4f}"),
                QTableWidgetItem(row.status),
            ]
            for c, it in enumerate(items):
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()

    def _on_table_clicked(self, row: int, col: int) -> None:
        _ = col
        self._select_breath(row)

    def _select_breath(self, idx: int) -> None:
        if not self.breath_rows:
            return

        idx = int(np.clip(idx, 0, len(self.breath_rows) - 1))
        self.current_idx = idx

        self.table.selectRow(idx)

        state = self._breath_state_from_bm(idx, len(self.breath_rows))
        self.canvas.plot_breath(state)

        self.card_inhale.set_value(state.inhale_t)
        self.card_inhale_pause.set_value(state.inhale_pause_t)
        self.card_exhale.set_value(state.exhale_t)
        self.card_exhale_pause.set_value(state.exhale_pause_t)

    def _prev_breath(self) -> None:
        self._select_breath(self.current_idx - 1)

    def _next_breath(self) -> None:
        self._select_breath(self.current_idx + 1)

    # ----------------------------
    # Actions
    # ----------------------------

    def _toggle_reject_current(self) -> None:
        i = self.current_idx
        self.editor.toggle_reject(i)
        self._refresh_row(i)
        self._select_breath(i)

    def _stub(self, msg: str) -> None:
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"{msg} (stub)", 2500)

    def _undo_current_breath(self) -> None:
        i = self.current_idx
        self.editor.undo_breath(i)
        self._refresh_row(i)
        self._select_breath(i)

    def _save_and_close(self) -> None:
        self._save_on_close = True
        setattr(self.bm, "_gui_save_requested", True)
        self.close()

    def _confirm_discard(self) -> bool:
        resp = QMessageBox.question(
            self,
            "Discard Changes?",
            "Close and discard all changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return resp == QMessageBox.StandardButton.Yes

    def _restore_from_backup(self) -> None:
        if self._backup is None:
            return
        for k, v in self._backup.items():
            setattr(self.bm, k, np.copy(v))

    def closeEvent(
        self, a0: QCloseEvent | None
    ) -> None:  # stupid type checker requires this.
        if a0 is None:
            return
        if self._save_on_close:
            a0.accept()
            return
        if self._confirm_discard():
            self._restore_from_backup()
            setattr(self.bm, "_gui_save_requested", False)
            a0.accept()
            return
        a0.ignore()

    def _make_backup(self) -> dict[str, np.ndarray]:
        keys = [
            "inhale_peaks",
            "exhale_troughs",
            "inhale_onsets",
            "exhale_onsets",
            "inhale_offsets",
            "exhale_offsets",
            "inhale_pause_onsets",
            "exhale_pause_onsets",
            "inhale_time2peak",
            "exhale_time2trough",
            "inhale_durations",
            "exhale_durations",
            "inhale_volumes",
            "exhale_volumes",
            "inhale_pause_durations",
            "exhale_pause_durations",
            "is_valid",
        ]
        backup: dict[str, np.ndarray] = {}
        for k in keys:
            if not hasattr(self.bm, k):
                continue
            if k != "is_valid" and not self._supports(k):
                continue
            val = getattr(self.bm, k)
            if val is None:
                continue
            backup[k] = np.copy(val)
        return backup

    def _create_inhale_pause(self) -> None:
        if not self._supports_inhale_pause:
            return
        i = self.current_idx
        fs = float(self.bm.srate)
        target = int(round(float(self.bm.exhale_onsets[i]) - (0.5 * fs)))
        self.editor.move_event(i, EventType.INHALE_PAUSE_ONSET, target)
        self._select_breath(i)

    def _remove_inhale_pause(self) -> None:
        if not self._supports_inhale_pause:
            return
        i = self.current_idx
        # TODO: add BreathEditor.clear_event(...) for purity
        self.bm.inhale_pause_onsets[i] = MISSING_EVENT
        if hasattr(self.bm, "statuses"):
            self.bm.statuses["inhale_pause_onsets"] = "edited"
        # recompute inhale-dependent stuff:
        self.editor.move_event(
            i, EventType.INHALE_ONSET, int(round(float(self.bm.inhale_onsets[i])))
        )
        self._select_breath(i)

    def _create_exhale_pause(self) -> None:
        if not self._supports_exhale_pause or not self._supports_exhale_offsets:
            return
        i = self.current_idx
        fs = float(self.bm.srate)
        target = int(round(float(self.bm.exhale_offsets[i]) - (0.5 * fs)))
        self.editor.move_event(i, EventType.EXHALE_PAUSE_ONSET, target)
        self._select_breath(i)

    def _remove_exhale_pause(self) -> None:
        if not self._supports_exhale_pause:
            return
        i = self.current_idx
        # TODO: add BreathEditor.clear_event(...) for purity
        self.bm.exhale_pause_onsets[i] = MISSING_EVENT
        if hasattr(self.bm, "statuses"):
            self.bm.statuses["exhale_pause_onsets"] = "edited"
        # recompute exhale-dependent stuff:
        self.editor.move_event(
            i, EventType.EXHALE_ONSET, int(round(float(self.bm.exhale_onsets[i])))
        )
        self._select_breath(i)

    # ----------------------------
    # real BreathMetrics data extraction
    # ----------------------------

    def _effective_exhale_offsets(self) -> np.ndarray:
        if (
            self._supports_exhale_offsets
            and getattr(self.bm, "exhale_offsets", None) is not None
        ):
            return np.asarray(self.bm.exhale_offsets, dtype=float)
        inhale_onsets = np.asarray(self.bm.inhale_onsets, dtype=float)
        offsets = np.full_like(inhale_onsets, MISSING_EVENT, dtype=float)
        if inhale_onsets.size > 1:
            offsets[:-1] = inhale_onsets[1:]
        return offsets

    def _make_rows_from_bm(self) -> list[BreathRow]:
        fs = float(self.bm.srate)
        inhale_onsets = np.asarray(self.bm.inhale_onsets, dtype=float)
        exhale_offsets = self._effective_exhale_offsets()
        n = inhale_onsets.shape[0]
        if exhale_offsets.shape[0] < n:
            pad = np.full(n - exhale_offsets.shape[0], MISSING_EVENT, dtype=float)
            exhale_offsets = np.concatenate([exhale_offsets, pad])

        # is_valid stored/attached by BreathEditor; default valid if missing
        is_valid = getattr(self.bm, "is_valid", None)
        if is_valid is None:
            is_valid = np.ones_like(inhale_onsets, dtype=bool)
        else:
            is_valid = np.asarray(is_valid, dtype=bool)

        rows: list[BreathRow] = []
        for i in range(n):
            onset_s = float(inhale_onsets[i]) / fs
            offset_s = float(exhale_offsets[i]) / fs
            if onset_s < 0:
                onset_s = float("nan")
            if offset_s < 0:
                offset_s = float("nan")
            status = "valid" if bool(is_valid[i]) else "rejected"
            rows.append(
                BreathRow(
                    breath_no=i + 1,
                    onset_s=onset_s,
                    offset_s=offset_s,
                    status=status,
                )
            )
        return rows

    def _breath_state_from_bm(self, idx: int, n: int) -> BreathViewState:
        fs = float(self.bm.srate)

        # breath window definition per your spec:
        inhale_onsets = self.bm.inhale_onsets
        exhale_offsets = self._effective_exhale_offsets()
        if idx >= len(inhale_onsets):
            return BreathViewState(
                breath_index=idx,
                n_breaths=n,
                status="valid",
            )
        start = int(round(float(inhale_onsets[idx])))
        if idx < len(exhale_offsets):
            end = int(round(float(exhale_offsets[idx])))
        else:
            end = start
        if start < 0:
            start = 0
        if end < start:
            end = start

        y_all = np.asarray(self.bm.baseline_corrected_respiration, dtype=np.float64)

        # add padding for context
        pad = int(round(1.0 * fs))
        lo = max(0, start - pad)
        hi = min(y_all.shape[0] - 1, end + pad)

        # PLOT X-AXIS CONVENTION:
        # Here, we plot in *absolute seconds* to make editing conversions easy.
        t = (np.arange(lo, hi + 1, dtype=np.float64) / fs).astype(np.float64)
        y = y_all[lo : hi + 1].astype(np.float64)

        # markers also in absolute seconds
        def s_to_sec(samp_arr, i) -> Optional[float]:
            if i >= len(samp_arr):
                return None
            v = float(samp_arr[i])
            if v < 0 or not np.isfinite(v):
                return None
            return v / fs

        inhale_t = s_to_sec(self.bm.inhale_onsets, idx)
        exhale_t = s_to_sec(self.bm.exhale_onsets, idx)
        inhale_pause_t = (
            s_to_sec(self.bm.inhale_pause_onsets, idx)
            if self._supports_inhale_pause
            and getattr(self.bm, "inhale_pause_onsets", None) is not None
            else None
        )
        exhale_pause_t = (
            s_to_sec(self.bm.exhale_pause_onsets, idx)
            if self._supports_exhale_pause
            and getattr(self.bm, "exhale_pause_onsets", None) is not None
            else None
        )

        is_valid = getattr(self.bm, "is_valid", None)
        status = "valid"
        if is_valid is not None and not bool(np.asarray(is_valid, dtype=bool)[idx]):
            status = "rejected"

        return BreathViewState(
            breath_index=idx,
            n_breaths=n,
            status=status,
            inhale_t=inhale_t,
            inhale_pause_t=inhale_pause_t,
            exhale_t=exhale_t,
            exhale_pause_t=exhale_pause_t,
            t=t,
            y=y,
        )


def main(bm_obj: Optional[Any] = None) -> int:
    _validate_bm_obj(bm_obj)

    app = QApplication(sys.argv)
    win = BreathMetricsMainWindow(bm_obj)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(
        "The GUI expects an existing breathmetrics object (notebook-style usage).\n\n"
        "Example (in a notebook):\n"
        "  from breathmetrics_gui import main\n"
        "  main(bm_obj)\n\n"
        "Or:\n"
        "  from breathmetrics_gui import BreathMetricsMainWindow\n"
        "  app = QApplication([])\n"
        "  win = BreathMetricsMainWindow(bm_obj)\n"
        "  win.show()\n"
        "  app.exec()\n"
    )
