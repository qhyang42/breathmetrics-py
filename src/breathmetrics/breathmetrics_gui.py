"""
BreathMetrics GUI backbone (PyQt6 + matplotlib) - Pylance-friendly typing.

pip install pyqt6 matplotlib numpy

Run:
python breathmetrics_gui.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
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


FloatArray = NDArray[np.float64]


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
        self.ax.plot(t, y, linewidth=2)

        def y_at(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            idx = int(np.argmin(np.abs(t - x)))
            return float(y[idx])

        def mark(
            x: Optional[float], yv: Optional[float], label: str, color: str
        ) -> None:
            if x is None or yv is None:
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

        mark(state.inhale_t, y_at(state.inhale_t), "", "#2f77ff")
        mark(state.inhale_pause_t, y_at(state.inhale_pause_t), "", "#12b8b0")
        mark(state.exhale_t, y_at(state.exhale_t), "", "#c9bf2a")
        mark(state.exhale_pause_t, y_at(state.exhale_pause_t), "Next Inhale", "black")

        self.ax.set_title(
            f"Breath {state.breath_index + 1}/{state.n_breaths} ({state.status})"
        )
        self.draw_idle()


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
        if np.isnan(x):
            self.value_label.setText("NaN")
            return
        self.value_label.setText(f"{x:.3f}")


# ----------------------------
# Main window
# ----------------------------


class BreathMetricsMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BreathMetrics GUI")
        self.resize(1500, 800)

        self.breath_rows: list[BreathRow] = self._make_fake_rows(n=130)
        self.current_idx: int = 0
        self.fs: float = 100.0

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
        self.save_all_btn = QPushButton("SAVE ALL CHANGES")

        self.add_note_btn.clicked.connect(lambda: self._stub("Add note"))
        self.reject_btn.clicked.connect(self._toggle_reject_current)
        self.undo_btn.clicked.connect(lambda: self._stub("Undo changes"))
        self.save_all_btn.clicked.connect(lambda: self._stub("Save all changes"))

        gl.addWidget(self.add_note_btn)
        gl.addWidget(self.reject_btn)
        gl.addWidget(self.undo_btn)
        gl.addSpacing(6)
        gl.addWidget(self.save_all_btn)

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
        gl.addWidget(self.canvas, stretch=1)

        cards = QGridLayout()
        cards.setHorizontalSpacing(24)
        cards.setVerticalSpacing(10)

        self.card_inhale = FeatureCard("Inhale", "#2f77ff", show_controls=False)
        self.card_inhale_pause = FeatureCard(
            "Inhale Pause", "#12b8b0", show_controls=True
        )
        self.card_exhale = FeatureCard("Exhale", "#c9bf2a", show_controls=False)
        self.card_exhale_pause = FeatureCard(
            "Exhale Pause", "#f1f000", show_controls=True
        )

        if self.card_inhale_pause.create_btn is not None:
            self.card_inhale_pause.create_btn.clicked.connect(
                lambda: self._stub("Create inhale pause")
            )
        if self.card_inhale_pause.remove_btn is not None:
            self.card_inhale_pause.remove_btn.clicked.connect(
                lambda: self._stub("Remove inhale pause")
            )

        if self.card_exhale_pause.create_btn is not None:
            self.card_exhale_pause.create_btn.clicked.connect(
                lambda: self._stub("Create exhale pause")
            )
        if self.card_exhale_pause.remove_btn is not None:
            self.card_exhale_pause.remove_btn.clicked.connect(
                lambda: self._stub("Remove exhale pause")
            )

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

        state = self._fake_breath_state(idx, len(self.breath_rows))
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
    # Actions (placeholder)
    # ----------------------------

    def _toggle_reject_current(self) -> None:
        # BreathRow is frozen, so we replace the row (cleaner typing + immutable model)
        row = self.breath_rows[self.current_idx]
        new_status = "rejected" if row.status != "rejected" else "valid"
        self.breath_rows[self.current_idx] = BreathRow(
            breath_no=row.breath_no,
            onset_s=row.onset_s,
            offset_s=row.offset_s,
            status=new_status,
        )

        item = self.table.item(self.current_idx, 3)
        if item is not None:
            item.setText(new_status)

        self._select_breath(self.current_idx)

    def _stub(self, msg: str) -> None:
        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(f"{msg} (stub)", 2500)

    # ----------------------------
    # Fake data (so the GUI runs immediately)
    # ----------------------------

    def _make_fake_rows(self, n: int) -> list[BreathRow]:
        rows: list[BreathRow] = []
        onset = 1.563
        for i in range(n):
            dur = 4.6 + 0.6 * float(np.sin(i / 8.0))
            offset = onset + dur
            rows.append(
                BreathRow(
                    breath_no=i + 1, onset_s=onset, offset_s=offset, status="valid"
                )
            )
            onset = offset + (0.4 + 0.2 * float(np.cos(i / 6.0)))
        return rows

    def _fake_breath_state(self, idx: int, n: int) -> BreathViewState:
        t: FloatArray = np.linspace(0.0, 9.0, 900, dtype=np.float64)
        y: FloatArray = (
            0.045 * np.exp(-0.5 * ((t - 3.3) / 0.85) ** 2)
            - 0.070 * np.exp(-0.5 * ((t - 5.0) / 0.55) ** 2)
            + 0.004 * np.sin(2 * np.pi * t * 1.3)
        ).astype(np.float64)

        # mimic screenshot example values
        inhale_t: float = 1.563
        inhale_pause_t: float = np.nan
        exhale_t: float = 4.047
        exhale_pause_t: float = 6.192

        shift = 0.05 * float(np.sin(idx / 10.0))
        y = (y + shift).astype(np.float64)

        status = self.breath_rows[idx].status

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


def main() -> None:
    app = QApplication(sys.argv)
    win = BreathMetricsMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
