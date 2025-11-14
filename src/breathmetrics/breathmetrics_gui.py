import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------- Matplotlib canvas wrapper ----------


class BreathPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setParent(parent)

    def plot_breath(self, t, y, title="Breath 1/130 (valid)"):
        self.ax.clear()
        self.ax.plot(t, y, linewidth=2)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.draw()


# ---------- Main window ----------


class BreathMetricsGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BreathMetrics GUI (Python)")
        self.resize(1400, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        # Left column: table
        main_layout.addLayout(self._build_left_table(), stretch=2)

        # Middle column: big buttons
        main_layout.addLayout(self._build_middle_buttons(), stretch=1)

        # Right column: plot + segment cards
        main_layout.addLayout(self._build_right_panel(), stretch=4)

        # Fake data for now
        self._populate_fake_breaths()

        # Plot the first breath initially
        self.table.selectRow(0)
        self.plot_selected_breath()

    # -------- Left: table --------
    def _build_left_table(self):
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("Select A Breath")
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Breath No.", "Onset", "Offset", "Status"]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )

        # --- these three lines are where Pylance was upset ---

        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

        vheader = self.table.verticalHeader()
        if vheader is not None:
            vheader.setVisible(False)

        layout.addWidget(self.table)

        selection_model = self.table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(self.plot_selected_breath)

        return layout

    # -------- Middle: navigation / actions --------
    def _build_middle_buttons(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        def big_button(text):
            btn = QtWidgets.QPushButton(text)
            btn.setMinimumHeight(60)
            btn.setStyleSheet("font-size: 16px;")
            return btn

        self.prev_btn = big_button("Previous Breath")
        self.next_btn = big_button("Next Breath")
        self.note_btn = big_button("Add Note To This Breath")
        self.reject_btn = big_button("Reject This Breath")
        self.undo_btn = big_button("Undo Changes To This Breath")

        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addSpacing(20)
        layout.addWidget(self.note_btn)
        layout.addWidget(self.reject_btn)
        layout.addWidget(self.undo_btn)
        layout.addStretch(1)

        self.save_btn = big_button("SAVE ALL CHANGES")
        layout.addWidget(self.save_btn)

        # Hook up basic prev/next behavior
        self.prev_btn.clicked.connect(self.go_prev_breath)
        self.next_btn.clicked.connect(self.go_next_breath)

        return layout

    # -------- Right: plot + segment cards --------
    def _build_right_panel(self):
        layout = QtWidgets.QVBoxLayout()

        # Plot title
        top_label = QtWidgets.QLabel("Plot and Edit This Breath")
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        top_label.setFont(font)
        top_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(top_label)

        # Matplotlib canvas
        self.canvas = BreathPlotCanvas()
        layout.addWidget(self.canvas, stretch=1)

        # Segment cards row
        cards_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(cards_layout)

        self.segment_cards = {}
        segments = [
            ("Inhale", "#1E88E5"),  # blue
            ("Inhale Pause", "#00897B"),  # teal
            ("Exhale", "#FDD835"),  # yellow
            ("Exhale Pause", "#FFB300"),  # orange
        ]

        for name, color in segments:
            card = self._build_segment_card(name, color)
            cards_layout.addWidget(card)

        return layout

    def _build_segment_card(self, label_text, color_hex):
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.Box)
        frame.setLineWidth(1)
        frame.setStyleSheet("QFrame { border-radius: 4px; }")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        header = QtWidgets.QLabel(label_text)
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(
            f"background-color: {color_hex}; font-weight: bold; padding: 4px;"
        )
        layout.addWidget(header)

        value_label = QtWidgets.QLabel("NaN")
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("font-size: 18px; padding: 8px;")
        layout.addWidget(value_label)

        btn_create = QtWidgets.QPushButton("Create")
        btn_remove = QtWidgets.QPushButton("Remove")
        layout.addWidget(btn_create)
        layout.addWidget(btn_remove)

        # Store for later updates
        self.segment_cards[label_text] = {
            "value": value_label,
            "btn_create": btn_create,
            "btn_remove": btn_remove,
        }

        return frame

    # -------- Fake data + plotting --------
    def _populate_fake_breaths(self):
        """Create some placeholder breaths to exercise the GUI."""
        n_breaths = 10
        self.breath_data = []

        self.table.setRowCount(n_breaths)

        t = np.linspace(0, 9, 500)

        for i in range(n_breaths):
            # fake waveform: a smooth inhale + exhale with some noise
            y = 0.05 * np.sin(2 * np.pi * (t / 9)) * np.exp(-((t - 4.5) ** 2) / 20)
            y += 0.003 * np.random.randn(len(t))

            onset = 1.5 + i * 4.0
            offset = onset + 4.5
            status = "valid"

            self.breath_data.append(
                {"t": t, "y": y, "onset": onset, "offset": offset, "status": status}
            )

            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{onset:.3f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{offset:.3f}"))
            self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(status))

    def current_row(self) -> int:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return -1

        rows = selection_model.selectedRows()
        if not rows:
            return -1

        return rows[0].row()

    def plot_selected_breath(self, *args):
        row = self.current_row()
        if row < 0:
            return
        breath = self.breath_data[row]
        title = f"Breath {row + 1}/{len(self.breath_data)} ({breath['status']})"
        self.canvas.plot_breath(breath["t"], breath["y"], title=title)

        # Example: update the segment duration numbers (fake)
        self.segment_cards["Inhale"]["value"].setText(
            f"{(breath['onset'] % 3) + 1:.3f}"
        )
        self.segment_cards["Exhale"]["value"].setText(
            f"{(breath['offset'] % 3) + 3:.3f}"
        )
        self.segment_cards["Inhale Pause"]["value"].setText("NaN")
        self.segment_cards["Exhale Pause"]["value"].setText(f"{(row + 1) * 0.1:.3f}")

    # -------- Prev / Next helpers --------
    def go_prev_breath(self):
        row = self.current_row()
        if row <= 0:
            return
        self.table.selectRow(row - 1)

    def go_next_breath(self):
        row = self.current_row()
        if row < 0:
            return
        if row + 1 >= self.table.rowCount():
            return
        self.table.selectRow(row + 1)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = BreathMetricsGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
