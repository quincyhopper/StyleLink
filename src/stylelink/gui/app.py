from __future__ import annotations

import multiprocessing
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from stylelink.gui.main_window import MainWindow
from stylelink.gui.state import AppState


def main() -> int:
    """Launch the desktop application.

    Returns:
        exit code.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    app = QApplication()
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    state = AppState()

    window = MainWindow(state=state)
    window.show()

    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
