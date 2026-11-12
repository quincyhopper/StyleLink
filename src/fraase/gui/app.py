from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fraase.gui.main_window import MainWindow
from fraase.gui.state import AppState


def main() -> int:
    """Launch the desktop application.

    Returns:
        exit code.
    """
    app = QApplication()
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    state = AppState()

    window = MainWindow(state=state)
    window.show()

    return app.exec()


if __name__ == "__main__":
    SystemExit(main())
