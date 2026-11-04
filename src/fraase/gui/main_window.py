from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QWidget

from fraase import __version__
from fraase.gui.pages.home_page import HomePage
from fraase.gui.pages.new_project_page import NewProjectPage
from fraase.gui.state import AppState


class MainWindow(QMainWindow):
    """Primary desktop window for FRAASE."""

    PAGE_HOME = 0
    PAGE_NEW_PROJECT = 1

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.showMaximized()  # Fit to fullscreen
        self.setWindowTitle("FRAASE")

        # The state can be passed of to other pages/services that need to be aware of what model we're using,
        # what project is active, etc
        self.state = state
        self.build_ui()

    def build_ui(self) -> None:
        """Build the main UI components (central widgets, menus, etc)."""
        self.build_menu_bar()

        # Make a central container that will occupy the central widget of the main window
        # All pages and central UI stuff will inherent from this widget
        self.central_container = QWidget()
        self.setCentralWidget(self.central_container)

        # The page stack will hold all of the different pages
        self.page_stack = QStackedWidget(self.central_container)

        self.home_page = HomePage()
        self.new_project_page = NewProjectPage(state=self.state)

        # Dictionary holding the page index and page widget.
        # Useful for self.navigate_to(page_id)
        self.page_widgets: dict[int, QWidget] = {
            self.PAGE_HOME: self.home_page,
            self.PAGE_NEW_PROJECT: self.new_project_page,
        }

        # Add all of the pages to the home stack
        for widget in self.page_widgets.values():
            self.page_stack.addWidget(widget)

        self.build_status_bar()

    def build_menu_bar(self) -> None:
        """Build the top menu bar of the application."""
        self.menuBar().setNativeMenuBar(False)  # This is needed to display the menu bar; not sure why

        # Home menu
        self.home_menu = self.menuBar().addMenu("&Home")

        # File menu
        self.file_menu = self.menuBar().addMenu("&File")
        new_project_action = self.file_menu.addAction("New Project")
        self.file_menu.addSeparator()
        open_project_action = self.file_menu.addAction("Open")
        self.file_menu.addSeparator()
        save_project_action = self.file_menu.addAction("Save")
        saveas_project_action = self.file_menu.addAction("Save as")

        # Add connections
        new_project_action.triggered.connect(self.on_new_project)
        # open_project_action.triggered.connect()
        # save_project_action.triggered.connect()
        # saveas_project_action.triggered.connect()

    def build_status_bar(self) -> None:
        """Build the status bar responsible for indicating the current project and any other details."""
        version_label = QLabel(text=f"FRAASE v{__version__}")
        self.statusBar().addPermanentWidget(version_label)

    def navigate_to(self, page_id: int) -> None:
        """Navigate to a page."""
        dst: QWidget | None = self.page_widgets.get(page_id)
        if dst is not None:
            # Handle the navigation here
            self.page_stack.setCurrentWidget(dst)

    @Slot()
    def on_new_project(self) -> None:
        """Close current project, navigate to the new case project page."""
        self.state.close_project()
        self.navigate_to(self.PAGE_NEW_PROJECT)
