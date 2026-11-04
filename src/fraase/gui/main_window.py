from __future__ import annotations

from enum import IntEnum

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QWidget

from fraase import __version__
from fraase.gui.pages.docs_page import DocsPage
from fraase.gui.pages.home_page import HomePage
from fraase.gui.pages.new_project_page import NewProjectPage
from fraase.gui.state import AppState


class PageID(IntEnum):
    """Enumeration of page indices/IDs."""

    HOME = 0
    NEW_PROJECT = 1
    DOCS = 2


class MainWindow(QMainWindow):
    """Primary desktop window for FRAASE."""

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.showMaximized()  # Fit to fullscreen
        self.setWindowTitle("FRAASE")

        self.state = state

        self.build_menu_bar()
        self.page_stack = QStackedWidget()  # The page stack will hold all of the different pages
        self.setCentralWidget(self.page_stack)
        self.build_status_bar()

    def build_pages(self) -> None:
        """Build all of the necessary pages."""
        self.home_page = HomePage()
        self.new_project_page = NewProjectPage(state=self.state)
        self.docs_page = DocsPage()

        # Dictionary holding the page index and page widget.
        # Useful for self.navigate_to(page_id)
        self.page_widgets: dict[PageID, QWidget] = {
            PageID.HOME: self.home_page,
            PageID.NEW_PROJECT: self.new_project_page,
            PageID.DOCS: self.docs_page,
        }

        # Add all of the pages to the home stack
        for widget in self.page_widgets.values():
            self.page_stack.addWidget(widget)

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
