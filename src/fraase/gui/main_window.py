from __future__ import annotations

import shutil
from enum import IntEnum
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QStackedWidget, QWidget

from fraase import __version__
from fraase.gui.pages.explore_page import ExplorePage
from fraase.gui.pages.home_page import HomePage
from fraase.gui.pages.new_project_dialog import NewProjectDialog
from fraase.gui.pages.search_page import SearchPage
from fraase.gui.state import PROJECT_DIR, AppState

MENUBAR_STYLESHEET_PATH = Path(__file__).parent / "static" / "styles" / "menu_bar.qss"


class PageID(IntEnum):
    """Enumeration of page indices/IDs."""

    HOME = 0
    EXPLORE = 1
    SEARCH = 2


class MainWindow(QMainWindow):
    """Primary desktop window for FRAASE."""

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.showMaximized()  # Fit to fullscreen
        self.setWindowTitle("FRAASE")

        self.state = state

        self.build_pages()
        self.build_menu_bar()
        self.build_status_bar()

    def build_pages(self) -> None:
        """Build all of the necessary pages."""
        # The page stack will hold all of the different pages
        self.page_stack = QStackedWidget()
        self.setCentralWidget(self.page_stack)

        self.home_page = HomePage(state=self.state)
        self.explore_page = ExplorePage(state=self.state, parent=self)
        self.search_page = SearchPage(state=self.state, parent=self)

        # Connect quick search button
        self.home_page.quick_search_requested.connect(lambda: self.navigate_to(PageID.SEARCH))
        self.home_page.quick_search_requested.connect(self.search_page._on_search_btn_clicked)

        # Dictionary holding the page index and page widget.
        # Useful for self.navigate_to(page_id)
        self.page_widgets: dict[PageID, QWidget] = {
            PageID.HOME: self.home_page,
            PageID.EXPLORE: self.explore_page,
            PageID.SEARCH: self.search_page,
        }

        # Add all of the pages to the home stack
        for widget in self.page_widgets.values():
            self.page_stack.addWidget(widget)

    def build_menu_bar(self) -> None:
        """Build the top menu bar of the application."""
        # Qt uses the macOS native menu bar by defualt. To render the menu bar in the application
        # itself, we have to disable this
        self.menuBar().setNativeMenuBar(False)

        with MENUBAR_STYLESHEET_PATH.open() as file:
            self.menuBar().setStyleSheet(file.read())

        # Home menu
        self.home_menu = self.menuBar().addAction("&Home")
        self.home_menu.triggered.connect(lambda: self.navigate_to(PageID.HOME))

        # File menu
        self.file_menu = self.menuBar().addMenu("&File")
        new_project_action = self.file_menu.addAction("New Project")
        new_project_action.triggered.connect(self.on_new_project)

        self.file_menu.addSeparator()

        open_project_action = self.file_menu.addAction("Open")
        open_project_action.triggered.connect(self.on_open_project)
        self.add_dataset_action = self.file_menu.addAction("Add new file")
        self.add_dataset_action.triggered.connect(lambda: NewProjectDialog(state=self.state, add_to_active=True).exec())

        self.file_menu.addSeparator()

        # Delete project
        self.delete_project_action = self.file_menu.addAction("Delete project")
        self.delete_project_action.triggered.connect(self.on_delete_project)

        self.file_menu.addSeparator()

        # The Explore and Search menus appear only when a project is active
        self.explore_menu = self.menuBar().addAction("&Explore documents")
        self.explore_menu.triggered.connect(lambda: self.navigate_to(PageID.EXPLORE))

        self.search_menu = self.menuBar().addAction("&Search")
        self.search_menu.triggered.connect(lambda: self.navigate_to(PageID.SEARCH))

        # Read the signal from the active project being changed, and diplay Explore and Search if project  exists
        self.state.project_changed.connect(self.on_project_changed)
        self.on_project_changed(self.state.active_project)

    def build_status_bar(self) -> None:
        """Build the status bar responsible for indicating the current project and any other details."""
        version_label = QLabel(text=f"FRAASE v{__version__}")
        self.statusBar().addPermanentWidget(version_label)

    def navigate_to(self, page_id: PageID) -> None:
        """Navigate to a page."""
        dst: QWidget | None = self.page_widgets.get(page_id)
        if dst is not None:
            # Handle the navigation here
            self.page_stack.setCurrentWidget(dst)

    @Slot(object)
    def on_project_changed(self, project: object) -> None:
        """Make Explore and Search visible if there is an active project."""
        has_project = project is not None
        self.explore_menu.setVisible(has_project)
        self.search_menu.setVisible(has_project)
        self.setWindowTitle(f"FRAASE - {project.project_dir.name}" if project else "FRAASE")
        self.delete_project_action.setVisible(has_project)
        self.add_dataset_action.setVisible(has_project)

    @Slot()
    def on_new_project(self) -> None:
        """Close current project, navigate to the new case project page."""
        dialog = NewProjectDialog(state=self.state)
        dialog.exec()

    @Slot()
    def on_open_project(self) -> None:
        """Open an existing project."""
        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        project_dir = QFileDialog.getExistingDirectory(self, "Open a project folder", dir=str(PROJECT_DIR))
        if project_dir:
            self.state.open_project(Path(project_dir))

    @Slot()
    def on_delete_project(self) -> None:
        """Delete the active project."""
        project = self.state.active_project
        if project is None:
            return None

        reply = QMessageBox.question(
            self,
            "Delete project",
            f"Are you sure you want to permanently delete '{project.project_dir.name}'?",
        )

        if reply == QMessageBox.StandardButton.Yes:
            project_dir = project.project_dir
            self.state.close_project()
            shutil.rmtree(project_dir)
            self.navigate_to(PageID.HOME)
