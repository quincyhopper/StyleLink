# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-16

### Added
- Support for .xlsx files.
- "author" and "text" column selector for files that lack these columns during import.
- "Delete project" menu action.
- Export to CSV button.
- Functionality for adding files to an existing project.
- Verbal description of LLR values.

### Changed
- Hide row selection options in the Explore page.
- Display the active project name in the application window's title.
- Add titles to the 3 panes in the results tab.
- Store projects in a dedicated default directory instead of the current working directory.

### Removed
- Docs page.
- "Save" and "Save As" menu actions.

## [0.1.0] - 2026-11-12

### Added
- First release of LAPSUM (although project is currently called "fraase").
- Basic document import functionality.
- Embedding generation and storage.
- Vector search.