# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-23

### Added
- Calibrator selector dropdown on the Search page ("General", "Forum posts", and "Chat logs")
- Verbal LLR interpretation column in exported CSV results.
- PyInstaller packaging support for standalone desktop application bundles.
- "Delete documents" button to Explore page

### Changed
- Load the STAR model and tokenizer directly from local `star_model/` directory (with automatic first-run download for git clones).
- Replaced pickled `scikit-learn` calibrator models with a JSON file containing the coefficients and intercepts (`calibrators.json`), removing the `scikit-learn` dependency.
- Simplified the Search and Home pages by hiding advanced query/target filtering tabs, the "Exclude self" toggle and the "Quick Search" box. This may be added later on if it is needed.
- Updated verbal LLR thresholds.
- Name of project from FRAASE/LAPSUM to StyleLink

### Removed
- `scikit-learn` runtime dependency.

## [0.2.1] - 2026-11-17

### Added
- Cancel button for embedding generation.
- "Getting started" guide on Home page.
- Installation instructions in `README.md`.

### Changed
- Significantly improved search speed by batching Sqlite lookups and lazy-loading centroid sample texts.

### Fixed
- Display true row count in filter tables when more than 1,000 rows match.

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