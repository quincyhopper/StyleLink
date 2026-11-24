#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

uv run pyinstaller \
    --name StyleLink \
    --windowed \
    --onedir \
    --noconfirm \
    --add-data "src/stylelink/gui/static/styles/menu_bar.qss:stylelink/gui/static/styles" \
    --add-data "src/stylelink/embedding/calibrators.json:stylelink/embedding" \
    --add-data "src/stylelink/embedding/star_model:stylelink/embedding/star_model" \
    src/stylelink/gui/app.py

hdiutil create -volname "StyleLink" -srcfolder dist/StyleLink.app -ov -format UDZO StyleLink-macOS.dmg