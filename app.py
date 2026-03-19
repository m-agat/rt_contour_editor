from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    # Avoid forcing an initial top-level size on Wayland. Some compositors
    # maximize the window immediately and Qt can then submit a buffer that
    # no longer matches the configured surface size.
    window.setMinimumSize(900, 900)
    window.show()
    return app.exec()

 
if __name__ == "__main__":
    raise SystemExit(main())
