"""Drive the Qt window like a user (clicks, slider, hover, pause, save) and screenshot each step: python tools/ui_smoke_qt.py OUT_DIR"""
import sys, time, pathlib
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtTest import QTest

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
app = QtWidgets.QApplication([])
log = []

def shot(name, win):
    app.processEvents()
    screen = app.primaryScreen()
    pix = screen.grabWindow(0)  # whole screen, so tooltips (separate windows) are included
    if pix.isNull() or pix.width() == 0:
        pix = win.grab()
        log.append(f"{name}: screen grab unavailable, window grab used")
    pix.save(str(OUT / f"{name}.png"))

def button(win, text):
    for b in win.findChildren(QtWidgets.QPushButton):
        if b.text() == text:
            return b
    raise RuntimeError(text)

def check(win, text):
    for c in win.findChildren(QtWidgets.QCheckBox):
        if c.text() == text:
            return c
    raise RuntimeError(text)

def steps():
    win = next(w for w in app.topLevelWidgets() if isinstance(w, QtWidgets.QMainWindow))
    win.raise_(); win.activateWindow()
    shot("00_starting", win)
    viewer = [w for w in win.findChildren(QtWidgets.QLabel) if w.objectName() == "value"]
    t0 = time.time()
    while viewer[0].text() in ("–", "") and time.time() - t0 < 20:
        QTest.qWait(200)
    log.append(f"first frame visible after {time.time()-t0:.1f}s of waiting (window opened earlier)")
    QTest.qWait(1500)
    shot("01_start", win)
    QTest.mouseClick(button(win, "Beam"), QtCore.Qt.LeftButton); QTest.qWait(600); shot("02_beam_clicked", win)
    QTest.mouseClick(button(win, "Top"), QtCore.Qt.LeftButton); QTest.qWait(600); shot("03_top_clicked", win)
    c = check(win, "Moving only"); QTest.mouseClick(c, QtCore.Qt.LeftButton); QTest.qWait(800); shot("04_moving_only", win)
    log.append(f"moving only checked: {c.isChecked()}")
    QTest.mouseClick(c, QtCore.Qt.LeftButton); QTest.qWait(300)
    c = check(win, "Ground points"); QTest.mouseClick(c, QtCore.Qt.LeftButton); QTest.qWait(600); shot("05_ground_off", win)
    # hover a checkbox and wait for the tooltip
    tgt = check(win, "Trails")
    QTest.mouseMove(tgt, QtCore.QPoint(10, 8)); QTest.qWait(1500); shot("06_hover_trails", win)
    log.append(f"status bar after hover: {win.statusBar().currentMessage()!r}")
    log.append(f"QToolTip visible after hover: {QtWidgets.QToolTip.isVisible()} text={QtWidgets.QToolTip.text()!r}")
    tips = [w for w in app.allWidgets() if w.isVisible() and (w.windowFlags() & QtCore.Qt.ToolTip) == QtCore.Qt.ToolTip and w is not win]
    for i, w in enumerate(tips):
        w.grab().save(str(OUT / f"06_tooltip_{i}.png")); log.append(f"tooltip widget {type(w).__name__} grabbed")
    app.sendEvent(button(win, "Pause"), QtCore.QEvent(QtCore.QEvent.Enter)); QTest.qWait(200)
    log.append(f"status bar on Pause enter: {win.statusBar().currentMessage()!r}")
    app.sendEvent(button(win, "Pause"), QtCore.QEvent(QtCore.QEvent.Leave)); QTest.qWait(200)
    log.append(f"status bar after leave: {win.statusBar().currentMessage()!r}")
    sl = [s for s in win.findChildren(QtWidgets.QSlider) if s.maximum() == 15][0]
    sl.setValue(12); QTest.qWait(600); shot("07_voxel_12", win)
    QTest.mouseClick(button(win, "Iso"), QtCore.Qt.LeftButton); QTest.mouseClick(button(win, "Height"), QtCore.Qt.LeftButton); QTest.qWait(400)
    QTest.mouseClick(button(win, "Pause"), QtCore.Qt.LeftButton); QTest.qWait(800); shot("08_paused", win)
    log.append(f"pause button text: {button(win, 'Resume').text() if any(b.text()=='Resume' for b in win.findChildren(QtWidgets.QPushButton)) else 'not Resume'}")
    QTest.mouseClick(button(win, "Save PLY"), QtCore.Qt.LeftButton); QTest.qWait(800); log.append(f"status after save: {win.statusBar().currentMessage()!r}")
    QTest.mouseClick(button(win, "Screenshot"), QtCore.Qt.LeftButton); QTest.qWait(600); log.append(f"status after shot: {win.statusBar().currentMessage()!r}")
    QTest.keyClick(win, QtCore.Qt.Key_Space); QTest.qWait(300)
    QTest.keyClick(win, QtCore.Qt.Key_3); QTest.qWait(500); shot("09_key_3", win)
    win.resize(1100, 700); QTest.qWait(700); shot("10_resized", win)
    (OUT / "log.txt").write_text("\n".join(log) + "\n")
    win.close()

QtCore.QTimer.singleShot(3500, steps)
from helios16p.cli import demo
demo(["--loop", "--max-frames", "100000"])
