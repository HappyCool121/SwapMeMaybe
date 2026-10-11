from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from liveswap.client.engine import ClientOptions, run_managed
from liveswap.contracts import Profile


class Runner(QThread):
    failure = Signal(str)

    def __init__(self, options, deployment):
        super().__init__()
        self.options, self.deployment = options, deployment
        self.stop_event = threading.Event()
        self.mailbox_lock = threading.Lock()
        self.latest_status = self.latest_preview = None

    def cache_status(self, value):
        with self.mailbox_lock:
            self.latest_status = value

    def cache_preview(self, value):
        with self.mailbox_lock:
            self.latest_preview = value

    def poll(self):
        with self.mailbox_lock:
            status, preview = self.latest_status, self.latest_preview
            self.latest_status = self.latest_preview = None
            return status, preview

    def run(self):
        try:
            asyncio.run(run_managed(self.options, deployment=self.deployment, stop=self.stop_event,
                                    status=self.cache_status, preview=self.cache_preview))
        except Exception as exc:
            self.failure.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SwapMeMaybe — delayed clothing replacement")
        self.resize(1060, 900)
        self.runner = None
        self.closing = False
        container = QWidget()
        layout = QVBoxLayout(container)
        title = QLabel("SwapMeMaybe")
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        layout.addWidget(title)
        layout.addWidget(QLabel("One garment · one person · fixed-delay camera and microphone for OBS"))
        form = QFormLayout()
        self.garment = self.file_field(form, "Garment image", "")
        self.profile = self.file_field(form, "Worker profile", "profiles/vace.json")
        self.key = self.file_field(form, "Private signing key", ".liveswap/client.pem")
        self.deployment = self.file_field(form, "Managed Nosana pool (optional)", "")
        self.benchmark = self.file_field(form, "Benchmark report (optional)", "")
        self.workers = QLineEdit("http://127.0.0.1:8000")
        self.workers.setPlaceholderText("Comma-separated HTTPS worker URLs, or clear this when using a managed pool")
        form.addRow("Worker URLs", self.workers)
        self.camera = QSpinBox()
        self.camera.setRange(0, 20)
        form.addRow("Camera index", self.camera)
        self.microphone = QSpinBox()
        self.microphone.setRange(-1, 100)
        self.microphone.setValue(-1)
        self.microphone.setSpecialValueText("System default")
        form.addRow("Microphone index", self.microphone)
        self.microphone_delay = QDoubleSpinBox()
        self.microphone_delay.setRange(-500, 500)
        self.microphone_delay.setSuffix(" ms")
        self.microphone_delay.setToolTip("Positive values delay audio. Calibrate with a recorded clap.")
        form.addRow("Microphone sync adjustment", self.microphone_delay)
        self.delay = QDoubleSpinBox()
        self.delay.setRange(5, 10)
        self.delay.setSingleStep(0.5)
        self.delay.setValue(8)
        self.delay.setSuffix(" seconds")
        form.addRow("Fixed delay", self.delay)
        self.duration = QSpinBox()
        self.duration.setRange(0, 86400)
        self.duration.setValue(300)
        self.duration.setSpecialValueText("Unlimited")
        self.duration.setSuffix(" seconds")
        form.addRow("Capture duration", self.duration)
        self.output = QLineEdit("tcp://127.0.0.1:10000?listen=1")
        form.addRow("Output URL or .mp4 path", self.output)
        layout.addLayout(form)
        self.experimental = QCheckBox("Experimental GPU run — appearance and performance are not qualified yet")
        self.synthetic = QCheckBox("Synthetic transport test — use profiles/mock.json and a local mock worker")
        layout.addWidget(self.experimental)
        layout.addWidget(self.synthetic)
        buttons = QHBoxLayout()
        self.start = QPushButton("Start session")
        self.stop = QPushButton("Stop session")
        self.stop.setEnabled(False)
        buttons.addWidget(self.start)
        buttons.addWidget(self.stop)
        layout.addLayout(buttons)
        self.state = QLabel("Ready to configure")
        layout.addWidget(self.state)
        self.picture = QLabel("Processed preview appears after the fixed-delay buffer fills")
        self.picture.setMinimumHeight(200)
        self.picture.setStyleSheet("background: #191e26; color: #dbe2ed; padding: 12px;")
        layout.addWidget(self.picture, 1)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(100)
        self.log.setMaximumHeight(95)
        layout.addWidget(self.log)
        self.setCentralWidget(container)
        self.start.clicked.connect(self.begin)
        self.stop.clicked.connect(self.request_stop)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(250)
        self.poll_timer.timeout.connect(self.poll_runner)
        self.poll_timer.start()

    def poll_runner(self):
        if self.runner:
            status, preview = self.runner.poll()
            if status is not None:
                self.update_status(status)
            if preview is not None:
                self.update_preview(preview)

    def file_field(self, form, label, value):
        field = QLineEdit(value)
        button = QPushButton("Browse")
        row = QHBoxLayout()
        row.addWidget(field)
        row.addWidget(button)

        def choose():
            path, _ = QFileDialog.getOpenFileName(self, label, str(Path.cwd()))
            if path:
                field.setText(path)
        button.clicked.connect(choose)
        form.addRow(label, row)
        return field

    def begin(self):
        try:
            profile = Profile.load(Path(self.profile.text()))
            deployment = Path(self.deployment.text()) if self.deployment.text().strip() else None
            options = ClientOptions(profile=profile, worker_urls=[value.strip() for value in self.workers.text().split(",") if value.strip()],
                                    key_path=Path(self.key.text()), garment_path=Path(self.garment.text()),
                                    camera=self.camera.value(), microphone=None if self.microphone.value() == -1 else self.microphone.value(),
                                    microphone_delay_ms=self.microphone_delay.value(),
                                    synthetic=self.synthetic.isChecked(), no_microphone=self.synthetic.isChecked(),
                                    delay=self.delay.value(), duration=self.duration.value(), output=self.output.text(),
                                    experimental=self.experimental.isChecked(),
                                    benchmark=Path(self.benchmark.text()) if self.benchmark.text().strip() else None)
            self.runner = Runner(options, deployment)
            self.runner.failure.connect(lambda message: QMessageBox.critical(self, "Session stopped", message))
            self.runner.finished.connect(self.finished)
            self.start.setEnabled(False)
            self.stop.setEnabled(True)
            self.runner.start()
        except Exception as exc:
            QMessageBox.critical(self, "Configuration error", str(exc))

    def request_stop(self):
        if self.runner:
            self.runner.stop_event.set()
            self.state.setText("Stopping and releasing session workers…")
            self.stop.setEnabled(False)

    def finished(self):
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        if self.closing:
            self.close()

    def update_status(self, data):
        cost = f"${data['estimated_pool_hourly_price_usd']:.2f}/h" if "estimated_pool_hourly_price_usd" in data else "cost unmeasured"
        self.state.setText(f"{data.get('phase')} · delay {data.get('fixed_delay_seconds', 0):g}s · useful generation {data.get('measured_useful_fps', 0):g} fps · missing {data.get('missing_frames', 0)} · {cost}")
        self.log.append(json.dumps(data))

    def update_preview(self, rgb: np.ndarray):
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.picture.setPixmap(QPixmap.fromImage(image).scaledToWidth(min(1000, self.picture.width())))

    def closeEvent(self, event: QCloseEvent):
        if self.runner and self.runner.isRunning():
            self.closing = True
            self.request_stop()
            event.ignore()
        else:
            event.accept()


def launch():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
