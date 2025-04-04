# gui.py
import sys
import os
import traceback

# --- Use PyQt5 ---
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QSizePolicy, QSpacerItem,
                             QFrame)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QPixmap, QColor, QPalette, QIcon # Added QIcon

# Import functions from your existing project files
try:
    from calibration import collect_calibration_data, fine_tune_model
    from detection import run_detection
    from utils import (BASE_MODEL_PATH, PERSONALIZED_MODEL_PATH,
                      FACE_CASCADE_PATH, CALIBRATION_DATA_DIR, IMAGE_DIR) # Use IMAGE_DIR
    PROJECT_FUNCTIONS_IMPORTED = True
except ImportError as e:
    print(f"[ERROR] Failed to import project modules: {e}")
    print("Please ensure calibration.py, detection.py, and utils.py are in the same directory.")
    PROJECT_FUNCTIONS_IMPORTED = False
    # Dummy functions if imports fail
    def collect_calibration_data(): print("Dummy collect_calibration_data"); return False
    def fine_tune_model(): print("Dummy fine_tune_model")
    def run_detection(use_personalized): print(f"Dummy run_detection(use_personalized={use_personalized})")
    BASE_MODEL_PATH = 'base_model.pth'
    PERSONALIZED_MODEL_PATH = 'personalized_model.pth'
    FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
    CALIBRATION_DATA_DIR = 'calibration_data'
    IMAGE_DIR = 'images'


# --- Worker Thread (Unchanged) ---
class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self._is_running = True

    @pyqtSlot()
    def run(self):
        try:
            if self.mode == 'calibrate':
                self.progress.emit("[Calibration] Starting...")
                if not os.path.exists(BASE_MODEL_PATH):
                     raise FileNotFoundError(f"Base model '{BASE_MODEL_PATH}' not found.")
                collection_success = collect_calibration_data()
                if collection_success:
                    data_found = False
                    if os.path.exists(CALIBRATION_DATA_DIR):
                         for _, dirs, files in os.walk(CALIBRATION_DATA_DIR):
                             if files: data_found = True; break
                    if data_found:
                         self.progress.emit("[Calibration] Data collected. Fine-tuning...")
                         fine_tune_model()
                         self.progress.emit("[Calibration] Fine-tuning complete.")
                    else:
                         self.progress.emit("[Calibration] Data collection OK, but no valid data found. Fine-tuning skipped.")
                else:
                    self.progress.emit("[Calibration] Aborted or failed by user. Fine-tuning skipped.")
            elif self.mode == 'detect_personalized':
                self.progress.emit("[Detection] Starting Personalized...")
                if not os.path.exists(PERSONALIZED_MODEL_PATH):
                     raise FileNotFoundError(f"Personalized model '{PERSONALIZED_MODEL_PATH}' not found.")
                run_detection(use_personalized=True)
                self.progress.emit("[Detection] Personalized window closed.")
            elif self.mode == 'detect_base':
                self.progress.emit("[Detection] Starting Base Model...")
                if not os.path.exists(BASE_MODEL_PATH):
                     raise FileNotFoundError(f"Base model '{BASE_MODEL_PATH}' not found.")
                run_detection(use_personalized=False)
                self.progress.emit("[Detection] Base Model window closed.")
            else:
                 raise ValueError(f"Unknown worker mode: {self.mode}")
        except Exception as e:
            error_details = traceback.format_exc()
            self.error.emit(f"Error in {self.mode}:\n{e}\n---\n{error_details}")
        finally:
            self.finished.emit()
    def stop(self): self._is_running = False

# --- Main Application Window ---
class EmotionRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.initUI()
        self.apply_stylesheet()
        # Use a QTimer for delayed file checks to ensure GUI is shown first
        QTimer.singleShot(100, self.check_files_and_log)

    def initUI(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle('Personal Expression Recognition')
        self.setMinimumSize(900, 620) # Slightly larger minimum size

        # --- Main Layout: Horizontal ---
        mainLayout = QHBoxLayout(self)
        mainLayout.setContentsMargins(0, 0, 0, 0) # No margins for main layout
        mainLayout.setSpacing(0) # No spacing for main layout

        # --- Left Panel (Image Area) ---
        leftPanelWidget = QWidget()
        leftPanelWidget.setObjectName("LeftPanel")
        leftPanelLayout = QVBoxLayout(leftPanelWidget)
        leftPanelLayout.setContentsMargins(30, 30, 30, 30) # Padding inside left panel
        leftPanelLayout.setAlignment(Qt.AlignCenter) # Center content

        self.imageLabel = QLabel()
        self.imageLabel.setObjectName("FaceImageLabel")
        image_path = os.path.join(IMAGE_DIR, "face.png")
        if os.path.exists(image_path):
             pixmap = QPixmap(image_path)

             scaled_pixmap = pixmap.scaled(QSize(450, 550), Qt.KeepAspectRatio, Qt.SmoothTransformation)
             self.imageLabel.setPixmap(scaled_pixmap)
             self.imageLabel.setAlignment(Qt.AlignCenter)
        else:
             self.imageLabel.setText("⚠️\nface.png not found\nin 'images' folder")
             self.imageLabel.setAlignment(Qt.AlignCenter)
             self.imageLabel.setFont(QFont("Segoe UI", 14))

        leftPanelLayout.addWidget(self.imageLabel)

        # --- Right Panel (Controls Area) ---
        rightPanelWidget = QWidget()
        rightPanelWidget.setObjectName("RightPanel")
        rightPanelLayout = QVBoxLayout(rightPanelWidget)
        rightPanelLayout.setContentsMargins(30, 25, 30, 30) # Padding inside right panel
        rightPanelLayout.setSpacing(18) # Spacing between elements

        # Title
        titleLabel = QLabel('Expression Recognition')
        titleLabel.setObjectName("TitleLabel")
        titleLabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        titleLabel.setWordWrap(True)

        # Status Area Label
        statusLabel = QLabel("Status Log:")
        statusLabel.setObjectName("StatusLabel")

        # Status Area
        self.statusTextEdit = QTextEdit()
        self.statusTextEdit.setObjectName("StatusArea")
        self.statusTextEdit.setReadOnly(True)
        self.statusTextEdit.setPlaceholderText("System messages will appear here...")
        self.statusTextEdit.setMinimumHeight(150) # Ensure decent minimum height
        self.statusTextEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow expanding

        # Buttons Area Label
        buttonsLabel = QLabel("Actions:")
        buttonsLabel.setObjectName("ButtonsLabel")

        # Buttons Layout (Vertical this time for better spacing perhaps)
        buttonLayout = QVBoxLayout()
        buttonLayout.setSpacing(12)

        self.calibrateButton = QPushButton(' 📊 Calibrate & Fine-tune Model')
        self.detectPersonalizedButton = QPushButton(' 😎 Run Personalized Detection')
        self.detectBaseButton = QPushButton(' 🤖 Run Base Model Detection')

        self.calibrateButton.setObjectName("ActionButton")
        self.detectPersonalizedButton.setObjectName("ActionButton")
        self.detectBaseButton.setObjectName("ActionButton")

        # Set fixed height and allow horizontal expansion
        button_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.calibrateButton.setSizePolicy(button_size_policy)
        self.detectPersonalizedButton.setSizePolicy(button_size_policy)
        self.detectBaseButton.setSizePolicy(button_size_policy)

        buttonLayout.addWidget(self.calibrateButton)
        buttonLayout.addWidget(self.detectPersonalizedButton)
        buttonLayout.addWidget(self.detectBaseButton)
        buttonLayout.addStretch(1) # Push buttons up

        # Assemble Right Panel
        rightPanelLayout.addWidget(titleLabel)
        rightPanelLayout.addWidget(statusLabel)
        rightPanelLayout.addWidget(self.statusTextEdit)
        rightPanelLayout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)) # Spacer
        rightPanelLayout.addWidget(buttonsLabel)
        rightPanelLayout.addLayout(buttonLayout)

        # --- Assemble Main Layout ---
        mainLayout.addWidget(leftPanelWidget, 3)  # Left panel takes 3 parts stretch
        mainLayout.addWidget(rightPanelWidget, 4) # Right panel takes 4 parts stretch

        # --- Connect Signals ---
        self.calibrateButton.clicked.connect(self.run_calibration_task)
        self.detectPersonalizedButton.clicked.connect(self.run_detect_personalized_task)
        self.detectBaseButton.clicked.connect(self.run_detect_base_task)

    def apply_stylesheet(self):
        styleSheet = """
            /* Main Window */
            #MainWindow {
                background-color: #FFFFFF; /* Clean white background */
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }

            /* Left Panel with Image */
            #LeftPanel {
                background-color: #F8F9FA; /* Very light gray */
                border-right: 1px solid #E0E0E0; /* Subtle separator line */
            }
            #FaceImageLabel {
                 /* border: 1px dashed #cccccc; */ /* Optional dashed border */
                 border-radius: 8px;
            }

            /* Right Panel with Controls */
            #RightPanel {
                background-color: #FFFFFF; /* White */
            }

            /* Main Title */
            #TitleLabel {
                font-size: 22pt;
                font-weight: 600; /* Semibold */
                color: #2C3E50; /* Dark blue-gray */
                margin-bottom: 5px;
            }

            /* Section Labels (Status, Actions) */
            #StatusLabel, #ButtonsLabel {
                font-size: 11pt;
                font-weight: 600; /* Semibold */
                color: #7F8C8D; /* Medium gray */
                margin-top: 10px;
                margin-bottom: 4px;
            }

            /* Status Text Area */
            #StatusArea {
                background-color: #FDFEFE; /* Very slightly off-white */
                border: 1px solid #EAECEE; /* Light border */
                border-radius: 6px;
                font-size: 9.5pt; /* Slightly smaller for logs */
                color: #34495E; /* Dark gray text */
                padding: 10px;
                font-family: "Consolas", "Courier New", monospace; /* Monospace font for logs */
            }

            /* Action Buttons */
            #ActionButton {
                background-color: #5DADE2; /* Primary blue */
                color: white;
                font-size: 11pt;
                font-weight: 500; /* Medium weight */
                padding: 13px 18px;
                border: none;
                border-radius: 6px;
                text-align: left; /* Align text left */
                min-height: 45px;
                outline: none; /* Remove focus outline */
                /* Transition effect (subtle) */
                /* Note: Transitions are less supported in basic QSS */
            }
            #ActionButton:hover {
                background-color: #3498DB; /* Slightly darker blue on hover */
            }
            #ActionButton:pressed {
                background-color: #2874A6; /* Darker blue when pressed */
            }
            #ActionButton:disabled {
                background-color: #EAECEE; /* Light gray when disabled */
                color: #BDC3C7;
            }

            /* Style Scrollbars */
            QScrollBar:vertical {
                border: none;
                background: #F4F6F7;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #D5DBDB;
                min-height: 25px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: #BFC9CA;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px; border: none; background: none;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                 background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                 background: none;
            }

            QScrollBar:horizontal {
               border: none;
               background: #F4F6F7;
               height: 10px;
               margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:horizontal {
                background: #D5DBDB;
                min-width: 25px;
                border-radius: 5px;
            }
             QScrollBar::handle:horizontal:hover {
                background: #BFC9CA;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                 width: 0px; border: none; background: none;
            }
            QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {
                 background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                 background: none;
            }
        """
        self.setStyleSheet(styleSheet)

    @pyqtSlot()
    def check_files_and_log(self):
        """Checks for essential files and logs status after GUI is shown."""
        self.log_status("System initialized. Checking files...")
        if not PROJECT_FUNCTIONS_IMPORTED:
             self.log_status("[ERROR] Project modules failed to import. Functionality limited.", error=True)

        if not os.path.exists(FACE_CASCADE_PATH):
             self.log_status(f"Required file missing: {os.path.basename(FACE_CASCADE_PATH)}", error=True)
        else:
             self.log_status(f"Face detector found.", ok=True)

        if not os.path.exists(BASE_MODEL_PATH):
             self.log_status(f"Base model not found: '{os.path.basename(BASE_MODEL_PATH)}'. Cannot run detection or calibration.")
        else:
             self.log_status(f"Base model found.", ok=True)

        if not os.path.exists(PERSONALIZED_MODEL_PATH):
             self.log_status(f"Personalized model not found: '{os.path.basename(PERSONALIZED_MODEL_PATH)}'. Run calibration first.")
        else:
             self.log_status(f"Personalized model found.", ok=True)

        if not os.path.exists(os.path.join(IMAGE_DIR, "face.png")):
             self.log_status(f"'face.png' not found in '{IMAGE_DIR}' folder.", error=True)


    @pyqtSlot(str)
    def log_status(self, message, error=False, ok=False):
        """Appends a styled message to the status text edit."""
        if error:
             prefix = "[ERROR] "
             color = "#E74C3C" # Red
        elif ok:
             prefix = "[OK] "
             color = "#2ECC71" # Green
        else:
             prefix = "[INFO] "
             color = "#34495E" # Dark Gray

        # Using HTML for coloring
        self.statusTextEdit.append(f'<font color="{color}">{prefix}{message}</font>')
        # Auto-scroll to the bottom
        cursor = self.statusTextEdit.textCursor()
        cursor.movePosition(cursor.End)
        self.statusTextEdit.setTextCursor(cursor)

    def set_buttons_enabled(self, enabled):
        self.calibrateButton.setEnabled(enabled)
        self.detectPersonalizedButton.setEnabled(enabled)
        self.detectBaseButton.setEnabled(enabled)

    def start_worker_thread(self, mode):
        if self.thread is not None and self.thread.isRunning():
             self.log_status("Task already running. Please wait.", error=True)
             return

        error_msg = None
        if mode == 'calibrate' and not os.path.exists(BASE_MODEL_PATH):
             error_msg = f"Cannot start Calibration: Base model missing."
        elif mode == 'detect_personalized' and not os.path.exists(PERSONALIZED_MODEL_PATH):
             error_msg = f"Cannot start Personalized Detection: Model missing."
        elif mode == 'detect_base' and not os.path.exists(BASE_MODEL_PATH):
             error_msg = f"Cannot start Base Detection: Base model missing."
        if mode in ['calibrate', 'detect_personalized', 'detect_base'] and not os.path.exists(FACE_CASCADE_PATH):
             error_msg = f"Cannot start: Face cascade file missing."

        if error_msg:
            self.log_status(error_msg, error=True)
            return

        self.set_buttons_enabled(False)
        # self.log_status(f"Starting task: {mode}...") # Worker sends specific starting message

        self.thread = QThread()
        self.worker = Worker(mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_task_finished)
        self.worker.error.connect(self.on_task_error)
        self.worker.progress.connect(self.log_status)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @pyqtSlot()
    def on_task_finished(self):
        self.log_status("Task finished.") # Worker already sent specific finished msg
        self.set_buttons_enabled(True)
        self.thread = None
        self.worker = None

    @pyqtSlot(str)
    def on_task_error(self, error_message):
        self.log_status(f"{error_message}", error=True)
        self.set_buttons_enabled(True)
        self.thread = None
        self.worker = None

    @pyqtSlot()
    def run_calibration_task(self): self.start_worker_thread('calibrate')
    @pyqtSlot()
    def run_detect_personalized_task(self): self.start_worker_thread('detect_personalized')
    @pyqtSlot()
    def run_detect_base_task(self): self.start_worker_thread('detect_base')

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
             self.thread.quit()
             self.thread.wait(500)
        event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    # Enable High DPI scaling for better visuals on modern displays
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    ex = EmotionRecognizerApp()
    ex.show()
    sys.exit(app.exec_())