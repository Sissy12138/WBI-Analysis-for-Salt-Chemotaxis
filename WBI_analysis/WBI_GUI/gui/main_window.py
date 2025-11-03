import sys
import cv2
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QSplitter, QLabel, QSizePolicy, QSlider, QPushButton, QProgressBar, QCheckBox, QGridLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from utils.signal_processor import WBI_LUT

class MainWindow(QMainWindow):
    def __init__(self, images, n_data, signals, cluster_neuron):
        super().__init__()
        self.images = images
        self.n_data = n_data
        signals = signals[cluster_neuron]
        # print(cluster_neuron[29])
        self.signals = signals
        self.cluster_neuron = cluster_neuron
        self.current_frame = 0
        self.current_z = 0
        self.selected_neuron = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.r_image = np.zeros((860, 600, 3), dtype=np.uint8)
        self.g_image = np.zeros((860, 596, 3), dtype=np.uint8)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ND2 Neuron GUI')
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout(self.central_widget)

        # Left side for nd2 images
        self.image_label1 = QLabel()
        self.image_label2 = QLabel()
        self.layout.addWidget(self.image_label1, 0, 0, 1, 2)
        self.layout.addWidget(self.image_label2, 0, 2, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.images) // 2 - 1)
        self.layout.addWidget(self.progress_bar, 1, 0, 1, 4)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.images) // 2 - 1)
        self.slider.valueChanged.connect(self.update_images)
        self.layout.addWidget(self.slider, 2, 0, 1, 3)

        self.frame_label = QLabel(f"{self.current_frame} / {len(self.images) // 2 - 1}")
        self.layout.addWidget(self.frame_label, 2, 3)

        self.manual_button = QPushButton("Manual Play")
        self.manual_button.clicked.connect(self.next_frame)
        self.layout.addWidget(self.manual_button, 3, 0)

        self.auto_button = QPushButton("Auto Play")
        self.auto_button.setCheckable(True)
        self.auto_button.clicked.connect(self.toggle_auto_play)
        self.layout.addWidget(self.auto_button, 3, 1)

        self.max_checkbox = QCheckBox("是否显示max")
        self.max_checkbox.stateChanged.connect(lambda: self.update_images(self.current_frame))
        self.layout.addWidget(self.max_checkbox, 3, 2)

        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self.clear_selection)
        self.layout.addWidget(self.clear_button, 3, 3)

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(self.images[0].shape[0] - 1)
        self.z_slider.valueChanged.connect(self.update_z)
        self.layout.addWidget(self.z_slider, 4, 0, 1, 2)

        self.z_label = QLabel(f"{self.current_z} / {self.images[0].shape[0]}")
        self.layout.addWidget(self.z_label, 4, 3)

        # Right side for neuron signals
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.layout.addWidget(self.right_widget, 0, 4, 5, 1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.right_layout.addWidget(self.scroll_area)

        self.scroll_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_widget)

        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.figure, self.ax = plt.subplots(figsize=(10, len(self.signals) * 2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.canvas.setFixedHeight(len(self.signals) * 100)  # Fixed height for the canvas
        self.scroll_layout.addWidget(self.canvas)

        self.figure_selected, self.ax_selected = plt.subplots(figsize=(10, 3))
        self.canvas_selected = FigureCanvas(self.figure_selected)
        self.right_layout.addWidget(self.canvas_selected)

        self.plot_signals_background()
        self.update_images(0)

        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def numpy_to_pixmap(self, image):
        from PySide6.QtGui import QImage, QPixmap
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def update_images(self, frame):
        self.current_frame = frame
        self.progress_bar.setValue(frame)
        self.frame_label.setText(f"{self.current_frame} / {len(self.images) // 2 - 1}")
        self.update_red_line()
        if self.selected_neuron is not None:
            for line in self.ax_selected.get_lines():
                if line.get_color() == 'r':
                    line.remove()
            self.ax_selected.axvline(x=self.current_frame, color='r', linestyle='--', linewidth=1)
            self.canvas_selected.draw()
            idx_neuron = self.cluster_neuron[self.selected_neuron]
            # idx_neuron = self.selected_neuron
            select_n_data = self.n_data[idx_neuron]
            select_n_pos = select_n_data['match_pos_acrs_vol']
            neuron_pos = select_n_pos[self.current_frame]
            
            if np.isnan(neuron_pos[2]):
                if self.max_checkbox.isChecked():
                    self.r_image[:, :, 0] = WBI_LUT(np.max(np.array(self.images[2*frame]).astype(np.uint8), axis=0)[:860, :])
                    self.g_image[:, :, 1] = WBI_LUT(np.max(np.array(self.images[2*frame + 1]).astype(np.uint8), axis=0)[:860, 4:])
                else:
                    self.r_image[:, :, 0] = WBI_LUT(np.array(self.images[2*frame][int(self.current_z)]).astype(np.uint8)[:860, :])
                    self.g_image[:, :, 1] = WBI_LUT(np.array(self.images[2*frame + 1][int(self.current_z)]).astype(np.uint8)[:860, 4:])
                self.r_image = cv2.putText(self.r_image, 'NAN', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                self.g_image = cv2.putText(self.g_image, 'NAN', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                self.current_z = neuron_pos[2]
                self.z_label.setText(f"{self.current_z} / {self.images[0].shape[0]}")
                if self.max_checkbox.isChecked():
                    self.r_image[:, :, 0] = WBI_LUT(np.max(np.array(self.images[2*frame]).astype(np.uint8), axis=0)[:860, :])
                    self.g_image[:, :, 1] = WBI_LUT(np.max(np.array(self.images[2*frame + 1]).astype(np.uint8), axis=0)[:860, 4:])
                else:
                    self.r_image[:, :, 0] = WBI_LUT(np.array(self.images[2*frame][int(self.current_z)]).astype(np.uint8)[:860, :])
                    self.g_image[:, :, 1] = WBI_LUT(np.array(self.images[2*frame + 1][int(self.current_z)]).astype(np.uint8)[:860, 4:])
                self.r_image = cv2.circle(self.r_image, (int(neuron_pos[0]), int(neuron_pos[1])), 10, (255, 0, 0), 2)
                self.g_image = cv2.circle(self.g_image, (int(neuron_pos[0]), int(neuron_pos[1])), 10, (0, 255, 0), 2)
        else:
            if self.max_checkbox.isChecked():
                self.r_image[:, :, 0] = WBI_LUT(np.max(np.array(self.images[2*frame]).astype(np.uint8), axis=0)[:860, :])
                self.g_image[:, :, 1] = WBI_LUT(np.max(np.array(self.images[2*frame + 1]).astype(np.uint8), axis=0)[:860, 4:])
            else:
                self.z_label.setText(f"{self.current_z} / {self.images[0].shape[0]}")
                self.r_image[:, :, 0] = WBI_LUT(np.array(self.images[2*frame][int(self.current_z)]).astype(np.uint8)[:860, :])
                self.g_image[:, :, 1] = WBI_LUT(np.array(self.images[2*frame + 1][int(self.current_z)]).astype(np.uint8)[:860, 4:])
        self.image_label1.setPixmap(self.numpy_to_pixmap(self.r_image))
        self.image_label2.setPixmap(self.numpy_to_pixmap(self.g_image))

    def update_z(self, z):
        self.current_z = z
        self.update_images(self.current_frame)

    def plot_signals_background(self):
        self.ax.clear()
        for i, signal in enumerate(self.signals):
            blur_signal = cv2.blur(signal, (7, 7))
            self.ax.plot(blur_signal * 10 + i * 10, color='blue', linewidth=1)  # Offset each signal by 20 units vertically
        self.ax.set_yticks([i * 10 for i in range(len(self.signals))])
        self.ax.set_yticklabels([f'Neuron {i}' for i in range(len(self.signals))])
        self.canvas.draw()

    def update_red_line(self):
        # Remove the previous red line
        for line in self.ax.get_lines():
            if line.get_color() == 'r':
                line.remove()
        # Add the new red vertical line
        self.ax.axvline(x=self.current_frame, color='r', linestyle='--')
        self.canvas.draw()

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % (len(self.images) // 2)
        self.slider.setValue(self.current_frame)

    def toggle_auto_play(self):
        if self.auto_button.isChecked():
            self.timer.start(300)  # Adjust the interval as needed
        else:
            self.timer.stop()

    def on_canvas_click(self, event):
        if event.inaxes == self.ax:
            neuron_idx = int(event.ydata // 10)
            if 0 <= neuron_idx < len(self.signals):
                # neuron_idx = np.where(self.cluster_neuron == neuron_idx)[0][0]
                self.selected_neuron = neuron_idx
                self.select_signals_background(neuron_idx)
                self.update_images(self.current_frame)

    def select_signals_background(self, idx):
        self.ax_selected.clear()
        signal = self.signals[idx]
        blur_signal = cv2.blur(signal, (7, 7))
        self.ax_selected.scatter(range(len(signal)), signal, s=1, c='gray')
        self.ax_selected.plot(blur_signal, color='blue', linewidth=1)
        self.ax_selected.set_title(f'Neuron {idx}')
        self.ax_selected.axvline(x=self.current_frame, color='r', linestyle='--')
        self.canvas_selected.draw()

    def clear_selection(self):
        self.selected_neuron = None
        self.ax_selected.clear()
        self.canvas_selected.draw()
        self.update_images(self.current_frame)