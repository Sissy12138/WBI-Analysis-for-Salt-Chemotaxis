import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QFileDialog, QPushButton, QWidget, QGridLayout
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen
import numpy as np
from pims import ND2Reader_SDK
import pandas as pd
from PySide6.QtCore import Qt, QTimer, QPoint
import cv2 as cv
import os
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class RenderArea(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.background, = self.axes.plot([], [], 'b-', lw=2) 
        self.point, = self.axes.plot([], [], 'ro')
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plot_background(self, x, y):
        self.background.set_data(x, y)
        self.draw()

    def update_point(self, x, y):
        self.point.set_data([x], [y])
        self.draw()

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(211)
        self.axes2 = fig.add_subplot(212)
        self.background2, = self.axes2.plot([], [], 'b-', lw=1) 
        self.point2, = self.axes2.plot([], [], 'ro')
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot_background2(self, x, y):
        self.background2.set_data(x, y)
        self.draw()

    def update_point2(self, x, y):
        self.point2.set_data([x], [y])
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ND2 Viewer")
        self.setGeometry(100, 100, 1800, 1000)
        central_widget = QWidget(self) 
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)

        self.index1 = 0
        self.index2 = 0
        self.index3 = 0
        self.nd_num = -1
        height = 832
        width = 600
        self.r_image = np.ones((height, width, 3), dtype=np.uint8)
        self.g_image = np.ones((height, width, 3), dtype=np.uint8)

        self.image_630 = QLabel(self)
        self.image_630.setFixedSize(width, height)

        self.image_525 = QLabel(self)
        self.image_525.setFixedSize(width, height)

        self.image_video = QLabel(self)
        self.image_video.setFixedSize(512, 512)

        self.openButton = QPushButton(self)
        self.openButton.setText('Open')
        self.openButton.clicked.connect(self.openFile)

        self.playButton = QPushButton(self)
        self.playButton.setText('Play')
        self.playButton.clicked.connect(self.playVideo)

        self.slider_frame = QSlider(Qt.Horizontal, self)
        self.slider_frame.setValue(0)
        self.slider_frame.setTickInterval(1)
        self.slider_frame.sliderMoved.connect(self.t_slide)
        self.slider_frame.sliderPressed.connect(self.t_slide)

        self.z_step = QSlider(Qt.Horizontal, self)
        self.z_step.setValue(0)
        self.z_step.setTickInterval(1)
        self.z_step.setTickPosition(QSlider.TicksBothSides)
        self.z_step.sliderMoved.connect(self.t_z)
        self.z_step.sliderPressed.connect(self.t_z)

        self.volume_num = QSlider(Qt.Horizontal, self)
        self.volume_num.setValue(0)
        self.volume_num.setTickInterval(1)
        self.volume_num.setTickPosition(QSlider.TicksBothSides)
        self.volume_num.sliderMoved.connect(self.t_z)
        self.volume_num.sliderPressed.connect(self.t_z)

        self.reander_area = RenderArea(self, width=5, height=5, dpi=100)

        self.zPlay_button = QPushButton(self)
        self.zPlay_button.setText('Next Z')
        self.zPlay_button.clicked.connect(self.zPlay)

        self.volumePlay_button = QPushButton(self)
        self.volumePlay_button.setText('Next Volume')
        self.volumePlay_button.clicked.connect(self.volumePlay)

        self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
        
        layout.addWidget(self.openButton, 0, 0)
        layout.addWidget(self.playButton, 0, 1)
        layout.addWidget(self.image_630, 0, 2, 2, 1)
        layout.addWidget(self.image_525, 0, 3, 2, 1)
        layout.addWidget(self.image_video, 3, 2, 3, 1)
        layout.addWidget(self.slider_frame, 5, 0, 1 ,2)
        layout.addWidget(self.reander_area, 3, 3, 3, 1)

        layout.addWidget(self.canvas, 1, 0, 3, 2)

        nd2_layout = QGridLayout()
        nd2_layout.addWidget(self.z_step, 0, 0)
        nd2_layout.addWidget(self.zPlay_button, 0, 1)
        nd2_layout.addWidget(self.volume_num, 1, 0)
        nd2_layout.addWidget(self.volumePlay_button, 1, 1)
        layout.addLayout(nd2_layout, 4, 0, 1, 2)

        self.timer = QTimer() 
        self.timer.timeout.connect(self.updateFrame)
    
    def openFile(self):
        file_name  = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.videoFile = cv.VideoCapture(file_name + '\\raw_data\\c1.mp4')
        self.t_c = pd.read_csv(file_name + '\\raw_data\\c1.txt', header = None)[1]
        self.data_s = pd.read_csv(file_name + '\\raw_data\\s-data.txt', header = None)
        self.t_s,self.nd_file, self.z, self.f = self.stack_time(self.data_s, file_name)
        self.data_stage = pd.read_csv(file_name + '\\raw_data\\stage_data.txt', header = None).values
        self.t_stage = self.data_stage[:, 0]
        
        self.x_data = self.data_stage[:, 3] / 1000
        self.y_data = self.data_stage[:, 4] / 1000
        # cal v
        dx = np.diff(self.x_data)
        dy = np.diff(self.y_data)
        dt = np.diff(self.t_stage / 1000)
        # 计算速度
        v = np.sqrt(dx ** 2 + dy ** 2) / dt
        # 计算移动平均值
        self.v_m = np.convolve(v, np.ones(60)/60, mode='valid')

        self.reander_area.axes.set_xlim(np.min(self.x_data), np.max(self.x_data))
        self.reander_area.axes.set_ylim(np.min(self.y_data), np.max(self.y_data))
        self.reander_area.plot_background(self.x_data, self.y_data)

        self.slider_frame.setMaximum(self.t_c.iloc[-1])
        self.slider_frame.setMinimum(self.t_c.iloc[0])

        self.canvas.axes.imshow(np.load(file_name + '\\calcium_intensity.npy'))
        self.canvas.axes.set_aspect('auto')
        self.canvas.axes2.set_xlim(np.min(self.t_stage[0:-1]), np.max(self.t_stage[0:-1]))
        self.canvas.axes2.set_ylim(np.min(self.v_m), np.max(self.v_m))
        self.canvas.plot_background2(self.t_stage[30:-30], self.v_m)

        self.t = self.t_c.iloc[0]
        self.t_frame = np.array([self.t_c.iloc[0], self.t_s[0][0], self.t_stage[0]])
        self.t_index = [1, 0, 0]

    def t_show(self, t):
        data1 = self.t_c >= t
        self.index1 = data1[data1 == 1].index[0]
        self.t_frame[0] = self.t_c.iloc[self.index1]
        self.videoFile.set(cv.CAP_PROP_POS_FRAMES, self.index1)
        self.load_frame()
        if t > self.t_stage[-1]:
            self.index3 = len(self.t_stage)
            self.t_frame[2] = self.t_c.iloc[-1]
        else:
            data1 = self.t_stage >= t
            self.index3 = np.where(data1 == True)[0][0]
            self.t_frame[2] = self.t_stage[self.index3]
            self.reander_area.update_point(self.x_data[self.index3], self.y_data[self.index3])
            if self.index3 >= 30 and self.index3 <= len(self.t_stage) - 30:
                    self.canvas.update_point2(self.t_stage[self.index3], self.v_m[self.index3 - 30])

        if t < self.t_s[0][0]:
            self.image_525.setPixmap(QPixmap())
            self.image_630.setPixmap(QPixmap())
            self.index2 = 0
            self.t_frame[1] = self.t_s[self.index2][0]
            self.z_step.setValue(0)
            self.volume_num.setValue(0)
        elif t > self.t_s[-1][0]:
            self.image_525.setPixmap(QPixmap())
            self.image_630.setPixmap(QPixmap())
            self.index2 = len(self.t_s)
            self.t_frame[1] = self.t_c.iloc[-1]
            self.z_step.setValue(0)
            self.volume_num.setValue(0)
        else:
            data1 = self.t_s[:,0] >= t
            self.index2 = np.where(data1 == True)[0][0]
            if self.t_s[self.index2 - 1][1] != self.t_s[self.index2][1]:
                self.image_525.setPixmap(QPixmap())
                self.image_630.setPixmap(QPixmap())
                self.z_step.setValue(0)
                self.volume_num.setValue(0)
            else:
                c_nd_num = self.t_s[self.index2][1]
                if self.nd_num != c_nd_num:
                    t_nd = self.t_s[np.where(self.t_s[:, 1] == c_nd_num), 0][0]
                    print(t_nd[0])
                    data1 = self.t_stage > t_nd[0]
                    f_start = np.where(data1 == 1)[0][0] - 1
                    data1 = self.t_stage > t_nd[-1]
                    f_stop = np.where(data1 == 1)[0][0]
                    self.reander_area.axes.set_xlim(np.min(self.x_data[f_start:f_stop]), np.max(self.x_data[f_start:f_stop]))
                    self.reander_area.axes.set_ylim(np.min(self.y_data[f_start:f_stop]), np.max(self.y_data[f_start:f_stop]))
                    self.reander_area.plot_background(self.x_data[f_start:f_stop], self.y_data[f_start:f_stop])
                    self.nd_num = c_nd_num
                self.load_image(self.nd_file[int(self.t_s[self.index2][1])], self.t_s[self.index2][2])
                z = self.z[int(self.t_s[self.index2][1])]
                self.z_step.setMaximum(z -1)
                self.z_step.setMinimum(0)
                self.z_step.setValue(int(self.t_s[self.index2][2] % z))
                f = self.f[int(self.t_s[self.index2][1])]
                self.volume_num.setMaximum(f)
                self.volume_num.setMinimum(0)
                self.volume_num.setValue(int(self.t_s[self.index2][2] / z))
                for lines in self.canvas.axes.lines:
                    lines.remove()
                self.canvas.axes.axvline(x=self.volume_num.value(), color='r', linestyle='--')
                self.canvas.draw()
            self.t_frame[1] = self.t_s[self.index2][0]
        self.t_index = self.t_frame == np.min(self.t_frame)
    
    def t_slide(self):
        self.timer.stop()
        self.t = self.slider_frame.value()
        self.t_show(self.t)
    
    def stack_time(self, s_data, path):
        nd2_files = []
        nd2_start = []
        nd2_l = []
        nd2_z = []
        nd2_f = []
        for file in os.listdir(path):
            if file.endswith('.nd2'):
                nd2_file, nd2_t, nd2_st, z, f = self.load_nd2(os.path.join(path, file))
                nd2_files.append(nd2_file)
                nd2_l.append(nd2_t)
                nd2_start.append(nd2_st)
                nd2_z.append(z)
                nd2_f.append(f)
        nd2_sort = np.argsort(nd2_start)
        nd2_files = [nd2_files[idx] for idx in nd2_sort]
        nd2_l = [nd2_l[idx] for idx in nd2_sort]
        nd2_z= [nd2_z[idx] for idx in nd2_sort]
        nd2_f= [nd2_f[idx] for idx in nd2_sort]

        s_start = s_data[s_data[1] == 2].index
        t_data = s_data[0]
        s_stop = s_data[s_data[1] == 3].index
        s_s = []
        for i in s_start:
            data = (s_stop >= i)
            s_s.append(s_stop[data == 1][0])
        
        t_length = (t_data[s_s].reset_index(drop=True) - t_data[s_start].reset_index(drop=True)) / 1000
        t_length = t_length.values
        t_s = np.empty([0,3])
        m = 0
        for i in range(len(s_start)):
            if np.abs(t_length[i] - nd2_l[m]) < 5:
                t_stamp = s_data[1][s_start[i] + 1]
                s_num = t_stamp % 100
                if nd2_z[m] == s_num:
                    s_len = int (round((t_stamp / 100) / s_num))
                    t_array = np.array(list(range(s_num) )) * s_len
                    t_l = int(round((s_data[0][s_start[i] + 1 : s_s[i] - 1]).diff().sum(skipna=True) / (s_s[i] - s_start[i] - 2)))
                    t_step = np.concatenate([pd.Series([s_data[0][s_start[i]] - t_l *2]), pd.Series([s_data[0][s_start[i]] - t_l]), s_data[0][s_start[i]: s_s[i]-1]])
                    tt = []
                    for t in t_step:
                        tt = np.concatenate([tt, t + t_array])
                    tt = np.column_stack((tt, np.ones((len(tt))) * m, np.arange(len(tt))))
                    t_s = np.vstack((t_s, tt))
                    m += 1
                    if m == len(nd2_l): break
        return t_s, nd2_files, nd2_z, nd2_f

    def zPlay(self):
        self.index2 += 1
        self.t = self.t_s[self.index2][0]
        self.t_show(self.t)
        self.slider_frame.setValue(self.t)
    
    def volumePlay(self):
        z = self.z[int(self.t_s[self.index2][1])]
        self.index2 += z
        self.t = self.t_s[self.index2][0]
        self.t_show(self.t)
        self.slider_frame.setValue(self.t)
    
    def t_z(self):
        self.timer.stop()
        dt =  self.volume_num.value() * self.z[int(self.t_s[self.index2][1])] + self.z_step.value() - self.t_s[self.index2][2]
        self.index2 += int(dt)
        self.t = self.t_s[self.index2][0]
        self.t_show(self.t)
        self.slider_frame.setValue(self.t)

    def playVideo(self):
        if self.videoFile:
            if not self.timer.isActive(): 
                self.timer.start(10)
            else:
                self.timer.stop()

    def updateFrame(self):
        t_index = self.t_index
        if t_index[0]:
            self.load_frame()
            self.index1 += 1
            if self.index1 == len(self.t_c) - 1:
                self.timer.stop()
                self.index1 = 0
                self.index2 = 0
                self.index3 = 0
                self.t_frame = np.array([self.t_c.iloc[0], self.t_s[0][0], self.t_stage[0]])
                self.videoFile.set(cv.CAP_PROP_POS_FRAMES, self.index1)
                self.slider_frame.setValue(self.t_c.iloc[0])
                self.t = self.slider_frame.value()
                self.t_index = [1,0,0]
                return
            else:
                self.t_frame[0] = self.t_c[self.index1]

        if t_index[1]:
            self.load_image(self.nd_file[int(self.t_s[self.index2][1])], self.t_s[self.index2][2])
            z = self.z[int(self.t_s[self.index2][1])]
            self.z_step.setMaximum(z -1)
            self.z_step.setMinimum(0)
            self.z_step.setValue(int(self.t_s[self.index2][2] % z))
            f = self.f[int(self.t_s[self.index2][1])]
            self.volume_num.setMaximum(f)
            self.volume_num.setMinimum(0)
            self.volume_num.setValue(int(self.t_s[self.index2][2] / z))
            for lines in self.canvas.axes.lines:
                lines.remove()
            self.canvas.axes.axvline(x=self.volume_num.value(), color='r', linestyle='--')
            self.canvas.draw()
            self.index2 += 1
            if self.index2 == len(self.t_s):
                self.t_frame[1] = self.t_c.iloc[-1]
                self.image_525.setPixmap(QPixmap())
                self.image_630.setPixmap(QPixmap())
            else:
                if self.t_s[self.index2 - 1][1] != self.t_s[self.index2][1]:
                    self.image_525.setPixmap(QPixmap())
                    self.image_630.setPixmap(QPixmap()) 
                self.t_frame[1] = self.t_s[self.index2][0]

        if t_index[2]:
            self.reander_area.update_point(self.x_data[self.index3], self.y_data[self.index3])
            if self.index3 >= 30 and self.index3 <= len(self.t_stage) - 30:
                    self.canvas.update_point2(self.t_stage[self.index3], self.v_m[self.index3 - 30])
            self.index3 += 1
            if self.index3 == len(self.t_stage):
                self.t_frame[2] = self.t_c.iloc[-1]
            else:
                self.t_frame[2] = self.t_stage[self.index3]
        self.timer.start(np.min(self.t_frame) - self.t)
        self.t = np.min(self.t_frame)
        self.slider_frame.setValue(self.t)
        self.t_index = self.t_frame == self.t

    def load_frame(self):
        ret, frame = self.videoFile.read()
        frame = cv.resize(frame, (512,512))
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
        else:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_video.setPixmap(pixmap)

    def load_nd2(self, file_path):
        images = ND2Reader_SDK(file_path)
        mdata = images.metadata
        f = images.shape[0]
        t = f / mdata['frame_rate']
        z = images.shape[1]
        start_t = mdata['time_start']
        images.bundle_axes = 'yx'
        images.iter_axes = 'tzc'
        
        return images, t, start_t.timestamp(), z, f

    def load_image(self, images, frame_num):
        image_data = np.array(images[2*frame_num]).astype(np.uint8)
        image_crop = image_data[184:1016, :]
        # image_crop = np.flip(np.transpose(image_crop))
        height, width = image_crop.shape
        self.r_image[:, :, 0] = image_crop / (np.max(image_crop) - np.min(image_crop)) * 255
        bytes_per_line = 3 * width
        self.image_630.setPixmap(QPixmap.fromImage(QImage(self.r_image, width, height, bytes_per_line, QImage.Format_RGB888)))
        image_data1 = np.array(images[2*frame_num + 1]).astype(np.uint8)
        image_crop1 = image_data1[184:1016, :]
        # image_crop1 = np.flip(np.transpose(image_crop1))
        self.g_image[:, :, 1] = image_crop1 / (np.max(image_crop1) - np.min(image_crop1)) * 255
        self.image_525.setPixmap(QPixmap.fromImage(QImage(self.g_image, width, height, bytes_per_line, QImage.Format_RGB888)))

    def closeEvent(self, event):
        for i in self.nd_file: i.close
        self.videoFile.release()
        return super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())