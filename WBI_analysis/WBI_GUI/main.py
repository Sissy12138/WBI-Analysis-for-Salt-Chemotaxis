import sys
from PySide6.QtWidgets import QApplication
from data.nd2_reader import read_nd2_file, read_neuron_data, read_neuron_signals, read_cluster
from gui.main_window import MainWindow

def main():
    path = r'E:\2025\20250422_WBI'

    # 需要的文件
    ''' 1. 原始荧光文件.nd2
        2. 处理：shifted_id_2_neuron_pos_acrs_t_result.npy文件
        3. 处理钙信号文件：calcium_intensity.npy
        4. 处理idx文件：idx.npy'''
    path_nd2 = path + r'\20250422-016.nd2'
    path_1 = r'Z:\data space+\C. elegans chemotaxis\20250422_WBI\fluro\016\0514'
    path_neuron_data = path_1 + r"\shifted_id_2_neuron_pos_acrs_t_result.npy"
    path_signals = path_1 + r"\calcium_intensity.npy"
    cluster_idx = path_1 +r"\idx.npy"
    n_idx = 20

    images = read_nd2_file(path_nd2)
    n_data = read_neuron_data(path_neuron_data)
    signals = read_neuron_signals(path_signals)
    cluster_neuron = read_cluster(cluster_idx)

    app = QApplication(sys.argv)
    main_window = MainWindow(images, n_data, signals, cluster_neuron)
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()