import sys
from PySide6.QtWidgets import QApplication
from data.nd2_reader import read_nd2_file, read_neuron_data, read_neuron_signals, read_cluster
from gui.main_window import MainWindow

def main():
    
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