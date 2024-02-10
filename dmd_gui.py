import threading
from os.path import expanduser
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from dmd_control import dmd
import importlib
import time

FILENAME_SEEED = '' # this shouold be always the same for every acquisition

def track(folder, interval):
    print('traacking')
    time.sleep(interval)
    dz = 0
    return dz

class trackingThread(QThread):
    def __init__(self, zstage=0, function=None, interval=0):
        super(trackingThread, self).__init__()
        self.terminate = False
        self.function = function
        self.interval = interval

    def run(self):  # This needs to the function name! Do your thing in this
        # take first z-proj
        while not self.isInterruptionRequested():
            time.sleep(self.interval)
            print('t')
            time.sleep(2)
            # dz = self.function()
            # if dz > something:
            #     stage.move(-dz)


class Movement(QMainWindow):

    def __init__(self):
        super().__init__()
        # --------------------------------------------------
        # params
        self.amount = 100 # step size
        self.angle_step = 250 # step size

        self.xp_folder = '~/'
        self.tracking_interval = 5 #seconds

        self.thread = None

        # --------------------------------------------------
        self.setWindowTitle('DMD control')

        self.window = QWidget()
        self.layout = QGridLayout(self.window)
        self.setCentralWidget(self.window)
        

        self.open = QPushButton('OPEN')
        self.open.setStyleSheet('color: green;')
        # self.open.clicked.connect()
        self.layout.addWidget(self.open, 1, 1)

        self.close = QPushButton('CLOSE')
        self.close.setStyleSheet('color: red;')
        # self.close.clicked.connect()
        self.layout.addWidget(self.close, 1, 2)

        # current XP
        # current target
        # current sample

        # self.ccw = QPushButton('CCW')
        # self.ccw.setStyleSheet('color: red;')
        # self.ccw.clicked.connect(self.counterclockwise)
        # self.layout.addWidget(self.ccw, 1, 1)

        # self.up = QPushButton('UP')
        # self.up.clicked.connect(self.go_up)
        # self.layout.addWidget(self.up, 2, 2)
        # self.down = QPushButton('DOWN')
        # self.down.clicked.connect(self.go_down)
        # self.layout.addWidget(self.down, 4, 2)

        # self.left = QPushButton('EFT')
        # self.left.clicked.connect(self.go_left)
        # self.layout.addWidget(self.left, 3, 1)
        # self.right = QPushButton('RIGHT')
        # self.right.clicked.connect(self.go_right)
        # self.layout.addWidget(self.right, 3, 3)

        # self.forward = QPushButton('FW')
        # self.forward.clicked.connect(self.go_forward)
        # self.layout.addWidget(self.forward, 5, 1)
        # self.backward = QPushButton('BW')
        # self.backward.clicked.connect(self.go_backward)
        # self.layout.addWidget(self.backward, 5, 3)

        # self.home = QPushButton('HOME')
        # self.home.setStyleSheet('font: bold;')
        # self.home.clicked.connect(self.go_home)
        # self.layout.addWidget(self.home, 3, 2)

        # self.step_label = QLabel('Linear step')
        # self.layout.addWidget(self.step_label, 1, 5)
        # self.step = QLineEdit(str(self.amount))
        # self.layout.addWidget(self.step, 1, 6)
        # self.step.textChanged.connect(self.stepChange)

        # self.angle_label = QLabel('Angular step')
        # self.layout.addWidget(self.angle_label, 2, 5)
        # self.angle = QLineEdit(str(self.angle_step))
        # self.layout.addWidget(self.angle, 2, 6)
        # self.angle.textChanged.connect(self.angleChange)

        # #images forlder for tracking
        # self.track_label = QLabel('Track sample -->')
        # self.layout.addWidget(self.track_label , 4, 5)
        # self.track_folder = QPushButton('XP folder')
        # self.track_folder.clicked.connect(self.choose_directory)
        # self.layout.addWidget(self.track_folder, 4, 6)

        #  #checkbox
        # self.continuous_tracking = QPushButton('Track?')
        # self.layout.addWidget(self.continuous_tracking, 5, 5)
        # self.continuous_tracking.clicked.connect(self.track)
        # self.stop = QPushButton('stop')
        # self.layout.addWidget(self.stop, 6, 5)
        # self.stop.clicked.connect(self.stop_tracking)


        # # MOVEMENTS
    
    # def go_up(self):
    #     print('up')
    #     self.z.move(self.amount)
    # def go_down(self):
    #     print('down')
    #     self.z.move(-self.amount)

    # def go_left(self):
    #     print('left')
    #     self.plane.move(1, self.amount)
    # def go_right(self):
    #     print('right')
    #     self.plane.move(1, -self.amount)

    # def go_forward(self):
    #     print('left')
    #     self.plane.move(2, self.amount)
    # def go_backward(self):
    #     print('right')
    #     self.plane.move(2, -self.amount)

    # def go_home(self):
    #     print('home')
    # def clockwise(self):
    #     print('cw')
    #     self.plane.move(4, self.angle_step)
    # def counterclockwise(self):
    #     print('CCW')
    #     self.plane.move(4, -self.angle_step)

    # def stepChange(self):
    #     self.amount = int(self.step.text())
    
    # def angleChange(self):
    #     self.angle_step = int(self.angle.text())

    # def choose_directory(self):
    #     self.xp_folder = QFileDialog.getExistingDirectory(self,
    #     "Open a folder", expanduser("~"), QFileDialog.ShowDirsOnly)

    # @pyqtSlot()
    # def track(self):
    #     self.thread = trackingThread()
    #     self.thread.start()

    # @pyqtSlot()
    # def stop_tracking(self):
    #     self.thread.requestInterruption()
    #     time.sleep(.5)
    #     self.thread = None

    def closeEvent(self, event):
        question = QMessageBox.question(self, 'Going away?', '  6(-_-)9', 
                                    QMessageBox.Yes | QMessageBox.No)
        event.ignore()
        if question == QMessageBox.Yes:
            if self.thread != None:
                self.thread.requestInterruption()
                time.sleep(.5)
            event.accept()



app = QApplication([])
program = Movement()
program.show()
app.exec()
    