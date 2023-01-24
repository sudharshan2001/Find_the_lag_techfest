import sys,PIL, os, shutil
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import  QApplication, QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QIcon
import cv2
from PIL import Image
from PyQt5.QtWidgets import QApplication
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchmetrics
import torch

global resize_factor
resize_factor=(1728 , 972)
# [(x_start, y_start), (x_end, y_end)]
class WelcomeScreen(QMainWindow):
    def __init__(self):
        super(WelcomeScreen, self).__init__()
        loadUi(os.path.join("UI Files","intro.ui"),self)
        
        self.upload_button.clicked.connect(self.goto_video_selection)

    def goto_video_selection(self):
        two_dim_plot_window = Create_video_selection()
        widget.addWidget(two_dim_plot_window)
        widget.setCurrentIndex(widget.currentIndex()+1)

class Create_video_selection(QMainWindow):
    def __init__(self):
        super(Create_video_selection, self).__init__()
        loadUi(os.path.join("UI Files","path_select_screen.ui"), self)
        self.twodplotcsvtext = ''
        self.twodplotcsv_2text= ''
        self.path = ''
        
        self.folderselect.clicked.connect(self.select_the_video)
        self.dimselector.clicked.connect(self.go_to_error)
        self.cancel_trackingtwo.clicked.connect(self.go_to_backhome)
        self.folderselect_csv.clicked.connect(self.go_to_folder_choose)
    
    def go_to_folder_choose(self):
        filename_csv = QFileDialog.getExistingDirectory()
        self.folder_csv_file_text = filename_csv
        folder_csv_file_text = filename_csv
        # print(filename_csv)
        if len(folder_csv_file_text) > 1:
            self.pathlabel_csv.setText(folder_csv_file_text)


    def go_to_backhome(self):
        back_to_home1 = WelcomeScreen()
        widget.addWidget(back_to_home1)
        widget.setCurrentIndex(widget.currentIndex() + 1)
    
    # Select CSV File
    def select_the_video(self):
        filename = QFileDialog.getOpenFileName(filter="Video Files (*.avi *.mp4)")
        self.video_file_text = filename[0]
        video_file_text = filename[0]
        self.path = filename[0]
        if len(video_file_text) > 1:
            self.pathlabel.setText(video_file_text)

    def go_to_error(self):
            if self.path == '':
                self.error.setText("Please Select the Path to Video")
            
            elif self.path[-3:] not in ['avi', 'mp4', 'MP4', 'AVI']:
                self.error.setText('Select Appropriate File Format')
                
            else:
                self.error.setText('')
                cam = cv2.VideoCapture(self.path)
                while(True):
                    ret,frame = cam.read()
    
                    if ret:
                        # if video is still left continue creating images
                        path_to_img_file = './Temporary Files/' + str(self.path.split('/')[-1][:-4]) + '.jpg'
                
                        cv2.imwrite(path_to_img_file, frame)
                        break
                    break

                popup_window =  createpopupforimage(path_to_img_file,self.path,self.folder_csv_file_text)
                widget.addWidget(popup_window)
                widget.setCurrentIndex(widget.currentIndex() + 1)

# To crop the image
class createpopupforimage(QMainWindow):
    def __init__(self,path_to_img_file,path,folder_csv_file_text):
        super(createpopupforimage, self).__init__()
        loadUi(os.path.join("UI Files","screen image.ui"), self)
        cropping = False
        self.path_to_img_file = path_to_img_file
        self.vidpath=path
        self.folder_csv_file_text = folder_csv_file_text

        img = Image.open(path_to_img_file)
        img=img.resize(resize_factor, PIL.Image.ANTIALIAS)
        file_cropped_save_name = path_to_img_file[:-4]+'resized_image.jpg'
        img.save(file_cropped_save_name)

        self.image = cv2.imread(file_cropped_save_name)
        oriImage =self.image.copy()
        self.oriImage2 =self.image.copy()

        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                # print(refPoint)
                if len(refPoint) == 2:  
                    self.roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0],:]
                    cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB, self.roi)
                    self.roi=cv2.resize(self.roi, resize_factor, interpolation = cv2.INTER_AREA)
                    
                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.image_project_label_2.setPixmap(QPixmap.fromImage(self.roi))
                    self.image_project_label_2.setScaledContents(True)
                    self.retrybuttondim_2.clicked.connect(self.crop_again)
                    self.dimcancel_2.clicked.connect(self.go_back_to_2D)
                    self.dimok_2.clicked.connect(self.goto2DPredicting)
                    cv2.destroyAllWindows()
                    
        cv2.namedWindow("screen image")
        cv2.setMouseCallback("screen image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("screen image", self.image)
                

            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("screen image", self.i)
            break

    def go_back_to_2D(self):
        two_dim_window_ag = Create_video_selection()
        widget.addWidget(two_dim_window_ag)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def goto2DPredicting(self):
        twodimloader = createpopupforimage2(self.path_to_img_file, self.vidpath, [(x_start, y_start), (x_end, y_end)],self.folder_csv_file_text)
        widget.addWidget(twodimloader)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def crop_again(self):
        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                # print(refPoint)
                if len(refPoint) == 2:  
                    self.roi = self.oriImage2[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0],:]
                    
                    cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB, self.roi)
                    self.roi=cv2.resize(self.roi, resize_factor, interpolation = cv2.INTER_AREA)
                    
                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.image_project_label_2.setPixmap(QPixmap.fromImage(self.roi))
                    self.image_project_label_2.setScaledContents(True)
                    
                    cv2.destroyAllWindows()
                    self.retrybuttondim_2.clicked.connect(self.crop_again)
                    self.dimcancel_2.clicked.connect(self.go_back_to_2D)
                    self.dimok_2.clicked.connect(self.goto2DPredicting)
                    
        cv2.namedWindow("screen image")
        cv2.setMouseCallback("screen image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("screen image", self.image)
                
            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("screen image", self.i)
            break


class createpopupforimage2(QMainWindow):
    def __init__(self,path_to_img_file,path, arr_for_box,folder_csv_file_text):
        super(createpopupforimage2, self).__init__()
        loadUi(os.path.join("UI Files","ledbox_select.ui"), self)
        cropping = False
        print(arr_for_box)
        self.arr_for_box = arr_for_box
        self.path_to_img_file = path_to_img_file
        self.vidpath=path
        self.folder_csv_file_text = folder_csv_file_text
        # print(self.folder_csv_file_text)

        img = Image.open(path_to_img_file)
        img=img.resize(resize_factor, PIL.Image.ANTIALIAS)
        file_cropped_save_name = path_to_img_file[:-4]+'resized_image.jpg'
        img.save(file_cropped_save_name)

        self.image = cv2.imread(file_cropped_save_name)
        oriImage =self.image.copy()
        self.oriImage2 =self.image.copy()

        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
   
                if len(refPoint) == 2:  
                    self.roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0],:]
                    cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB, self.roi)
                    self.roi=cv2.resize(self.roi,resize_factor, interpolation = cv2.INTER_AREA)
                    
                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.image_project_label.setPixmap(QPixmap.fromImage(self.roi))
                    self.image_project_label.setScaledContents(True)
                    self.retrybuttondim.clicked.connect(self.crop_again)
                    self.dimcancel.clicked.connect(self.go_back_to_2D)
                    self.dimok.clicked.connect(self.goto2DPredicting)
                    cv2.destroyAllWindows()
                    
        cv2.namedWindow("LED image")
        cv2.setMouseCallback("LED image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("LED image", self.image)
                

            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("LED image", self.i)
            break

    def go_back_to_2D(self):
        two_dim_window_ag = Create_video_selection()
        widget.addWidget(two_dim_window_ag)
        widget.setCurrentIndex(widget.currentIndex() + 1)
        
    def goto2DPredicting(self):
        twodimloader = Createfinalgui(self.path_to_img_file, self.vidpath, [(x_start, y_start), (x_end, y_end)],self.folder_csv_file_text,self.arr_for_box)
        widget.addWidget( twodimloader)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def crop_again(self):
        x_start, y_start, x_end, y_end = 0, 0, 0, 0
        def mouse_crop(event, x, y, flags, param):
            global x_start, y_start, x_end, y_end, cropping

            if event == cv2.EVENT_LBUTTONDOWN:
                x_start, y_start, x_end, y_end = x, y, x, y
                cropping = True

            elif event == cv2.EVENT_MOUSEMOVE:
                try:
                    if cropping == True:
                        x_end, y_end = x, y
                except:
                    pass

            elif event == cv2.EVENT_LBUTTONUP:
                
                x_end, y_end = x, y
                cropping = False  

                refPoint = [(x_start, y_start), (x_end, y_end)]
                print(refPoint)
                if len(refPoint) == 2:  
                    self.roi = self.oriImage2[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0],:]
                    
                    cv2.cvtColor(self.roi, cv2.COLOR_BGR2RGB, self.roi)
                    self.roi=cv2.resize(self.roi, resize_factor, interpolation = cv2.INTER_AREA)
                    
                    self.roi = QImage(self.roi.data.tobytes(), self.roi.shape[1],self.roi.shape[0],QImage.Format_RGB888)
                    self.image_project_label.setPixmap(QPixmap.fromImage(self.roi))
                    self.image_project_label.setScaledContents(True)
                    
                    cv2.destroyAllWindows()
                    self.retrybuttondim.clicked.connect(self.crop_again)
                    self.dimcancel.clicked.connect(self.go_back_to_2D)
                    self.dimok.clicked.connect(self.goto2DPredicting)
                    
        cv2.namedWindow("LED image")
        cv2.setMouseCallback("LED image", mouse_crop)
        
        while True:
            self.i = self.image.copy()

            if not cropping:
                cv2.imshow("LED image", self.image)
                
            elif cropping:
                cv2.rectangle(self.i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                cv2.imshow("LED image", self.i)
            break


class Createfinalgui(QMainWindow):
    def __init__(self,path_to_img_file,vidpath, arr_for_box,folder_csv_file_text,arr_for_box2):
            super(Createfinalgui, self).__init__()
            loadUi(os.path.join("UI Files","final_ui.ui"), self)
            self.path_to_img_file = path_to_img_file
            self.vidpath = vidpath
            self.arr_for_box = arr_for_box
            self.folder_csv_file_text = folder_csv_file_text
            self.arr_for_screen = arr_for_box2

            [(x_start_for_box, y_start_for_box), (x_end_for_box, y_end_for_box)] = self.arr_for_box
            [(x_start_for_image, y_start_for_image), (x_end_for_image, y_end_for_image)] = self.arr_for_screen

            filename = vidpath

            video_name = filename.split('/')[-1].split('.')[0]
            list_ = []

            

            if os.path.exists(video_name):
                shutil.rmtree(video_name)
                os.makedirs(video_name)
            else:
                os.makedirs(video_name)

            cap = cv2.VideoCapture(filename)
            fps_of_video    = cap.get(cv2.CAP_PROP_FPS)
            print(fps_of_video)
            current_frame = 0
            while True:
                QApplication.processEvents()
                ret, frame = cap.read()
                if ret == True:
                    name =video_name+'/'+video_name.replace(' ','')+'_'+str(current_frame)+'.jpg'
                    list_.append(name)

                    cv2.imwrite(name, frame)
                    current_frame += 1
                    # print(ret, current_frame)
                QApplication.processEvents()
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break
                
            cap.release()
        
            df=pd.DataFrame()
            df["path"]=list_
            print(df.head)
            lower_range = np.array([110,50,50])
            upper_range = np.array([130,255,255])
                
            blick_or_not_blink = []
            for i in tqdm(list_):
                QApplication.processEvents()
                img=cv2.imread(i)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img =  cv2.resize(img, resize_factor, interpolation = cv2.INTER_AREA)
                input_blink = img[y_start_for_box:y_end_for_box,x_start_for_box:x_end_for_box]
                QApplication.processEvents()
                hsv = cv2.cvtColor(input_blink, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, lower_range, upper_range)
                QApplication.processEvents()
                hasGreen = np.sum(mask)
                if hasGreen>10000:
                    blick_or_not_blink.append(1)
                else:
                    blick_or_not_blink.append(0)

            plt.imshow(input_blink)
            plt.show()

            blick_or_not_blink_3 = blick_or_not_blink.copy()
            print(blick_or_not_blink_3)

            for i in range(1,len(blick_or_not_blink)-20):
                QApplication.processEvents()
                previous_number = blick_or_not_blink[i-1]
                current_number = blick_or_not_blink[i]
                next_number = blick_or_not_blink[i+1]
                
                QApplication.processEvents()
                if (previous_number==1) and (next_number==0) and (current_number==1):
                    QApplication.processEvents()
                    blick_or_not_blink_3[i+1:i+20] = [1 for _ in range(len(blick_or_not_blink_3[i+1:i+20]))]
                    print(blick_or_not_blink_3[i], i, blick_or_not_blink_3[i+1:i+15],blick_or_not_blink[i+1:i+20])
                    
            df["blink_status"]=blick_or_not_blink_3
            present_df = df[df['blink_status']==1]

            print(present_df.head())

            init_and_end=[]
            for i in range(1,len(present_df)):
                QApplication.processEvents()
                if (int(present_df.iloc[i]['path'].split('/')[-1].split('_')[-1].split('.')[0])- int(present_df.iloc[i-1]['path'].split('/')[-1].split('_')[-1].split('.')[0])) > 1:
                    init_and_end.append(i-1)

            init_and_end.append(len(present_df))
            print('init_and_end', init_and_end)
            
            QApplication.processEvents()
            for i in tqdm(range(len(init_and_end))):
                background = None
                if i!=0:
                    full_d = present_df.iloc[init_and_end[i-1]+1:init_and_end[i]]
                else:
                    full_d =present_df.iloc[:init_and_end[i]]
                # print(full_d)
                fps_values = []
                final_outputs = []
                for path in full_d['path'].values[1:]:
            #         print(path)
                    QApplication.processEvents()
                    img=cv2.imread(path)
                    img =  cv2.resize(img, resize_factor, interpolation = cv2.INTER_AREA)
                    frame = img[y_start_for_image:y_end_for_image,x_start_for_image:x_end_for_image]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray = cv2.GaussianBlur(gray, (5,5),0)
                    
                    if background is None:
                        QApplication.processEvents()
                        background=gray
                        background = np.expand_dims(background,0)
                        background = torch.Tensor(background)
                        background = torch.permute(background,(0,3,1,2))
                        
                        gray_init = np.expand_dims(gray,0)
                        QApplication.processEvents()
                        gray_init = torch.Tensor(gray_init)
                        gray_init = torch.permute(gray_init,(0,3,1,2))

                        QApplication.processEvents()
                        ssim_init = torchmetrics.StructuralSimilarityIndexMeasure(kernel_size=5,return_full_image=True,)
                        final_init = ssim_init(gray_init,background)

                        continue
                        

                    gray = np.expand_dims(gray,0)
                    QApplication.processEvents()
                    gray = torch.Tensor(gray)
                    gray = torch.permute(gray,(0,3,1,2))
                    QApplication.processEvents()
                    ssim = torchmetrics.StructuralSimilarityIndexMeasure(kernel_size=5,return_full_image=True)
                    final = ssim(gray,background)
                    
                    final_outputs.append((final,path.split('.')[-2].split('_')[-1]))
            #     break
                print(full_d.head(1))
                print(full_d.head(1)['path'].values)
                initial_fps = int(full_d.head(1)['path'].values[0].split('/')[-1].split('_')[-1].split('.')[0])
                print('init', initial_fps)
                
                log_arr=[]
                number = []
                for i in final_outputs:
                    QApplication.processEvents()
                    img = cv2.absdiff(np.asarray(final_init[1][0].permute(1,2,0).cpu().detach().numpy()), np.asarray(i[0][1][0].permute(1,2,0).detach().numpy()))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.asarray(img,dtype=np.uint8)

                    _, contours, hierarchy = cv2.findContours(image=img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                    image_copy = img.copy()

                    QApplication.processEvents()
                    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                    log_arr.append(np.sum(image_copy))
                    number.append(int(i[1]))

                log_arr2 = (log_arr-np.mean(log_arr))/np.std(log_arr)
            #     print()
                log_array = np.log(log_arr2)
                for i in range(len(number)):
                    QApplication.processEvents()
            #         print(number[i], log_array[i])
                    if not np.isnan(log_array[i]):
                        changed_fps = number[i]
                        print('changed', changed_fps)
                        diff = changed_fps+1-initial_fps-1
                        print('diff', (diff/fps_of_video)*1000)
                        break

app = QApplication(sys.argv)
welcome = WelcomeScreen()

widget = QtWidgets.QStackedWidget()
widget.addWidget(welcome)
widget.setFixedHeight(800)
widget.setFixedWidth(1200)
widget.setWindowTitle('Find the Lag')
widget.setWindowIcon(QIcon(os.path.join('icons','icon.png')))
widget.show()

try:
    sys.exit(app.exec_())
except:
    print("Exiting")