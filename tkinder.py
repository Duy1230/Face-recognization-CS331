import cv2
import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import ImageTk, Image
from model import *


FACE_MATCHING_THRESHOLD = 0.72


class CameraApp:
    def __init__(self, window, window_title):

        # set window
        self.window = window
        self.window.title(window_title)

        # set video source
        self.video_source = 0
        self.vid = None

        # Main window size
        self.width = 720
        self.height = 480
        self.window.geometry(f"{self.width}x{self.height}")

        # set video size
        self.vidWidth = 300
        self.vidHeight = 280
        self.vidX = 0
        self.vidY = 0

        # set grid
        self.window.columnconfigure(0, weight=3)
        self.window.columnconfigure(1, weight=4)
        self.window.rowconfigure(0, weight=1)

        # Set theme
        self.buttonStyle = ttk.Style()
        self.frameStyle = ttk.Style()
        self.labelFrameStyle = ttk.Style()
        self.labelStyle = ttk.Style()
        self.entryStyle = ttk.Style()

        ####### MODEL PARAMETERS ########
        self.face = None
        self.is_blink = False
        self.fake_blink = False
        self.eye_state = []
        self.prev_frame_face = None
        self.prev_face = None

        ####### VIDEO FRAME ########

        # Add video frame
        self.frameStyle.configure(
            'vidFrame.TFrame', borderwidth=1, relief='solid', bordercolor='#adb5bd')

        self.vidFrame = ttk.Frame(self.window, width=self.vidWidth/2,
                                  height=self.vidHeight, bootstyle="default", style="vidFrame.TFrame")
        self.vidFrame.grid(row=0, column=0, sticky='nsew')

        self.canvas = tk.Canvas(
            self.vidFrame, width=self.vidWidth, height=self.vidHeight)
        self.canvas.place(relx=0.04, rely=0.04)

        # Add button for video frame
        self.buttonStyle.configure('openVideo.TButton', font=(
            'calibri', 15, 'bold'), background="#198754", bordercolor="#198754")
        self.btn_upload_video = ttk.Button(
            self.vidFrame, text="Upload Video", command=self.upload_video, style="openVideo.TButton", bootstyle=INFO)
        self.btn_upload_video.place(
            relx=0.25, rely=0.75, relwidth=0.45, anchor=tk.CENTER)

        self.buttonStyle.configure('openImage.TButton', font=(
            'calibri', 15, 'bold'), background="#198754", bordercolor="#198754")
        self.btn_upload_image = ttk.Button(
            self.vidFrame, text="Upload Image", command=self.upload_image, style="openImage.TButton", bootstyle=DANGER)
        self.btn_upload_image.place(
            relx=0.25, rely=0.85, relwidth=0.45,  anchor=tk.CENTER)

        self.buttonStyle.configure('openCamera.TButton', font=(
            'calibri', 15, 'bold'), background="#198754", bordercolor="#198754")
        self.btn_open_camera = ttk.Button(
            self.vidFrame, text="Open Camera", command=self.open_camera, style="openCamera.TButton", bootstyle=INFO)
        self.btn_open_camera.place(
            relx=0.75, rely=0.75, relwidth=0.45, anchor=tk.CENTER)

        self.buttonStyle.configure('close.TButton', font=(
            'calibri', 15, 'bold'), background="#dc3545", bordercolor="#dc3545")
        self.btn_close = ttk.Button(
            self.vidFrame, text="Close", command=self.close_window, style="close.TButton", bootstyle=DANGER)
        self.btn_close.place(relx=0.75, rely=0.85,
                             relwidth=0.45,  anchor=tk.CENTER)

        ####### USER FRAME ########
        self.frameStyle.configure('userFrame.TFrame', background="#9FA6B2")

        # Add user frame
        self.userFrame = ttk.Frame(
            self.window, width=self.vidWidth/2, height=self.vidHeight, style="userFrame.TFrame")
        self.userFrame.grid(row=0, column=1, sticky='nsew')

        # Add face image
        self.faceImage = tk.Canvas(
            self.userFrame, width=self.vidWidth, height=self.vidHeight)
        self.faceImage.place(relx=0.12, rely=0.04)

        # Add label where prediction is shown
        self.labelStyle.configure('prediction.TLabel', font=(
            'roboto', 15, 'bold'), background="#9FA6B2", foreground="#198754")
        self.predictionLabel = ttk.Label(
            self.userFrame, text="", style="prediction.TLabel")
        self.predictionLabel.place(
            relx=0.05, rely=0.7, relwidth=0.8,  anchor=tk.W)

        # Add options frame
        self.labelFrameStyle.configure(
            'options.TLabelframe', background="#9FA6B2", borderwidth=1, relief='solid', bordercolor="#332D2D")
        self.labelFrameStyle.configure('options.TLabelframe.Label', font=(
            'Helvetica', 12, 'bold'), foreground='#332D2D', background="#9FA6B2")
        self.lbFrame1 = ttk.LabelFrame(
            self.userFrame, height=100, text="Options", bootstyle="secondary", style="options.TLabelframe")
        self.lbFrame1.place(relx=0.5, rely=0.85,
                            relwidth=0.9,  anchor=tk.CENTER)

        # Add label for options buttons
        self.labelStyle.configure('options.TLabel', font=(
            'helvetica', 12, 'normal'), background="#9FA6B2")
        self.nameLabel = ttk.Label(
            self.lbFrame1, text="Options:", style="options.TLabel")
        self.nameLabel.place(relx=0.15, rely=0.25,
                             relwidth=0.25,  anchor=tk.CENTER)

        # Add detected button for options frame
        self.buttonStyle.configure('detect.TButton', font=(
            'calibri', 12, 'bold'), background="#3B71CA", bordercolor="#3B71CA")
        self.btn_detect = ttk.Button(
            self.lbFrame1, text="Matching Face", command=self.detect_face, style="detect.TButton")
        self.btn_detect.place(relx=0.38, rely=0.25,
                              relwidth=0.35,  anchor=tk.CENTER)
        self.btn_detect.config(state=tk.DISABLED)

        # Add add data button for options frame
        self.buttonStyle.configure('addData.TButton', font=(
            'calibri', 12, 'bold'), background="#3B71CA", bordercolor="#3B71CA")
        self.add_to_database = ttk.Button(
            self.lbFrame1, text="Add to Database", command=self.add_face, style="addData.TButton")
        self.add_to_database.place(
            relx=0.77, rely=0.25, relwidth=0.40,  anchor=tk.CENTER)
        self.add_to_database.config(state=tk.DISABLED)

        # Add label and textbox for user information
        self.labelStyle.configure('name.TLabel', font=(
            'helvetica', 12, 'normal'), background="#9FA6B2")
        self.nameLabel = ttk.Label(
            self.lbFrame1, text="Name:", style="name.TLabel")
        self.nameLabel.place(relx=0.15, rely=0.75,
                             relwidth=0.25,  anchor=tk.CENTER)

        self.entryStyle.configure('name.TEntry', font=(
            'helvetica', 12, 'bold'), background="#9FA6B2", borderwidth=2, relief='solid', bordercolor='#3B71CA')
        self.nameEntry = ttk.Entry(self.lbFrame1, style="name.TEntry")
        self.nameEntry.place(relx=0.59, rely=0.75,
                             relwidth=0.77,  anchor=tk.CENTER)

        self.delay = 1

        self.addAssets()
        # self.update()

    def addAssets(self, components="all"):
        if components == "video_default_image" or components == "all":
            self.photo = Image.open("assets/open_camera.png")
            self.photo = ImageTk.PhotoImage(image=self.photo)
            self.canvas.create_image(
                self.vidX, self.vidY, image=self.photo, anchor=tk.NW)
        if components == "user_default_image" or components == "all":
            self.faceDetected = Image.open("assets/face_user.png")
            self.faceDetected = ImageTk.PhotoImage(image=self.faceDetected)
            self.faceImage.create_image(
                self.vidX, self.vidY, image=self.faceDetected, anchor=tk.NW)
        if components == "model_parameter_default" or components == "all":
            self.face = None
            # self.is_blink = False
            self.fake_blink = False
            self.eye_state = []
            self.prev_frame_face = None
        if components == "options_disable":
            self.btn_detect.config(state=tk.DISABLED)
            self.add_to_database.config(state=tk.DISABLED)
        if components == "options_enable":
            self.btn_detect.config(state=tk.NORMAL)
            self.add_to_database.config(state=tk.NORMAL)

    def upload_image(self):
        if self.vid is not None:
            self.open_camera()
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=(
            ("Image files", "*.jpg;*.jpeg;*.png;*.gif"), ("All files", "*.*")))
        if file_path:
            self.addAssets("options_enable")
            self.face = cv2.imread(file_path)
            temp_image = cv2.resize(self.face, (self.vidWidth, self.vidHeight))
            self.faceDetected = Image.fromarray(
                cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB))
            self.faceDetected = ImageTk.PhotoImage(image=self.faceDetected)
            self.faceImage.create_image(
                self.vidX, self.vidY, image=self.faceDetected, anchor=tk.NW)
            cv2.imwrite("assets/detected.png", self.face)

    def upload_video(self):
        if self.vid is None:
            self.video_source = filedialog.askopenfilename(title="Select Video File", filetypes=(
                ("Video files", "*.mp4;*.avi;*.mkv"), ("All files", "*.*")))
            if self.video_source:
                if not self.is_blink:
                    self.addAssets("options_disable")
                self.addAssets("model_parameter_default")
                self.vid = cv2.VideoCapture(self.video_source)
                self.btn_upload_video.configure(text="Close Video")
                self.buttonStyle.configure('openVideo.TButton', font=(
                    'calibri', 15, 'bold'), background="#ffc107", bordercolor="#ffc107")
                self.btn_open_camera.configure(state=tk.DISABLED)
                self.update()
        elif self.vid.isOpened():
            self.btn_upload_video.configure(text="Open Video")
            self.buttonStyle.configure('openVideo.TButton', font=(
                'calibri', 15, 'bold'), background="#198754", bordercolor="#198754")
            self.addAssets("video_default_image")
            self.vid.release()
            self.btn_open_camera.configure(state=tk.NORMAL)
            self.vid = None

    def open_camera(self):
        self.video_source = 0
        if self.vid is None:
            if not self.is_blink:
                self.addAssets("options_disable")
            self.addAssets("model_parameter_default")
            self.vid = cv2.VideoCapture(self.video_source)
            self.btn_open_camera.configure(text="Close Camera")
            self.buttonStyle.configure('openCamera.TButton', font=(
                'calibri', 15, 'bold'), background="#ffc107", bordercolor="#ffc107")
            self.btn_upload_video.configure(state=tk.DISABLED)
            self.update()
        elif self.vid.isOpened():
            # self.addAssets("options_disable")
            # self.addAssets("user_default_image")
            self.btn_open_camera.configure(text="Open Camera")
            self.buttonStyle.configure('openCamera.TButton', font=(
                'calibri', 15, 'bold'), background="#198754", bordercolor="#198754")
            self.addAssets("video_default_image")
            self.vid.release()
            self.btn_upload_video.configure(state=tk.NORMAL)
            self.vid = None

    def update(self):
        if self.vid is None:
            return
        ret, frame = self.vid.read()
        if self.is_blink:
            self.btn_detect.config(state=tk.NORMAL)
            self.add_to_database.config(state=tk.NORMAL)

        # for webcam input
        if ret and self.video_source == 0:
            frame, self.face, self.is_blink, self.fake_blink, self.eye_state, self.prev_frame_face, self.prev_face = detect_for_frame(
                frame,
                self.face,
                self.is_blink,
                self.fake_blink,
                self.eye_state,
                self.prev_frame_face,
                self.prev_face
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(
                cv2.resize(frame, (self.vidWidth, self.vidHeight))))
            self.canvas.create_image(
                self.vidX, self.vidY, image=self.photo, anchor=tk.NW)

        # for video input
        if ret and self.video_source != 0:
            num_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            if num_frame % 4 == 0:
                frame, self.face, self.is_blink, self.fake_blink, self.eye_state, self.prev_frame_face, self.prev_face = detect_for_frame(
                    frame,
                    self.face,
                    self.is_blink,
                    self.fake_blink,
                    self.eye_state,
                    self.prev_frame_face,
                    self.prev_face
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(
                    cv2.resize(frame, (self.vidWidth, self.vidHeight))))
                self.canvas.create_image(
                    self.vidX, self.vidY, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def detect_face(self):
        temp_face = cv2.cvtColor(self.face, cv2.COLOR_BGR2RGB)
        temp_face = cv2.resize(temp_face, (self.vidWidth, self.vidHeight))
        self.faceDetected = ImageTk.PhotoImage(
            image=Image.fromarray(temp_face))
        self.faceImage.create_image(
            self.vidX, self.vidY, image=self.faceDetected, anchor=tk.NW)
        # cv2.imwrite("assets/detected.png", self.face)
        name, distance = faceMatching(self.face)
        if distance > FACE_MATCHING_THRESHOLD:
            self.predictionLabel.configure(
                text="Hello " + name + " " + str(distance)[:5])
            self.labelStyle.configure(
                'prediction.TLabel', foreground="#198754")
        else:
            self.predictionLabel.configure(
                text="Unknown " + name + " " + str(distance)[:5])
            self.labelStyle.configure(
                'prediction.TLabel', foreground="#dc3545")

    def add_face(self):
        temp_face = cv2.cvtColor(self.face, cv2.COLOR_BGR2RGB)
        temp_face = cv2.resize(temp_face, (self.vidWidth, self.vidHeight))
        self.faceDetected = ImageTk.PhotoImage(
            image=Image.fromarray(temp_face))
        self.faceImage.create_image(
            self.vidX, self.vidY, image=self.faceDetected, anchor=tk.NW)
        if self.nameEntry.get() == "":
            self.entryStyle.configure('nameEntry.TEntry', font=(
                'calibri', 15, 'bold'), background="#9FA6B2", borderwidth=2, relief='solid', bordercolor='#dc3545')
            self.predictionLabel.configure(text="Please enter name")
        else:
            # cv2.imwrite("assets/detected.png", self.face)
            addFaceToDatabase(self.face, self.nameEntry.get())
            self.entryStyle.configure('nameEntry.TEntry', font=(
                'calibri', 15, 'bold'), background="#9FA6B2", borderwidth=2, relief='solid', bordercolor='#3B71CA')
            self.predictionLabel.configure(
                text="Added " + self.nameEntry.get())

    def close_window(self):
        if self.vid is None:
            self.window.destroy()
        elif self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

    def run(self):
        self.window.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root, "Camera App")
    # app.load_background_image('woman.png')
    app.run()
