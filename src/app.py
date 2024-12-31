'''
    This script provides a GUI for an image processing application using the Tkinter framework. 
    The application allows users to load images, apply various processing techniques 
    (e.g., Histogram Equalization, CLAHE, Sharpening-Smoothing Image Filter, brightness/contrast adjustment, etc), 
    and save the processed images. 
    Author: Kevin Antony Gomez
'''

import tkinter as tk
from PIL import Image, ImageTk
from PIL.PngImagePlugin import PngImageFile
from tkinter import DoubleVar, filedialog
import cv2
import img_processors as imp

class App:
    def __init__(self) -> None:
        self.root = tk.Tk() # init main window
        self.SCALE_FACTOR = 0.85 # window and top row frame scaling factor
        self.window_sizer(self.SCALE_FACTOR) # set window geometry
        self.select_img_window()
        self.root.mainloop()
    

    def select_img_window(self) -> None:
        '''
        Presents a window with a button to select the input image
        Args: None
        Returns: None
        '''
        self.clear_window()
        self.root.title('Select an image')
        browse_btn = tk.Button(self.root, text="Browse Images", command=self.handle_browse_imgs, \
            padx=int(self.window_width*0.02), pady=int(self.window_height*0.02), font=("Arial", 14))
        browse_btn.place(relx=0.5, rely=0.5, anchor="center")


    def browse_imgs(self) -> str:
        '''
        Presents a file browser to select image files (.jpg, .jpeg, .png)
        Args: None
        Returns: 
            file_path (str): path to the selected image
        '''
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            return file_path
    
    def handle_browse_imgs(self) -> None:
        '''
        Handles the browse_btn click event and transitions to the main image processing window
        Args: None
        Returns: None
        '''
        self.file_path = self.browse_imgs()
        self.img_processor_window()


    def window_sizer(self, scale_factor:float=0.8) -> None:
        '''
        Sizes the GUI window using the screen's dimensions and a scaling factor
        Args:
            scale_factor (float): A float value applied to the screen dimensions to set 
                        the window size. Must be in (0,1]
        Returns: None
        '''
        assert (scale_factor > 0 and scale_factor <=1), 'scale_factor must be in (0, 1]'
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.window_width = int(self.screen_width * scale_factor)
        self.window_height = int(self.screen_height * scale_factor)
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.minsize(self.window_width, self.window_height)
        self.root.maxsize(self.window_width, self.window_height)


    def save_img(self) -> None:            
        '''
        Presents a file browser to select the destination, set the name of the image, and save it 
        Args: None
        Returns: None
        '''
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")],
            title="Save Image"
        )
        if save_path:
            if self.curr_process != None and self.curr_process != 'save':
                self.curr_img_bgr = self.tmp_img_bgr
            img = self.curr_img_bgr
            cv2.imwrite(save_path, img)
            print(f"Image saved as {save_path}")


    def clear_window(self) -> None:
        '''
        Clears all widgets/elements in the current window
        Args: None
        Returns: None
        '''
        for widget in self.root.winfo_children():
            widget.destroy()


    def resize_image(self, image:PngImageFile, frame_width:int, frame_height:int) -> PngImageFile:
        '''
        Resizes the selected image to fit in a frame within the image processing window while
        respecting aspect ratio. Resamples using LANCZOS
        Args: 
            image (PngImageFile): Image to resize
            frame_width (int): Width of the frame in which the image needs to be placed
            frame_height (int): Height of the frame in which the image needs to be placed
        Returns:
            PngImageFile: Resized image
        '''
        image_width, image_height = image.size
        aspect_ratio = image_width / image_height
        if frame_width / frame_height > aspect_ratio:
            new_height = frame_height
            new_width = int(frame_height * aspect_ratio)
        else:
            new_width = frame_width
            new_height = int(frame_width / aspect_ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


    def reset_sliders(self) -> None:
        '''
        Resets all sliders to their default values
        Args: None
        Returns: None
        '''
        self.brightness_val.set(1.0)
        self.contrast_val.set(1.0)
        self.clahe_val.set(0.0)
        self.sat_val.set(1.0)
        self.kappa_val.set(0.0)
        self.lighten_val.set(1.0)
        self.scale_val.set(100.0)
        self.red_val.set(1.0)
        self.green_val.set(1.0)
        self.blue_val.set(1.0)

    
    def reset_img(self, label:tk.Label) -> None:
        '''
        Handles the reset button click event.
        Args: 
            label (tk.Label): label containing the image that needs to be reset
        Returns: None
        '''
        label.config(image=self.original_img_tk)
        label.pack(expand=True)
        self.reset_sliders()
        self.curr_img_bgr = cv2.imread(self.file_path)
        self.tmp_img_bgr = None
        self.curr_process = None


    def indicate_processing(self) -> tk.Label:
        '''
        Presents a loading indicator when an image processing event is triggered.
        Must be destroyed manually as required.
        Args: None
        Returns:
            tk.Label: Loading indicator label
        '''
        loading_label = tk.Label(self.root, text="Processing...", font=("Arial", 16))
        loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self.root.update_idletasks() 
        return loading_label


    def process_img(self, cmd:str, label:tk.Label, value=None) -> None:
        '''
        Processes the current image based on the specified command and updates the GUI label.
        Args:
            cmd (str): The processing command to apply to the image. Supported commands include:
                    - 'upscale_lancoz': Upscales the image using the Lanczos algorithm with a scaling factor
                    - 'hist_eq': Histogram equalization
                    - 'denoise_medianblur': Applies median blur denoising
                    - 'clahe_eq': Contrast Limited Adaptive Histogram Equalization with a specified clip limit
                    - 'brighten': Increases brightness by a specified value
                    - 'lighten': Lightens the image by a specified value
                    - 'contrast': Adjusts image contrast by a specified value
                    - 'sat': Adjusts image saturation by a specified value
                    - 'ssif': Applies SSIF processing with a specified kappa value
                    - 'r_channel': Multiplies the red channel by a specified factor
                    - 'g_channel': Multiplies the green channel by a specified factor
                    - 'b_channel': Multiplies the blue channel by a specified factor
            label (tk.Label): The GUI label to display the processed image.
            value (optional): A parameter associated with some commands (e.g., scaling factor, brightness value).
                            Expected to be a Tkinter variable with a `get()` method.

        Returns: None
        '''
        loading_label = self.indicate_processing()
        if cmd == 'upscale_lancoz':
            if self.curr_process != None and self.curr_process != 'upscale_lancoz':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'upscale_lancoz'
            value = value/100
            img = imp.upscale_lancoz(self.curr_img_bgr, value=value)

        if cmd == 'hist_eq':
            if self.curr_process != None:
                self.curr_img_bgr = self.tmp_img_bgr
            self.reset_sliders()
            self.curr_process = 'hist_eq'
            img = imp.hist_eq_ycrcb(self.curr_img_bgr)
        
        if cmd == 'denoise_medianblur':
            if self.curr_process != None:
                self.curr_img_bgr = self.tmp_img_bgr
            self.reset_sliders()
            self.curr_process = 'denoise_medianblur'
            img = imp.denoise_medianBlur(self.curr_img_bgr)
        
        if cmd == 'clahe_eq':
            if self.curr_process != None and self.curr_process != 'clahe_eq':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'clahe_eq'
            value = int(value.get())
            img = imp.clahe_ycrcb(self.curr_img_bgr, value)

        if cmd == 'brighten':
            if self.curr_process != None and self.curr_process != 'brighten':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'brighten'
            value = int(value.get())
            img = imp.brighten(self.curr_img_bgr, value)

        if cmd == 'lighten':
            if self.curr_process != None and self.curr_process != 'lighten':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'lighten'
            value = int(value.get())
            img = imp.lighten(self.curr_img_bgr, value)
        
        if cmd == 'contrast':
            if self.curr_process != None and self.curr_process != 'contrast':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'contrast'
            value = value.get()
            img = imp.contrast(self.curr_img_bgr, value)
        
        if cmd == 'sat':
            if self.curr_process != None and self.curr_process != 'sat':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'sat'
            value = value.get()
            img = imp.saturate(self.curr_img_bgr, value)

        if cmd == 'ssif':
            if self.curr_process != None and self.curr_process != 'ssif':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'ssif'
            value = value.get()
            img = imp.SSIF(self.curr_img_bgr, kappa=value)

        if cmd == 'r_channel':
            if self.curr_process != None and self.curr_process != 'r_channel':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'r_channel'
            value = value.get()
            (b, g, r) = cv2.split(self.curr_img_bgr)
            r = cv2.multiply(r, value)
            img = cv2.merge([b, g, r])
        
        if cmd == 'g_channel':
            if self.curr_process != None and self.curr_process != 'g_channel':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'g_channel'
            value = value.get()
            (b, g, r) = cv2.split(self.curr_img_bgr)
            g = cv2.multiply(g, value)
            img = cv2.merge([b, g, r])
        
        if cmd == 'b_channel':
            if self.curr_process != None and self.curr_process != 'b_channel':
                self.curr_img_bgr = self.tmp_img_bgr
                self.reset_sliders()
            self.curr_process = 'b_channel'
            value = value.get()
            (b, g, r) = cv2.split(self.curr_img_bgr)
            b = cv2.multiply(b, value)
            img = cv2.merge([b, g, r])

        self.tmp_img_bgr = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.resize_image(img, self.top_row_frame_width, self.top_row_height)
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img
        label.pack(expand=True)
        loading_label.destroy()
    
    def img_processor_window(self) -> None:
        '''
        Initializes and displays the image processing window within the GUI.
        This method clears any existing content in the root window, sets up a new layout,
        and provides functionality for loading, displaying, and manipulating images. It
        also initializes sliders, buttons, and labels for various image processing commands.
        Args: None
        Returns: None
        '''
        self.clear_window()
        self.root.title('Image Processor')
        top_row_gap_size = int(self.window_width * 0.01)
        self.top_row_height = int(self.window_height * self.SCALE_FACTOR)  # Top row frame height
        bottom_row_height = self.window_height - self.top_row_height
        self.top_row_frame_width = (self.window_width - top_row_gap_size) // 2

        # Create frames
        top_left_frame = tk.Frame(self.root, width=self.top_row_frame_width, height=self.top_row_height)
        top_right_frame = tk.Frame(self.root, width=self.top_row_frame_width, height=self.top_row_height)
        bottom_frame = tk.Frame(self.root, width=self.window_width, height=bottom_row_height, bg="black")

        # Place the frames using grid
        top_left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, top_row_gap_size // 2))
        top_right_frame.grid(row=0, column=2, sticky="nsew", padx=(top_row_gap_size // 2, 0))
        bottom_frame.grid(row=1, column=0, columnspan=3, sticky="nsew")

        # Load and resize images
        original_img = Image.open(self.file_path)
        self.curr_img_bgr = cv2.imread(self.file_path)
        self.tmp_img_bgr = None
        self.curr_process = None
        self.original_img_resized = self.resize_image(original_img, self.top_row_frame_width, self.top_row_height)
        self.editing_img_resized = self.original_img_resized.copy()

        # Convert to PhotoImage and store as instance variables
        self.original_img_tk = ImageTk.PhotoImage(self.original_img_resized)
        self.editing_img_tk = ImageTk.PhotoImage(self.editing_img_resized)

        # Labels for images within each frame
        top_left_label = tk.Label(top_left_frame, image=self.original_img_tk)
        top_left_label.pack(expand=True)
        top_right_label = tk.Label(top_right_frame, image=self.editing_img_tk)
        top_right_label.pack(expand=True)

        # Add buttons and sliders to the bottom row
        select_btn = tk.Button(bottom_frame, text="Select", command=lambda:self.select_img_window())
        save_btn = tk.Button(bottom_frame, text="Save", command=lambda: self.save_img())
        reset_btn = tk.Button(bottom_frame, text="Reset", command=lambda:self.reset_img(top_right_label))
        denoise_btn = tk.Button(bottom_frame, text="Denoise", command=lambda:self.process_img('denoise_medianblur', top_right_label))
        hist_btn = tk.Button(bottom_frame, text="Hist Eq.", command=lambda:self.process_img('hist_eq', top_right_label))
        self.scale_val = DoubleVar() 
        upscale_slider = tk.Scale(bottom_frame, variable=self.scale_val, from_=100, to=1000, resolution=50, orient="horizontal")
        upscale_btn = tk.Button(bottom_frame, text="Upscale", command=lambda: self.process_img('upscale_lancoz', top_right_label, self.scale_val.get()))
        self.brightness_val = DoubleVar()
        brightness_slider = tk.Scale(bottom_frame, variable=self.brightness_val, from_=1, to=200, orient="horizontal", command=lambda event:self.process_img('brighten', top_right_label, self.brightness_val))
        self.lighten_val = DoubleVar()
        lighten_slider = tk.Scale(bottom_frame, variable=self.lighten_val, from_=1, to=100, orient="horizontal", command=lambda event:self.process_img('lighten', top_right_label, self.lighten_val))
        self.contrast_val = DoubleVar()
        contrast_slider = tk.Scale(bottom_frame, variable=self.contrast_val, from_=1, to=10, resolution=0.1, orient="horizontal", command=lambda event:self.process_img('contrast', top_right_label, self.contrast_val))
        self.clahe_val = DoubleVar()
        clahe_slider = tk.Scale(bottom_frame, variable=self.clahe_val, from_=0, to=10, orient="horizontal", command=lambda event:self.process_img('clahe_eq', top_right_label, self.clahe_val))
        self.sat_val = DoubleVar()
        sat_slider = tk.Scale(bottom_frame, variable=self.sat_val, from_=1, to=10, resolution=0.1,  orient="horizontal", command=lambda event:self.process_img('sat', top_right_label, self.sat_val))
        self.kappa_val = DoubleVar()
        ssif_slider = tk.Scale(bottom_frame, variable=self.kappa_val, from_=0, to=10, resolution=0.1,  orient="horizontal", command=lambda event:self.process_img('ssif', top_right_label, self.kappa_val))
        self.red_val = DoubleVar()
        red_slider = tk.Scale(bottom_frame, variable=self.red_val, from_=1, to=5, resolution=0.1, orient="horizontal", command=lambda event:self.process_img('r_channel', top_right_label, self.red_val))
        self.green_val = DoubleVar()
        green_slider = tk.Scale(bottom_frame, variable=self.green_val, from_=1, to=5, resolution=0.1, orient="horizontal", command=lambda event:self.process_img('g_channel', top_right_label, self.green_val))
        self.blue_val = DoubleVar()
        blue_slider = tk.Scale(bottom_frame, variable=self.blue_val, from_=1, to=5, resolution=0.1, orient="horizontal", command=lambda event:self.process_img('b_channel', top_right_label, self.blue_val))

        # Add buttons to the bottom frame
        select_btn.grid(row=0, column=0, sticky="nsew", padx=1, pady=3)
        save_btn.grid(row=0, column=1, sticky="nsew", padx=1, pady=3)
        reset_btn.grid(row=0, column=2, sticky="nsew", padx=1, pady=3)
        denoise_btn.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        hist_btn.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)
        upscale_btn.grid(row=0, column=10, sticky="nsew", padx=3, pady=3)

        # Slider labels
        clahe_label = tk.Label(bottom_frame, text="CLAHE")
        lighten_label = tk.Label(bottom_frame, text="Lighten")
        brightness_label = tk.Label(bottom_frame, text="Brighten")
        contrast_label = tk.Label(bottom_frame, text="Contrast")
        saturation_label = tk.Label(bottom_frame, text="Saturation")
        ssif_label = tk.Label(bottom_frame, text="SSIF")
        red_label = tk.Label(bottom_frame, text="Red Channel")
        green_label = tk.Label(bottom_frame, text="Green Channel")
        blue_label = tk.Label(bottom_frame, text="Blue Channel")
        clahe_label.grid(row=0, column=4, sticky="nsew", padx=3, pady=3)
        brightness_label.grid(row=0, column=5, sticky="nsew", padx=3, pady=3)
        lighten_label.grid(row=0, column=6, sticky="nsew", padx=3, pady=3)
        contrast_label.grid(row=0, column=7, sticky="nsew", padx=3, pady=3)
        saturation_label.grid(row=0, column=8, sticky="nsew", padx=3, pady=3)
        ssif_label.grid(row=0, column=9, sticky="nsew", padx=3, pady=3)
        red_label.grid(row=0, column=11, sticky="nsew", padx=3, pady=3)
        green_label.grid(row=0, column=12, sticky="nsew", padx=3, pady=3)
        blue_label.grid(row=0, column=13, sticky="nsew", padx=3, pady=3)

        # Add sliders to the bottom frame
        clahe_slider.grid(row=1, column=4, sticky="nsew", padx=3, pady=3)
        brightness_slider.grid(row=1, column=5, sticky="nsew", padx=3, pady=3)
        lighten_slider.grid(row=1, column=6, sticky="nsew", padx=3, pady=3)
        contrast_slider.grid(row=1, column=7, sticky="nsew", padx=3, pady=3)
        sat_slider.grid(row=1, column=8, sticky="nsew", padx=3, pady=3)
        ssif_slider.grid(row=1, column=9, sticky="nsew", padx=3, pady=3)
        upscale_slider.grid(row=1, column=10, sticky="nsew", padx=3, pady=3)
        red_slider.grid(row=1, column=11, sticky="nsew", padx=3, pady=3)
        green_slider.grid(row=1, column=12, sticky="nsew", padx=3, pady=3)
        blue_slider.grid(row=1, column=13, sticky="nsew", padx=3, pady=3)

        for col in range(14): # should be set to the number of elements in the bottom frame
            bottom_frame.grid_columnconfigure(col, weight=1)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=4)
        self.root.grid_rowconfigure(1, weight=1)
    

if __name__ == '__main__':
    App()