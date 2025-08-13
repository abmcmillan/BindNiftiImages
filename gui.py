import customtkinter as ctk
from tkinter import filedialog, Listbox, END, font
from tkinterdnd2 import DND_FILES, TkinterDnD
import subprocess
import threading
import queue
import SimpleITK as sitk
import numpy as np
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def read_dicom_series(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
    if not dicom_names:
        raise ValueError(f"No DICOM series found in directory: {directory}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()

class ImageViewer(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.image = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Matplotlib Canvas ---
        self.figure = Figure(figsize=(5, 5), dpi=100, facecolor=self.cget("fg_color")[1])
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # --- Viewer Controls ---
        controls_frame = ctk.CTkFrame(self)
        controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.view_var = ctk.StringVar(value="Axial")
        self.mip_var = ctk.BooleanVar(value=False)

        ctk.CTkLabel(controls_frame, text="View:").pack(side="left", padx=5)
        views = ["Axial", "Coronal", "Sagittal"]
        for view in views:
            rb = ctk.CTkRadioButton(controls_frame, text=view, variable=self.view_var, value=view, command=self.update_view)
            rb.pack(side="left", padx=5)

        self.mip_check = ctk.CTkCheckBox(controls_frame, text="MIP", variable=self.mip_var, command=self.update_view)
        self.mip_check.pack(side="left", padx=20)

        self.slice_slider = ctk.CTkSlider(controls_frame, from_=0, to=100, command=lambda val: self.update_view())
        self.slice_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.slice_label = ctk.CTkLabel(controls_frame, text="Slice: 0/0")
        self.slice_label.pack(side="left", padx=5)

    def set_image(self, file_path):
        try:
            if os.path.isdir(file_path):
                self.image = read_dicom_series(file_path)
            else:
                self.image = sitk.ReadImage(str(file_path))
            self.update_view(update_slider=True)
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Failed to load image:\n{os.path.basename(file_path)}", ha='center', va='center', color='white')
            self.ax.axis('off')
            self.canvas.draw()
            self.image = None

    def update_view(self, update_slider=False):
        if self.image is None:
            return

        if update_slider:
            view = self.view_var.get()
            if view == "Axial":
                max_slice = self.image.GetDepth() - 1
            elif view == "Coronal":
                max_slice = self.image.GetHeight() - 1
            else: # Sagittal
                max_slice = self.image.GetWidth() - 1
            self.slice_slider.configure(to=max_slice)
            self.slice_slider.set(max_slice // 2)

        slice_num = int(self.slice_slider.get())
        max_slice_val = int(self.slice_slider.cget("to"))
        self.slice_label.configure(text=f"Slice: {slice_num}/{max_slice_val}")

        self.slice_slider.configure(state="normal" if not self.mip_var.get() else "disabled")

        # Get the 2D slice to display
        img_slice_np = self.get_slice_as_np(slice_num)

        self.ax.clear()
        self.ax.imshow(img_slice_np, cmap='gray', aspect='equal')
        self.ax.axis('off')
        self.canvas.draw()

    def get_slice_as_np(self, slice_num):
        view = self.view_var.get()
        use_mip = self.mip_var.get()

        if use_mip:
            axis = {"Axial": 0, "Coronal": 1, "Sagittal": 2}[view]
            mip_filter = sitk.MaximumProjectionImageFilter()
            mip_filter.SetProjectionDimension(axis)
            img_slice = mip_filter.Execute(self.image)
        else:
            if view == "Axial":
                img_slice = self.image[:, :, slice_num]
            elif view == "Coronal":
                img_slice = self.image[:, slice_num, :]
            else: # Sagittal
                img_slice = self.image[slice_num, :, :]

        return sitk.GetArrayViewFromImage(img_slice).T # Transpose for correct display in matplotlib

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.title("Image Binder")
        self.geometry("1400x900")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.log_queue = queue.Queue()

        # --- Left Frame ---
        self.left_frame = ctk.CTkFrame(self, width=400)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)

        controls_frame = ctk.CTkFrame(self.left_frame)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        # ... (buttons) ...
        ctk.CTkButton(controls_frame, text="Add File(s)", command=self.add_files).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(controls_frame, text="Add Directory", command=self.add_directory).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(controls_frame, text="Remove", command=self.remove_selected).pack(side="left", padx=5, pady=5)
        ctk.CTkButton(controls_frame, text="Clear", command=self.clear_list).pack(side="left", padx=5, pady=5)

        self.file_listbox = Listbox(self.left_frame, selectmode="browse") # browse allows one selection
        self.file_listbox.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.file_listbox.drop_target_register(DND_FILES)
        self.file_listbox.dnd_bind('<<Drop>>', self.handle_drop)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # --- Right Frame ---
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=2) # Viewer
        self.right_frame.grid_rowconfigure(1, weight=1) # Options
        self.right_frame.grid_rowconfigure(2, weight=0) # Status box

        # --- Image Viewer ---
        self.image_viewer = ImageViewer(self.right_frame)
        self.image_viewer.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # --- Options Panel ---
        options_frame = ctk.CTkFrame(self.right_frame)
        options_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        # ... (options widgets) ...
        ctk.CTkLabel(options_frame, text="Output File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.output_path_entry = ctk.CTkEntry(options_frame, width=300)
        self.output_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(options_frame, text="Save As...", command=self.select_output_file).grid(row=0, column=2, padx=5, pady=5)
        interpolator_list = ['linear', 'nearestNeighbor', 'gaussian', 'bSpline', 'cosineWindowedSinc', 'hammingWindowedSinc', 'lanczosWindowedSinc', 'genericLabel']
        ctk.CTkLabel(options_frame, text="Interpolation:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.interp_menu = ctk.CTkOptionMenu(options_frame, values=interpolator_list)
        self.interp_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.as_float_var = ctk.BooleanVar()
        self.as_float_check = ctk.CTkCheckBox(options_frame, text="Save as Float", variable=self.as_float_var)
        self.as_float_check.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(options_frame, text="Voxel Size (X, Y, Z):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        voxel_frame = ctk.CTkFrame(options_frame)
        voxel_frame.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        self.voxel_x = ctk.CTkEntry(voxel_frame, width=60, placeholder_text="auto")
        self.voxel_x.pack(side="left", padx=5)
        self.voxel_y = ctk.CTkEntry(voxel_frame, width=60, placeholder_text="auto")
        self.voxel_y.pack(side="left", padx=5)
        self.voxel_z = ctk.CTkEntry(voxel_frame, width=60, placeholder_text="auto")
        self.voxel_z.pack(side="left", padx=5)

        # --- Run and Status Panel ---
        run_status_frame = ctk.CTkFrame(self.right_frame)
        run_status_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        run_status_frame.grid_columnconfigure(0, weight=1)
        self.run_button = ctk.CTkButton(run_status_frame, text="Run Binding", command=self.run_binding)
        self.run_button.grid(row=0, column=1, padx=10, pady=10)
        self.status_box = ctk.CTkTextbox(run_status_frame, height=80)
        self.status_box.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.log_message("Welcome to the Image Binder!")

    # ... (methods from previous steps) ...
    def add_files(self, files=None):
        if not files:
            files = filedialog.askopenfilenames(title="Select image files")
        if files:
            for file in files: self.file_listbox.insert(END, file)
    def add_directory(self):
        directory = filedialog.askdirectory(title="Select a directory")
        if directory: self.file_listbox.insert(END, directory)
    def remove_selected(self):
        for i in reversed(self.file_listbox.curselection()): self.file_listbox.delete(i)
    def clear_list(self):
        self.file_listbox.delete(0, END)
    def handle_drop(self, event):
        files = self.tk.splitlist(event.data)
        self.add_files(files=files)
    def select_output_file(self):
        file = filedialog.asksaveasfilename(title="Save output file as...", defaultextension=".nii.gz", filetypes=[("Nifti GZ", "*.nii.gz"), ("All files", "*.*")])
        if file:
            self.output_path_entry.delete(0, END)
            self.output_path_entry.insert(0, file)
    def on_file_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            file_path = event.widget.get(index)
            self.image_viewer.set_image(file_path)
    def log_message(self, message, end_of_line=True):
        self.status_box.configure(state="normal")
        self.status_box.insert(END, message + ("\n" if end_of_line else ""))
        self.status_box.see(END)
        self.status_box.configure(state="disabled")
    def process_queue(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if message == "PROCESS_FINISHED":
                    self.run_button.configure(state="normal")
                    self.log_message("\n--- Binding complete! ---")
                    # Auto-display the output file
                    output_file = self.output_path_entry.get()
                    if output_file and os.path.exists(output_file):
                         self.image_viewer.set_image(output_file)
                    return
                elif message.startswith("ERROR:"):
                     self.run_button.configure(state="normal")
                     self.log_message(message)
                     return
                self.log_message(message, end_of_line=False)
        except queue.Empty:
            pass
        if self.binding_thread.is_alive():
            self.after(100, self.process_queue)
        else:
            self.run_button.configure(state="normal")
    def run_subprocess(self, command):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            for line in iter(process.stdout.readline, ''):
                self.log_queue.put(line)
            process.stdout.close()
            stderr_output = process.stderr.read()
            process.stderr.close()
            if process.wait() != 0:
                self.log_queue.put(f"ERROR: Process failed.\n{stderr_output}")
        except Exception as e:
            self.log_queue.put(f"ERROR: An exception occurred: {e}")
        finally:
            self.log_queue.put("PROCESS_FINISHED")
    def run_binding(self):
        input_files = self.file_listbox.get(0, END)
        if not input_files:
            self.log_message("Error: No input files specified.")
            return
        output_file = self.output_path_entry.get()
        if not output_file:
            self.log_message("Error: Output file not specified.")
            return
        command = ['python', 'BindImages.py', '--input'] + list(input_files) + ['--output', output_file]
        if self.as_float_var.get(): command.append('--as_float')
        command.extend(['--interp_type', self.interp_menu.get()])
        voxel_sizes = [self.voxel_x.get(), self.voxel_y.get(), self.voxel_z.get()]
        if all(vs.strip() for vs in voxel_sizes):
            try:
                [float(vs) for vs in voxel_sizes]
                command.extend(['--voxel_size'] + voxel_sizes)
            except ValueError:
                self.log_message("Error: Voxel sizes must be numbers.")
                return
        self.log_message("--- Starting binding process ---")
        self.run_button.configure(state="disabled")
        self.binding_thread = threading.Thread(target=self.run_subprocess, args=(command,))
        self.binding_thread.start()
        self.after(100, self.process_queue)

if __name__ == "__main__":
    app = App()
    app.mainloop()
