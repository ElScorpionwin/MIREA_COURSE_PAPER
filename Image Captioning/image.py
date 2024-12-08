import tkinter as tk
from tkinter import filedialog, Text, messagebox
from PIL import Image, ImageTk
from imageai.Detection import ObjectDetection
import os
import transformers

class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning App")
        
        # Настройка интерфейса
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.text_field = Text(root, height=10, width=50)
        self.text_field.pack()
        
        self.load_button = tk.Button(root, text="Загрузить изображения", command=self.load_images)
        self.load_button.pack()
        
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath("yolov3.pt")
        self.detector.loadModel()

    def load_images(self):
        files = filedialog.askopenfilenames(initialdir="/", title="Выберите изображения", 
                                             filetypes=(("JPEG файлы", "*.jpg"), ("PNG файлы", "*.png")))
        if files:
            self.process_images(files)

    def process_images(self, files):
        for file_path in files:
            self.display_image(file_path)
            input_image = file_path
            output_image = "output_" + os.path.basename(file_path)
            detections = self.detector.detectObjectsFromImage(input_image=input_image, output_image_path=output_image)
            print(detections)
            keywords = [f"{detection['name']} ({detection['percentage_probability']:.2f}%)" for detection in detections]
            description = f"На изображении: {', '.join(keywords)}."
            self.text_field.insert(tk.END, description + "\n")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()