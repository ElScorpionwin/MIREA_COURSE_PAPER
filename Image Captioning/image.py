import os
import tkinter as tk
from tkinter import filedialog, Text, messagebox
from PIL import Image, ImageTk
from imageai.Classification import ImageClassification
from transformers import pipeline

# Инициализация модели для классификации изображений
classifier = ImageClassification()
classifier.setModelTypeAsResNet()
classifier.setModelPath("resnet50_weights_tf_dim_ordering_tf_kernels.h5")  # Убедитесь, что файл модели доступен
classifier.loadModel()

# Инициализация модели для генерации текста
text_generator = pipeline("text-generation", model="gpt2")

class ImageCaptioningApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Captioning App")

        self.image_label = tk.Label(master, text="Загрузите изображение:")
        self.image_label.pack()

        self.image_frame = tk.Frame(master)
        self.image_frame.pack()

        self.image_display = tk.Label(self.image_frame)
        self.image_display.pack()

        self.text_output = Text(master, height=10, width=50)
        self.text_output.pack()

        self.load_button = tk.Button(master, text="Обзор", command=self.load_image)
        self.load_button.pack()

        self.process_button = tk.Button(master, text="Обработать изображение", command=self.process_image)
        self.process_button.pack()

        self.images = []

    def load_image(self):
        file_paths = filedialog.askopenfilenames(title="Выберите изображения", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        for file_path in file_paths:
            self.images.append(file_path)
            self.display_image(file_path)

    def display_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img = ImageTk.PhotoImage(img)
        self.image_display.config(image=img)
        self.image_display.image = img

    def process_image(self):
        if not self.images:
            messagebox.showwarning("Предупреждение", "Пожалуйста, загрузите изображение.")
            return

        captions = []
        for image_path in self.images:
            predictions, probabilities = classifier.classifyImage(image_path, result_count=5)
            caption = f"Объекты: {', '.join(predictions)}"
            captions.append(caption)

        # Генерация текста на основе предсказаний
        generated_text = text_generator(" ".join(captions), max_length=50, num_return_sequences=1)[0]['generated_text']
        
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, generated_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()
