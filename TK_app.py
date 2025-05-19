import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import joblib
import numpy as np
import json

label_map = {0: 'Normal', 1: 'Viral Pneumonia', 2: 'Covid'}
MODEL_ROOT = r".\Project\Models"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class XRayApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XRay Covid19 Detect")
        self.geometry("1000x600")
        self.resizable(False, False)
        self.configure(fg_color="#242424")  # Set background color of the app
        self.setup_ui()

    def setup_ui(self):
        # Welcome Page (Divided into two sections with a divider line in between)
        self.welcome_page = ctk.CTkFrame(self, fg_color="#242424")
        self.welcome_page.pack(fill="both", expand=True)

        # Split the page into two sections: Image and Text
        self.slider_frame = ctk.CTkFrame(self.welcome_page, width=1000, height=600, corner_radius=0, fg_color="#242424")
        self.slider_frame.pack(side="top", fill="both", expand=True)

        # Left section (X-ray image)
        self.left_frame = ctk.CTkFrame(self.slider_frame, width=600, height=600, corner_radius=0, fg_color="#242424")
        self.left_frame.pack(side="left", fill="y", expand=True)  # Increased width of the left section
        img = Image.open(r".\Project\01 (1).jpeg")  # Open the image
        img = img.resize((600, 600), Image.Resampling.LANCZOS)  # Resize the image with equal width and height
        tk_img = ImageTk.PhotoImage(img)

        self.img_label = ctk.CTkLabel(self.left_frame, image=tk_img, text="")
        self.img_label.image = tk_img
        self.img_label.pack(expand=True, fill="both")  # Ensure the image takes the full space and maintains equal padding

        # Divider (a thin line between the image and the text) with some space from top and bottom
        self.divider = ctk.CTkFrame(self.slider_frame, width=2, height=500, corner_radius=0, fg_color="gray")
        self.divider.pack(side="left", fill="y", padx=(20, 0))  # Divider now has padding to leave space at top and bottom

        # Right section (Text description)
        self.right_frame = ctk.CTkFrame(self.slider_frame, width=400, height=600, corner_radius=0, fg_color="#242424")
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.text_label = ctk.CTkLabel(
            self.right_frame,
            text="Hello Doctor,\nuse this app to detect and classify chest X-ray images\ninto Normal, Viral Pneumonia, or COVID-19.",
            font=ctk.CTkFont(size=24, weight="bold"),  
            text_color="#ffffff",
            wraplength=350  
        )      
        self.text_label.pack(side="top", pady=100, padx=20)  # Center text vertically

        # Button on the right side after the text
        self.start_button = ctk.CTkButton(
            self.right_frame,
            text="Start",
            command=self.start_app,
            corner_radius=10,
            fg_color="#3e8ef7",
            hover_color="#2c6dd9",
            text_color="#ffffff",
            width=150,  # Increased width for the button
            height=50,  # Increased height for the button
            font=ctk.CTkFont(size=15, weight="bold")  # Increased font size for the text
        )
        self.start_button.pack(side="bottom", pady=100)  # Button at the bottom of the text

    def start_app(self):
        # Hide the welcome page and start the main app
        self.welcome_page.pack_forget()  # Hide the welcome page
        self.setup_main_app()  # Start the main app here

    def setup_main_app(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=20, fg_color="#333333")
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar.pack_propagate(False)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="XRay COVID Detector", font=ctk.CTkFont(size=20, weight="bold"), text_color="#ffffff")
        self.logo_label.pack(pady=(20, 10))

        self.logo_text_label = ctk.CTkLabel(
            self.sidebar,
            text="A diagnostic AI tool for classifying chest X-ray \nimages into one of three categories: \n🟢 Normal \n🟡 Viral Pneumonia \n🔴 COVID-19",
            font=ctk.CTkFont(size=12),
            text_color="#ffffff"
        )
        self.logo_text_label.pack(pady=(10, 10))

        ctk.CTkLabel(self.sidebar, text="Select a Model:", anchor="w", text_color="#ffffff").pack(pady=(10, 2), padx=10, fill="x")

        self.model_var = ctk.StringVar()
        model_dirs = [d for d in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, d))]
        self.model_menu = ctk.CTkComboBox(self.sidebar, values=model_dirs, variable=self.model_var, command=self.load_model_info)
        self.model_menu.pack(padx=10, pady=(0, 15), fill="x")

        # Upload Button
        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="Upload X-Ray Image",
            command=self.upload_image,
            corner_radius=10,
            fg_color="#3e8ef7",
            hover_color="#2c6dd9",
            text_color="#ffffff"
        )
        self.upload_btn.pack(padx=10, pady=(10, 5), fill="x")

        # Divider
        ctk.CTkLabel(self.sidebar, text="───────────────────", text_color="gray").pack()

        self.image_label = ctk.CTkLabel(self.sidebar, text="")
        self.image_label.pack(pady=(10, 5))

        self.result_label = ctk.CTkLabel(self.sidebar, text="", font=ctk.CTkFont(size=18, weight="bold"), text_color="white", wraplength=250)
        self.result_label.pack(padx=10, pady=(10, 0))

        # Main Content
        self.main_content = ctk.CTkScrollableFrame(self, corner_radius=20, fg_color="#333333")
        self.main_content.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.main_content.grid_columnconfigure(0, weight=1)

        self.accuracy_label = None

        self.report_table = ttk.Treeview(self.main_content, columns=("Class", "Precision", "Recall", "F1-Score", "Support"), show="headings", height=5)
        for col in self.report_table["columns"]:
            self.report_table.heading(col, text=col)
            self.report_table.column(col, anchor="center", width=120)
        self.report_table.pack(pady=(10, 10), padx=20)

        style = ttk.Style()
        style.theme_use("default")

        # Modify table style
        style.configure("Treeview", 
                        background="#333333",  # Table background color
                        foreground="white",  # Text color
                        fieldbackground="#333333",  # Cell background color
                        rowheight=25)  # Row height

        style.configure("Treeview.Heading",
                        background="#444444",  # Header background color
                        foreground="white",  # Header text color
                        font=("Arial", 10, "bold"))  # Header font

        style.map("Treeview", background=[("selected", "#3e8ef7")])  # Selected row color

        self.image_frames = []

    def clear_metrics(self):
        for widget in self.image_frames:
            widget.destroy()
        self.image_frames = []

    def load_model_info(self, selected_model=None):
        self.clear_metrics()

        # Clear the uploaded image when model changes
        self.image_label.configure(image="")
        self.image_label.image = None  # Clear the image

        for item in self.report_table.get_children():
            self.report_table.delete(item)
        self.result_label.configure(text="")
        if self.accuracy_label:
            self.accuracy_label.destroy()
            self.accuracy_label = None

        self.model_name = self.model_var.get()
        self.model_folder = os.path.join(MODEL_ROOT, self.model_name)

        report_path = os.path.join(self.model_folder, f"{self.model_name}_report.json")
        cm_path = os.path.join(self.model_folder, f"{self.model_name}_confusion_matrix.png")
        lc_path = os.path.join(self.model_folder, f"{self.model_name}_learning_curve.png")
        loss_path = os.path.join(self.model_folder, f"{self.model_name}_loss_curve.png")

        if os.path.exists(report_path):
            with open(report_path) as f:
                self.report_data = json.load(f)

            acc = self.report_data.get("accuracy", 0)
            class_report = self.report_data.get("classification_report", {})

            accuracy_text = f"Model: {self.model_name} — Accuracy: {acc:.2f}"
            self.accuracy_label = ctk.CTkLabel(self.main_content, text=accuracy_text, font=ctk.CTkFont(size=16, weight="bold"), text_color="#ffffff")
            self.accuracy_label.pack(pady=(10, 0))

            for label, metrics in class_report.items():
                if not label.isdigit():
                    continue
                label_name = label_map.get(int(label), label)
                self.report_table.insert("", "end", values=(
                    label_name,
                    f"{metrics.get('precision', 0):.2f}",
                    f"{metrics.get('recall', 0):.2f}",
                    f"{metrics.get('f1-score', 0):.2f}",
                    metrics.get('support', 0)
                ))

        for title, path in [("Confusion Matrix", cm_path), ("Learning Curve", lc_path), ("Loss Curve", loss_path)]:
            if os.path.exists(path):
                img = Image.open(path).resize((450, 350))
                tk_img = ImageTk.PhotoImage(img)
                lbl = ctk.CTkLabel(self.main_content, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color="#ffffff")
                lbl.pack(pady=(10, 0))
                panel = ctk.CTkLabel(self.main_content, image=tk_img, text="")
                panel.image = tk_img
                panel.pack(pady=(0, 10))
                self.image_frames.append(lbl)
                self.image_frames.append(panel)

    def upload_image(self):
        if not self.model_name:
            messagebox.showwarning("No Model Selected", "Please select a model before uploading an image.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return

        img = Image.open(file_path).convert("L").resize((64, 64))
        display_img = img.resize((200, 200))
        tk_img = ImageTk.PhotoImage(display_img)
        self.image_label.configure(image=tk_img)
        self.image_label.image = tk_img

        img_array = np.array(img).reshape(1, -1)
        model_path = os.path.join(self.model_folder, f"{self.model_name}.pkl")

        try:
            self.model = joblib.load(model_path)
            pred = self.model.predict(img_array)[0]
            pred_label = label_map.get(pred, "Unknown")

            color_map = {
                "Normal": "green",
                "Viral Pneumonia": "orange",
                "Covid": "red"
            }
            self.result_label.configure(text=f"Predicted Class: {pred_label}", text_color=color_map.get(pred_label, "white"))

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not predict image: {e}")

if __name__ == "__main__":
    app = XRayApp()
    app.mainloop()
