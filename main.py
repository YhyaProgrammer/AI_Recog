import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    def cosine_similarity(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a.flatten(), b.flatten()) / (a_norm * b_norm)
from torchvision import models, transforms
import warnings

warnings.filterwarnings("ignore")


DB_PATH = "face_database"
EMBEDDINGS_FILE = "embeddings.npy"
SIMILARITY_THRESHOLD = 0.65

os.makedirs(DB_PATH, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except:
    model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if os.path.exists(EMBEDDINGS_FILE):
    db_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
else:
    db_embeddings = {}


def extract_embedding(frame_or_path):
    """Extract embedding from a frame or image path"""
    try:
        if isinstance(frame_or_path, str):
            frame = cv2.imread(frame_or_path)
            if frame is None:
                return None
        else:
            frame = frame_or_path

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy()

        return embedding
    except:
        return None


def recognize_face(embedding):
    if not db_embeddings:
        return "Unknown", 0.0

    best_match = None
    best_score = 0

    for name, db_emb in db_embeddings.items():
        try:
            sim = cosine_similarity(embedding, db_emb)
            if isinstance(sim, np.ndarray):
                sim = sim[0][0] if sim.ndim > 1 else sim[0]
        except:
            sim = 0.0

        if sim > best_score:
            best_score = sim
            best_match = name

    if best_score > SIMILARITY_THRESHOLD:
        return best_match, best_score
    else:
        return "Unknown", best_score



class ModernButton(tk.Canvas):
    """Custom modern button widget"""

    def __init__(self, parent, text, command, bg_color="#7c3aed", hover_color="#6d28d9",
                 text_color="#ffffff", width=160, height=45, **kwargs):
        super().__init__(parent, width=width, height=height, bg=parent['bg'],
                         highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text

        self.rect = self.create_rounded_rect(0, 0, width, height, 10, fill=bg_color, outline="")
        self.text_id = self.create_text(width / 2, height / 2, text=text, fill=text_color,
                                        font=("Segoe UI", 12, "bold"))

        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _on_click(self, event):
        if self.command:
            self.command()

    def _on_enter(self, event):
        self.itemconfig(self.rect, fill=self.hover_color)

    def _on_leave(self, event):
        self.itemconfig(self.rect, fill=self.bg_color)


class RecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Recognition System")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self.root.configure(bg="#0a0a0a")

        self.current_frame = None
        self.current_embedding = None
        self.frame_counter = 0
        self.video_running = False

        self.setup_ui()

        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.update_status("Camera Connected", "#10b981")
                self.video_running = True
                self.update_frame()
            else:
                self.update_status("Camera Not Available", "#ef4444")
        except Exception as e:
            self.update_status("Camera Error", "#ef4444")

    def setup_ui(self):
        # Title at top
        title_label = tk.Label(self.root, text="AI Face Recognition System",
                               font=("Segoe UI", 32, "bold"),
                               fg="#ffffff", bg="#0a0a0a")
        title_label.pack(pady=20)

        # Main content container
        content = tk.Frame(self.root, bg="#0a0a0a")
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 30))

        # LEFT COLUMN - Camera
        left_col = tk.Frame(content, bg="#0a0a0a")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 20))

        tk.Label(left_col, text="Live Feed",
                 font=("Segoe UI", 18, "bold"),
                 fg="#ffffff", bg="#0a0a0a").pack(pady=(0, 15))

        # Camera with border
        cam_border = tk.Frame(left_col, bg="#7c3aed", bd=0)
        cam_border.pack()

        cam_inner = tk.Frame(cam_border, bg="#000000")
        cam_inner.pack(padx=3, pady=3)

        self.camera_label = tk.Label(cam_inner, bg="#000000",
                                     text="Initializing Camera...",
                                     font=("Segoe UI", 14),
                                     fg="#666666")
        self.camera_label.pack()

        # RIGHT COLUMN - Scrollable Controls
        right_col_container = tk.Frame(content, bg="#0a0a0a", width=400)
        right_col_container.grid(row=0, column=1, sticky="nsew")
        right_col_container.grid_propagate(False)

        # Create canvas for scrolling
        canvas = tk.Canvas(right_col_container, bg="#0a0a0a", highlightthickness=0)
        scrollbar = tk.Scrollbar(right_col_container, orient="vertical", command=canvas.yview)

        # Scrollable frame inside canvas
        scrollable_frame = tk.Frame(canvas, bg="#0a0a0a")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Configure grid
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)

        # Status Section
        status_box = tk.Frame(scrollable_frame, bg="#141414")
        status_box.pack(fill=tk.X, pady=(0, 20))

        status_content = tk.Frame(status_box, bg="#141414")
        status_content.pack(pady=20, padx=25)

        tk.Label(status_content, text="STATUS",
                 font=("Segoe UI", 14, "bold"),
                 fg="#888888", bg="#141414").pack()

        self.status_label = tk.Label(status_content, text="System Ready",
                                     font=("Segoe UI", 18, "bold"),
                                     fg="#10b981", bg="#141414")
        self.status_label.pack(pady=(8, 0))

        self.info_label = tk.Label(status_content, text="Users & Objects: 0",
                                   font=("Segoe UI", 13),
                                   fg="#999999", bg="#141414")
        self.info_label.pack(pady=(10, 0))

        # Add User Section
        add_box = tk.Frame(scrollable_frame, bg="#141414")
        add_box.pack(fill=tk.X, pady=(0, 20))

        add_content = tk.Frame(add_box, bg="#141414")
        add_content.pack(pady=20, padx=25, fill=tk.X)

        tk.Label(add_content, text="ADD USER FROM CAMERA",
                 font=("Segoe UI", 14, "bold"),
                 fg="#ffffff", bg="#141414").pack(anchor=tk.W)

        tk.Label(add_content, text="Name:",
                 font=("Segoe UI", 11),
                 fg="#aaaaaa", bg="#141414").pack(anchor=tk.W, pady=(15, 5))

        name_box = tk.Frame(add_content, bg="#1a1a1a")
        name_box.pack(fill=tk.X, pady=(0, 15))

        self.name_entry = tk.Entry(name_box,
                                   font=("Segoe UI", 13),
                                   bg="#1a1a1a",
                                   fg="#ffffff",
                                   insertbackground="#ffffff",
                                   relief=tk.FLAT,
                                   bd=0)
        self.name_entry.pack(fill=tk.X, padx=12, pady=10)

        ModernButton(add_content, "CAPTURE & ADD",
                     command=self.add_user_from_camera,
                     width=340, height=50).pack()

        # Import Section
        import_box = tk.Frame(scrollable_frame, bg="#141414")
        import_box.pack(fill=tk.X, pady=(0, 20))

        import_content = tk.Frame(import_box, bg="#141414")
        import_content.pack(pady=20, padx=25, fill=tk.X)

        tk.Label(import_content, text="IMPORT FROM FILES",
                 font=("Segoe UI", 14, "bold"),
                 fg="#ffffff", bg="#141414").pack(anchor=tk.W, pady=(0, 5))

        tk.Label(import_content, text="Filename becomes the person's name",
                 font=("Segoe UI", 10),
                 fg="#888888", bg="#141414").pack(anchor=tk.W, pady=(0, 15))

        ModernButton(import_content, "SELECT IMAGES",
                     command=self.import_images,
                     bg_color="#059669",
                     hover_color="#047857",
                     width=340, height=48).pack(pady=(0, 10))

        ModernButton(import_content, "SELECT FOLDER",
                     command=self.import_folder,
                     bg_color="#0284c7",
                     hover_color="#0369a1",
                     width=340, height=48).pack()

        # Database Management
        manage_box = tk.Frame(scrollable_frame, bg="#141414")
        manage_box.pack(fill=tk.X, pady=(0, 20))

        manage_content = tk.Frame(manage_box, bg="#141414")
        manage_content.pack(pady=20, padx=25, fill=tk.X)

        tk.Label(manage_content, text="DATABASE MANAGEMENT",
                 font=("Segoe UI", 14, "bold"),
                 fg="#ffffff", bg="#141414").pack(anchor=tk.W, pady=(0, 15))

        mgmt_buttons = tk.Frame(manage_content, bg="#141414")
        mgmt_buttons.pack(fill=tk.X)

        ModernButton(mgmt_buttons, "REFRESH",
                     command=self.refresh_database,
                     bg_color="#6b7280",
                     hover_color="#4b5563",
                     width=165, height=45).pack(side=tk.LEFT, padx=(0, 10))

        ModernButton(mgmt_buttons, "CLEAR ALL",
                     command=self.clear_database,
                     bg_color="#dc2626",
                     hover_color="#b91c1c",
                     width=165, height=45).pack(side=tk.LEFT)

        self.update_user_count()

    def update_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    def update_user_count(self):
        user_count = len(db_embeddings)
        self.info_label.config(text=f" Users & Objects: {user_count}")

    def update_frame(self):
        if not self.video_running or self.cap is None or not self.cap.isOpened():
            return

        try:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(30, self.update_frame)
                return

            self.current_frame = frame.copy()

            # Recognition every 5 frames
            if self.frame_counter % 5 == 0:
                embedding = extract_embedding(frame)
                if embedding is not None:
                    self.current_embedding = embedding
                    name, confidence = recognize_face(embedding)

                    if name != "Unknown" and confidence > SIMILARITY_THRESHOLD:
                        self.update_status(f"{name} ({confidence:.2f})", "#10b981")
                    else:
                        self.update_status("Unknown", "#f59e0b")

            self.frame_counter += 1

            # Display frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)

            self.camera_label.config(image=photo, text="", width=640, height=480)
            self.camera_label.image = photo

        except Exception as e:
            pass

        self.root.after(30, self.update_frame)

    def add_user_from_camera(self):
        name = self.name_entry.get().strip()

        if name == "":
            self.update_status("Enter Name First", "#f59e0b")
            return

        if self.current_embedding is None:
            self.update_status("No Frame Available", "#f59e0b")
            return

        if name in db_embeddings:
            self.update_status(f"'{name}' Already Exists", "#f59e0b")
            return

        try:
            db_embeddings[name] = self.current_embedding
            np.save(EMBEDDINGS_FILE, db_embeddings)

            img_path = os.path.join(DB_PATH, f"{name}.jpg")
            cv2.imwrite(img_path, self.current_frame)

            self.update_status(f"'{name}' Added", "#10b981")
            self.name_entry.delete(0, tk.END)
            self.update_user_count()

        except Exception as e:
            self.update_status("Failed to Add", "#ef4444")

    def import_images(self):
        """Select one or multiple images, filename becomes name"""
        filepaths = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not filepaths:
            return

        self.update_status("Importing...", "#0284c7")
        self.root.update()

        try:
            added = 0
            skipped = 0
            errors = 0

            for filepath in filepaths:
                # Get name from filename (without extension)
                filename = os.path.basename(filepath)
                name = os.path.splitext(filename)[0]

                # Skip if already exists
                if name in db_embeddings:
                    skipped += 1
                    continue

                # Extract embedding
                embedding = extract_embedding(filepath)
                if embedding is not None:
                    db_embeddings[name] = embedding

                    # Copy image to database folder
                    dest_path = os.path.join(DB_PATH, f"{name}.jpg")
                    frame = cv2.imread(filepath)
                    if frame is not None:
                        cv2.imwrite(dest_path, frame)
                        added += 1
                    else:
                        errors += 1
                else:
                    errors += 1

            # Save database
            np.save(EMBEDDINGS_FILE, db_embeddings)
            self.update_user_count()
            self.update_status(f"Done: {added} added, {skipped} skipped, {errors} errors", "#10b981")

        except Exception as e:
            self.update_status(f"Import Failed", "#ef4444")

    def import_folder(self):
        """Select folder, all image filenames become names"""
        folder = filedialog.askdirectory(title="Select folder with images")

        if not folder:
            return

        self.update_status("Importing...", "#0284c7")
        self.root.update()

        try:
            added = 0
            skipped = 0
            errors = 0

            # Process all image files in folder
            for filename in os.listdir(folder):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue

                # Get name from filename (without extension)
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(folder, filename)

                # Skip if already exists
                if name in db_embeddings:
                    skipped += 1
                    continue

                # Extract embedding
                embedding = extract_embedding(img_path)
                if embedding is not None:
                    db_embeddings[name] = embedding

                    # Copy image to database folder
                    dest_path = os.path.join(DB_PATH, f"{name}.jpg")
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        cv2.imwrite(dest_path, frame)
                        added += 1
                    else:
                        errors += 1
                else:
                    errors += 1

            # Save database
            np.save(EMBEDDINGS_FILE, db_embeddings)
            self.update_user_count()
            self.update_status(f"Done: {added} added, {skipped} skipped, {errors} errors", "#10b981")

        except Exception as e:
            self.update_status(f"Import Failed", "#ef4444")

    def refresh_database(self):
        global db_embeddings
        if os.path.exists(EMBEDDINGS_FILE):
            db_embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        else:
            db_embeddings = {}
        self.update_user_count()
        self.update_status("Database Refreshed", "#10b981")

    def clear_database(self):
        response = messagebox.askyesno("Confirm", "Delete ALL users? Cannot be undone.")
        if response:
            global db_embeddings
            db_embeddings = {}
            np.save(EMBEDDINGS_FILE, db_embeddings)
            self.update_user_count()
            self.update_status("Database Cleared", "#10b981")

    def on_closing(self):
        self.video_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()