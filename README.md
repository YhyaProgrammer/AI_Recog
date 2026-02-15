# 🤖 AI Face Recognition System

A real-time face recognition system built with Python, OpenCV, and PyTorch. Features a modern dark-themed GUI with live camera feed and simple database management.

## ✨ Features

- **🔴 Real-time Recognition** - Instant face detection and recognition from webcam
- **📁 Multiple Import Methods** - Add users from camera, select images, or import entire folders
- **🧠 ResNet18 Neural Network** - Powered by PyTorch pre-trained model
- **💾 Persistent Database** - Face embeddings saved locally for fast recognition
- **🎨 Modern UI** - Clean, dark-themed interface with scrollable controls
- **🛠️ Simple Management** - Refresh or clear database with one click

## 📋 Requirements

- 🐍 Python 3.8 or higher
- 📷 Webcam (for live recognition)
- 💻 Windows/Linux/Mac OS

## 🚀 Installation

1. **📥 Clone or download this repository**

2. **📦 Install dependencies**
```bash
pip install -r requirements.txt
```

3. **▶️ Run the application**
```bash
python face_recognition_simple_import.py
```

## 🎯 Usage

### 👤 Adding Users

#### 📸 Method 1: From Camera
1. 👀 Look at the camera
2. ✏️ Enter a name in the "Name" field
3. ➕ Click **CAPTURE & ADD**
4. ✅ Done - user is now in the database

#### 🖼️ Method 2: Select Images
1. 🖱️ Click **SELECT IMAGES**
2. 📁 Choose one or multiple image files (JPG, PNG, BMP, GIF, TIFF)
3. 🏷️ Filename becomes the person's name (e.g., `John.jpg` → "John")
4. 🔄 System automatically processes and adds them

#### 📂 Method 3: Select Folder
1. 📂 Click **SELECT FOLDER**
2. 📁 Choose a folder containing face images
3. 🏷️ All image filenames become names
4. 📦 Bulk import complete

### 🔍 Recognition

- 🎥 Once users are added, the system automatically recognizes faces in the live feed
- 📊 Recognition results show in the STATUS section with confidence score
- 🟢 Green = Recognized user
- 🟠 Orange = Unknown face

### 🗂️ Database Management

- 🔄 **REFRESH** - Reload database from disk
- 🗑️ **CLEAR ALL** - Delete all users (requires confirmation)

## 📁 File Structure

```
📂 project/
├── 🤖 face_recognition_simple_import.py  # Main application
├── 📦 requirements.txt                   # Python dependencies
├── 📄 README.md                         # This file
├── 💾 embeddings.npy                    # Face embeddings database (auto-created)
└── 📁 face_database/                    # Stored face images (auto-created)
    ├── John.jpg
    ├── Sarah.jpg
    └── ...
```

## 🔧 Technical Details

- **🧠 Model**: ResNet18 (pretrained on ImageNet)
- **📐 Recognition Method**: Cosine similarity of face embeddings
- **🎯 Threshold**: 0.65 similarity score (adjustable in code)
- **⚡ Processing**: Every 5th frame for optimal performance
- **💽 Storage**: NumPy arrays for embeddings, JPG for images

## ⚙️ Configuration

You can adjust these settings in the code:

```python
SIMILARITY_THRESHOLD = 0.65  # 🎚️ Recognition sensitivity (0.0-1.0)
DB_PATH = "face_database"    # 📁 Where images are stored
EMBEDDINGS_FILE = "embeddings.npy"  # 💾 Database file
```

## 🐛 Troubleshooting

**📷 Camera not working?**
- 🔒 Make sure no other application is using the webcam
- ⚙️ Check camera permissions in your OS settings
- 🔄 Try restarting the application

**🎯 Recognition not accurate?**
- 💡 Ensure good lighting when capturing faces
- 📸 Add multiple images of the same person from different angles
- ⚖️ Adjust `SIMILARITY_THRESHOLD` (lower = more strict)

**❌ Import fails?**
- 🖼️ Check that image files are valid (not corrupted)
- 👀 Ensure images contain clearly visible faces
- 🔄 Try different image formats (JPG, PNG)

## 💡 Performance Tips

- 🖼️ Use JPG format for faster processing
- 📏 Keep images under 5MB for best performance
- 🚫 Close other camera applications
- 💡 Add well-lit, front-facing photos for best recognition

## 📜 License

This project is open source and available for educational and commercial use.

## 🙏 Credits

Built By the Creator Of Course.
Please Leave a Star if you liked it.


---

**✨ Made for real-time face recognition projects**