Smart Image Deduplication and Verification System
Project Overview

Managing large image datasets often results in challenges such as duplicate files, low-quality uploads, and fake or modified images. This project provides a Streamlit-based solution to detect duplicates, verify authenticity, and keep only the best quality images.

The system is designed with two key objectives:

Automate duplicate removal using multiple deep learning models.

Ensure authenticity and quality by applying liveness detection and image quality checks.

In addition to automated detection, the application also allows users to manually upload an image to check if it already exists in the database.

Features

Multiple algorithms tested: Uses DeepFace with models like VGG-Face, Facenet, ArcFace, and OpenFace. Benchmarking helps identify which model provides the most reliable results.

Authenticity checks: Detects screenshots, blurred or AI-generated copies using liveness and image quality scoring.

Smart duplicate removal: Among duplicates, only the highest-quality image is preserved based on resolution, sharpness, color richness, and file size.

Manual upload option: Users can upload an image to verify if it is already present in the database.

Streamlit interface: Provides a simple, interactive front-end for both technical and non-technical users.

System Flow

Input: Images from a folder or an uploaded file.

Preprocessing: Image resizing, color histogram calculation, sharpness and brightness analysis.

Feature extraction: Embeddings generated using DeepFace models.

Similarity check: Pairwise comparison using cosine similarity or Euclidean distance.

Authenticity scoring: Liveness detection and image quality metrics.

Smart deletion strategy: Duplicates removed while keeping the best version.

Manual verification: Users can upload an image to check whether it already exists in the database.

Tech Stack

Frontend: Streamlit

Backend: Python

Libraries: DeepFace, OpenCV, Pillow, NumPy, Pandas, scikit-learn

ML Frameworks: TensorFlow, Keras, PyTorch (depending on chosen model)

Installation and Setup
Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

Install Requirements
pip install -r requirements.txt

Run the Application
streamlit run app.py


The application will open in your browser at http://localhost:8501.

Use Cases

Cleaning large image datasets in research or enterprise settings.

Preventing redundant storage in cloud-based platforms.

Verifying identity images in sensitive systems where authenticity is critical.

Manual verification when users want to check whether an image already exists in the system.

Future Improvements

Integration with cloud storage for large-scale deployments.

More advanced liveness detection using deepfake detection techniques.

Optimization to handle millions of images efficiently.
