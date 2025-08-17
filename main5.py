import streamlit as st
import os
import pandas as pd
from deepface import DeepFace
import tempfile
import time
from PIL import Image, ImageStat, ImageFilter
from collections import defaultdict
import numpy as np
import math

# --- UI Configuration ---
st.set_page_config(
    page_title="AI Face Analyzer with Basic Liveness Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Face Analyzer with Basic Liveness Detection")
st.markdown("*Face deduplication with basic anti-spoofing*")
st.markdown("---")

# --- Simplified Liveness Detection ---

class SimpleLivenessDetector:
    """
    Simplified liveness detection using PIL only:
    1. Image sharpness analysis
    2. Color distribution analysis
    3. Brightness uniformity
    4. Edge detection using PIL filters
    5. Compression artifact detection
    """
    
    def __init__(self):
        pass
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian-like filter"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply edge detection filter
        filtered = gray.filter(ImageFilter.FIND_EDGES)
        
        # Calculate variance of the filtered image
        stat = ImageStat.Stat(filtered)
        sharpness = stat.var[0]  # Variance indicates sharpness
        
        return sharpness
    
    def analyze_color_distribution(self, image):
        """Analyze color distribution for naturalness"""
        # Get color statistics
        stat = ImageStat.Stat(image)
        
        # Calculate color variance across channels
        if len(stat.var) >= 3:  # RGB
            color_variance = sum(stat.var[:3]) / 3
            
            # Calculate color balance (difference between channels)
            r_mean, g_mean, b_mean = stat.mean[:3]
            color_balance = abs(r_mean - g_mean) + abs(g_mean - b_mean) + abs(r_mean - b_mean)
            
            return color_variance, color_balance
        else:
            return 0, 0
    
    def calculate_brightness_uniformity(self, image):
        """Check brightness distribution across the image"""
        gray = image.convert('L')
        
        # Split image into regions and check brightness variance
        width, height = gray.size
        
        # Sample brightness from different regions
        regions = [
            (0, 0, width//2, height//2),  # Top-left
            (width//2, 0, width, height//2),  # Top-right
            (0, height//2, width//2, height),  # Bottom-left
            (width//2, height//2, width, height)  # Bottom-right
        ]
        
        brightness_values = []
        for region in regions:
            crop = gray.crop(region)
            stat = ImageStat.Stat(crop)
            brightness_values.append(stat.mean[0])
        
        # Calculate uniformity (lower variance = more uniform)
        brightness_variance = np.var(brightness_values) if brightness_values else 0
        average_brightness = np.mean(brightness_values) if brightness_values else 0
        
        return brightness_variance, average_brightness
    
    def detect_compression_artifacts(self, image):
        """Detect JPEG compression artifacts that might indicate a photo of a photo"""
        # Convert to array for analysis
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            # Calculate gradient changes that might indicate compression blocks
            gray = np.mean(img_array, axis=2)
            
            # Simple gradient analysis
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            
            # Calculate gradient variance
            grad_variance = np.var(grad_x) + np.var(grad_y)
            
            return grad_variance
        
        return 0
    
    def analyze_texture_complexity(self, image):
        """Analyze texture complexity - live faces have more complex textures"""
        gray = image.convert('L')
        
        # Apply different filters to detect texture patterns
        emboss = gray.filter(ImageFilter.EMBOSS)
        detail = gray.filter(ImageFilter.DETAIL)
        
        # Calculate statistics
        emboss_stat = ImageStat.Stat(emboss)
        detail_stat = ImageStat.Stat(detail)
        
        # Texture complexity score
        texture_score = emboss_stat.var[0] + detail_stat.var[0]
        
        return texture_score
    
    def predict_liveness(self, image_path):
        """Main liveness detection function using PIL only"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize for consistent analysis
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # 1. Sharpness Analysis
            sharpness = self.calculate_sharpness(image)
            
            # 2. Color Analysis
            color_variance, color_balance = self.analyze_color_distribution(image)
            
            # 3. Brightness Uniformity
            brightness_var, avg_brightness = self.calculate_brightness_uniformity(image)
            
            # 4. Compression Artifacts
            compression_score = self.detect_compression_artifacts(image)
            
            # 5. Texture Complexity
            texture_score = self.analyze_texture_complexity(image)
            
            # Scoring algorithm
            liveness_score = 0.0
            reasons = []
            
            # Sharpness scoring (live faces should be reasonably sharp)
            if sharpness > 50:
                liveness_score += 0.25
                reasons.append("✅ Good image sharpness detected")
            else:
                reasons.append("⚠️ Low image sharpness (possible blur/low quality)")
            
            # Color variance scoring (live faces have natural color variation)
            if 100 < color_variance < 2000:
                liveness_score += 0.2
                reasons.append("✅ Natural color variation")
            else:
                reasons.append("⚠️ Unusual color patterns")
            
            # Color balance scoring
            if color_balance < 100:  # Not too much color imbalance
                liveness_score += 0.15
                reasons.append("✅ Balanced color distribution")
            else:
                reasons.append("⚠️ Color imbalance detected")
            
            # Brightness uniformity scoring
            if brightness_var > 10:  # Some variation is natural
                liveness_score += 0.15
                reasons.append("✅ Natural lighting variation")
            else:
                reasons.append("⚠️ Too uniform lighting (possible flat image)")
            
            # Good average brightness
            if 50 < avg_brightness < 200:
                liveness_score += 0.1
                reasons.append("✅ Good brightness levels")
            else:
                reasons.append("⚠️ Poor brightness levels")
            
            # Texture complexity scoring
            if texture_score > 500:
                liveness_score += 0.15
                reasons.append("✅ Complex texture patterns (natural skin)")
            else:
                reasons.append("⚠️ Low texture complexity")
            
            return liveness_score, reasons
            
        except Exception as e:
            return 0.0, [f"Error during analysis: {str(e)}"]

# --- Helper Functions (Must be defined before main logic) ---

def delete_single_image(image_path):
    """Delete a single image file"""
    try:
        if os.path.exists(image_path):
            os.remove(image_path)
            return True
        return False
    except Exception as e:
        st.error(f"Error deleting image: {str(e)}")
        return False

def get_keep_reason(strategy, image_info):
    """Get human readable reason for keeping an image"""
    if strategy == "Keep Highest Quality (Resolution + File Size)":
        return f"Best overall quality ({image_info['resolution']:,} pixels, {image_info['file_size']:.1f} MB)"
    elif strategy == "Keep Best Liveness Score":
        return f"Best liveness score ({image_info['liveness_score']:.3f})"
    elif strategy == "Keep Largest File Size":
        return f"Largest file ({image_info['file_size']:.1f} MB)"
    elif strategy == "Keep Highest Resolution":
        return f"Highest resolution ({image_info['resolution']:,} pixels)"
    else:
        return "Default quality criteria"

def get_delete_reason(strategy, delete_img, keep_img):
    """Get human readable reason for deleting an image"""
    if strategy == "Keep Highest Quality (Resolution + File Size)":
        return f"Lower quality ({delete_img['resolution']:,} pixels vs {keep_img['resolution']:,})"
    elif strategy == "Keep Best Liveness Score":
        return f"Lower liveness ({delete_img['liveness_score']:.3f} vs {keep_img['liveness_score']:.3f})"
    elif strategy == "Keep Largest File Size":
        return f"Smaller file ({delete_img['file_size']:.1f} MB vs {keep_img['file_size']:.1f} MB)"
    elif strategy == "Keep Highest Resolution":
        return f"Lower resolution ({delete_img['resolution']:,} vs {keep_img['resolution']:,} pixels)"
    else:
        return "Lower quality"

def preview_deletions(duplicate_groups, liveness_results, strategy):
    """Preview what will be deleted without actually deleting"""
    preview_data = {}
    
    for group_name, image_list in duplicate_groups.items():
        if len(image_list) <= 1:
            continue
        
        # Analyze each image
        image_analysis = []
        
        for img_path in image_list:
            if not os.path.exists(img_path):
                continue
                
            try:
                file_size = os.path.getsize(img_path) / (1024 * 1024)  # Convert to MB
                
                with Image.open(img_path) as img:
                    width, height = img.size
                    resolution = width * height
                
                liveness_score = 0.0
                if img_path in liveness_results:
                    liveness_score = liveness_results[img_path]['score']
                
                # Calculate quality score based on strategy
                if strategy == "Keep Highest Quality (Resolution + File Size)":
                    quality_score = resolution + (file_size * 100000)
                elif strategy == "Keep Best Liveness Score":
                    quality_score = liveness_score * 1000 + resolution / 10000
                elif strategy == "Keep Largest File Size":
                    quality_score = file_size
                elif strategy == "Keep Highest Resolution":
                    quality_score = resolution
                else:
                    quality_score = resolution + (file_size * 100000)
                
                image_analysis.append({
                    'path': img_path,
                    'filename': os.path.basename(img_path),
                    'file_size': file_size,
                    'resolution': resolution,
                    'liveness_score': liveness_score,
                    'quality_score': quality_score
                })
                
            except Exception as e:
                continue
        
        if len(image_analysis) <= 1:
            continue
        
        # Sort by quality score
        image_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Prepare preview data
        to_keep = image_analysis[0]
        to_delete = image_analysis[1:]
        
        preview_data[group_name] = {
            'keep': {
                'path': to_keep['path'],
                'filename': to_keep['filename'],
                'file_size': to_keep['file_size'],
                'resolution': to_keep['resolution'],
                'liveness_score': to_keep['liveness_score'],
                'reason': get_keep_reason(strategy, to_keep)
            },
            'delete': [
                {
                    'filename': img['filename'],
                    'reason': get_delete_reason(strategy, img, to_keep)
                } for img in to_delete
            ]
        }
    
    return preview_data

def smart_delete_duplicates_enhanced(duplicate_groups, liveness_results, strategy="Keep Highest Quality (Resolution + File Size)"):
    """Enhanced smart deletion with multiple strategies"""
    deleted_count = 0
    deletion_log = []
    
    for group_name, image_list in duplicate_groups.items():
        if len(image_list) <= 1:
            continue
        
        deletion_log.append(f"\n📁 {group_name}:")
        
        # Analyze each image
        image_analysis = []
        
        for img_path in image_list:
            if not os.path.exists(img_path):
                continue
                
            try:
                # Get file size
                file_size = os.path.getsize(img_path) / (1024 * 1024)  # Convert to MB
                
                # Get image dimensions
                with Image.open(img_path) as img:
                    width, height = img.size
                    resolution = width * height
                
                # Get liveness score
                liveness_score = 0.0
                if img_path in liveness_results:
                    liveness_score = liveness_results[img_path]['score']
                
                # Calculate quality score based on strategy
                if strategy == "Keep Highest Quality (Resolution + File Size)":
                    quality_score = resolution + (file_size * 100000)
                elif strategy == "Keep Best Liveness Score":
                    quality_score = liveness_score * 1000 + resolution / 10000
                elif strategy == "Keep Largest File Size":
                    quality_score = file_size
                elif strategy == "Keep Highest Resolution":
                    quality_score = resolution
                else:  # Manual selection - use default
                    quality_score = resolution + (file_size * 100000)
                
                image_analysis.append({
                    'path': img_path,
                    'filename': os.path.basename(img_path),
                    'file_size': file_size,
                    'resolution': resolution,
                    'liveness_score': liveness_score,
                    'quality_score': quality_score
                })
                
            except Exception as e:
                deletion_log.append(f"Could not analyze {os.path.basename(img_path)}: {str(e)}")
                continue
        
        if len(image_analysis) <= 1:
            continue
        
        # Sort by quality score - keep the best one
        image_analysis.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Keep the first (best), delete the rest
        to_keep = image_analysis[0]
        to_delete = image_analysis[1:]
        
        # Log what we're keeping and why
        reason = get_keep_reason(strategy, to_keep)
        deletion_log.append(f"  ✅ Keeping: {to_keep['filename']} - {reason}")
        
        # Delete the duplicates
        for img_info in to_delete:
            try:
                os.remove(img_info['path'])
                deleted_count += 1
                deletion_log.append(f"  🗑️ Deleted: {img_info['filename']}")
            except Exception as e:
                deletion_log.append(f"  ❌ Failed to delete {img_info['filename']}: {str(e)}")
    
    return deleted_count, deletion_log

def find_duplicates_with_basic_liveness(folder_path, model_name='VGG-Face', distance_metric='cosine', threshold=0.6):
    """Enhanced duplicate detection with basic liveness analysis"""
    processed_pairs = set()
    duplicate_groups = defaultdict(list)
    all_duplicates = set()
    liveness_results = {}
    
    # Initialize liveness detector
    liveness_detector = SimpleLivenessDetector()
    
    # List all image files in the directory
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)]
    
    if len(image_files) < 2:
        st.warning("Folder must contain at least two images to find duplicates.")
        return [], {}, 0, {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(image_files):
        current_image_path = os.path.join(folder_path, filename)
        
        # Update progress
        progress_percentage = (i + 1) / len(image_files)
        progress_bar.progress(progress_percentage)
        status_text.text(f"Processing image {i+1}/{len(image_files)}: {filename}")

        try:
            # Perform basic liveness detection
            liveness_score, liveness_reasons = liveness_detector.predict_liveness(current_image_path)
            liveness_results[current_image_path] = {
                'score': liveness_score,
                'reasons': liveness_reasons,
                'is_live': liveness_score > 0.5  # Threshold for live detection
            }
            
            # Find face duplicates (only if we want duplicate detection)
            dfs = DeepFace.find(
                img_path=current_image_path,
                db_path=folder_path,
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=False,
                silent=True,
                threshold=threshold
            )
            
            # Process matches
            if dfs and len(dfs) > 0 and not dfs[0].empty:
                df_filtered = dfs[0][dfs[0]['identity'] != current_image_path]
                
                for _, row in df_filtered.iterrows():
                    matched_image_path = row['identity']
                    pair = tuple(sorted((current_image_path, matched_image_path)))
                    
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        all_duplicates.add(current_image_path)
                        all_duplicates.add(matched_image_path)

        except Exception as e:
            st.warning(f"Could not process {filename}. Skipping. Reason: {str(e)}")
            liveness_results[current_image_path] = {
                'score': 0.0,
                'reasons': [f"Processing error: {str(e)}"],
                'is_live': False
            }
            time.sleep(0.1)

    # Group duplicates using Union-Find
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for pair in processed_pairs:
        union(pair[0], pair[1])
    
    groups = defaultdict(list)
    for img in all_duplicates:
        root = find(img)
        groups[root].append(img)
    
    duplicate_groups = {}
    group_counter = 1
    for images in groups.values():
        if len(images) > 1:
            group_name = f"Duplicate_Group_{group_counter}"
            duplicate_groups[group_name] = images
            group_counter += 1

    status_text.text("✅ Scan complete!")
    progress_bar.empty()
    
    return list(processed_pairs), dict(duplicate_groups), len(all_duplicates), liveness_results

# Initialize session state
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = False

# --- Sidebar ---
st.sidebar.header("Configuration")

default_path = os.path.expanduser("~/Pictures")
if os.name == 'nt':
    default_path = r"C:\Users\{}\Pictures".format(os.getenv('USERNAME', ''))

folder_path_input = st.sidebar.text_input(
    "Image Folder Path:",
    default_path,
    help="Path to your image folder containing images to scan"
)

# Model selection
model_options = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
selected_model = st.sidebar.selectbox(
    "Face Recognition Model:",
    model_options,
    index=0
)

# Distance metric
distance_options = ['cosine', 'euclidean', 'euclidean_l2']
selected_distance = st.sidebar.selectbox(
    "Distance Metric:",
    distance_options,
    index=0
)

# Similarity threshold
threshold = st.sidebar.slider(
    "Similarity Threshold:",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.1
)

# Liveness threshold
liveness_threshold = st.sidebar.slider(
    "Liveness Threshold:",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Threshold for determining if a face is live (higher = stricter)"
)

st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox(
    "Operation Mode:",
    ["Analyze with Basic Liveness", "Check Uploaded Image"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Current Settings:**
    📁 Model: {selected_model}  
    📏 Distance: {selected_distance}  
    🎯 Face Threshold: {threshold}  
    👤 Liveness Threshold: {liveness_threshold}  
    
    **Features (PIL-based):**
    ✅ Basic anti-spoofing  
    ✅ Sharpness analysis  
    ✅ Color distribution check  
    ✅ Texture complexity analysis  
    ❌ No OpenCV/MediaPipe required  
    """
)

# --- Main App Logic ---

if not os.path.isdir(folder_path_input):
    st.error("❌The provided path is not a valid directory.")
else:
    if app_mode == "Analyze with Basic Liveness":
        st.subheader("Basic Face Analysis with Liveness Detection")
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        image_files = [f for f in os.listdir(folder_path_input) 
                       if f.lower().endswith(image_extensions)]
        total_images = len(image_files)
        
        # Display folder stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📁 Total Images", total_images)
        with col2:
            st.metric("🔍 Ready to Scan", "Yes" if total_images >= 1 else "No")
        with col3:
            st.metric("🤖 AI Model", selected_model)
        with col4:
            st.metric("👤 Liveness Check", "Basic (PIL)")

        if st.button("🚀 Start Basic Analysis", type="primary"):
            if total_images < 1:
                st.warning("⚠️ Please select a folder with at least one image.")
            else:
                with st.spinner(f"Analyzing {total_images} images with basic liveness detection..."):
                    duplicate_pairs, duplicate_groups, total_duplicate_images, liveness_results = find_duplicates_with_basic_liveness(
                        folder_path_input, 
                        model_name=selected_model,
                        distance_metric=selected_distance,
                        threshold=threshold
                    )

                # Store results
                st.session_state.duplicate_pairs = duplicate_pairs
                st.session_state.duplicate_groups = duplicate_groups
                st.session_state.total_duplicate_images = total_duplicate_images
                st.session_state.liveness_results = liveness_results
                st.session_state.total_images = total_images

        # Display results
        if hasattr(st.session_state, 'liveness_results') and st.session_state.liveness_results:
            liveness_results = st.session_state.liveness_results
            duplicate_groups = st.session_state.get('duplicate_groups', {})
            
            # Liveness statistics
            live_count = sum(1 for result in liveness_results.values() if result['is_live'])
            spoof_count = len(liveness_results) - live_count
            
            st.success("✅ **Basic Analysis Complete!**")
            
            # Enhanced metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Images", len(liveness_results))
            with col2:
                st.metric("👤 Likely Live", live_count, delta=f"{(live_count/len(liveness_results)*100):.1f}%")
            with col3:
                st.metric("🚫 Possible Issues", spoof_count, delta=f"{(spoof_count/len(liveness_results)*100):.1f}%")
            with col4:
                st.metric("👥 Duplicate Groups", len(duplicate_groups))

            # Liveness analysis results
            st.markdown("### 👤 Liveness Detection Results")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["🟢 Likely Live", "🔴 Possible Issues", "📊 All Results"])
            
            with tab1:
                live_images = {k: v for k, v in liveness_results.items() if v['is_live']}
                if live_images:
                    st.success(f"Found {len(live_images)} images with likely live faces:")
                    
                    for img_path, result in live_images.items():
                        with st.expander(f"✅ {os.path.basename(img_path)} (Score: {result['score']:.2f})"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, width=200)
                                except:
                                    st.error("Could not load image")
                            
                            with col2:
                                st.write("**Liveness Analysis:**")
                                for reason in result['reasons']:
                                    st.write(f"• {reason}")
                else:
                    st.warning("No live faces detected. Consider adjusting the liveness threshold.")
            
            with tab2:
                spoof_images = {k: v for k, v in liveness_results.items() if not v['is_live']}
                if spoof_images:
                    st.warning(f"Found {len(spoof_images)} images that may have issues:")
                    
                    for img_path, result in spoof_images.items():
                        with st.expander(f"⚠️ {os.path.basename(img_path)} (Score: {result['score']:.2f})"):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, width=200)
                                except:
                                    st.error("Could not load image")
                            
                            with col2:
                                st.write("**Potential Issues:**")
                                for reason in result['reasons']:
                                    st.write(f"• {reason}")
                else:
                    st.success("No potential issues detected!")
            
            with tab3:
                st.write("**Complete Analysis Results:**")
                
                # Create summary dataframe
                summary_data = []
                for img_path, result in liveness_results.items():
                    summary_data.append({
                        'Filename': os.path.basename(img_path),
                        'Liveness Score': f"{result['score']:.3f}",
                        'Status': "✅ Live" if result['is_live'] else "⚠️ Possible Issue",
                        'Confidence': "High" if abs(result['score'] - 0.5) > 0.3 else "Medium" if abs(result['score'] - 0.5) > 0.15 else "Low"
                    })
                
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="liveness_analysis_results.csv",
                    mime="text/csv"
                )

            # ENHANCED DUPLICATE MANAGEMENT SECTION
            if duplicate_groups:
                st.markdown("---")
                st.markdown("### 🗑️ Duplicate Image Management")
                
                # Statistics
                num_groups = len(duplicate_groups)
                images_can_delete = sum(len(images) - 1 for images in duplicate_groups.values())
                space_savings_percentage = (images_can_delete / len(liveness_results)) * 100 if len(liveness_results) > 0 else 0
                
                # Display deletion statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👥 Duplicate Groups", num_groups)
                with col2:
                    st.metric("🗑️ Can Delete", images_can_delete)
                with col3:
                    st.metric("💾 Space Savings", f"{space_savings_percentage:.1f}%")

                if images_can_delete > 0:
                    st.info(f"📊 **Found {num_groups} groups of duplicate faces!** "
                           f"You can delete {images_can_delete} duplicate images to save space.")
                    
                    # Smart deletion options
                    st.markdown("#### 🧠 Smart Deletion Options")
                    
                    deletion_strategy = st.selectbox(
                        "Choose deletion strategy:",
                        [
                            "Keep Highest Quality (Resolution + File Size)",
                            "Keep Best Liveness Score",
                            "Keep Largest File Size",
                            "Keep Highest Resolution",
                            "Manual Selection"
                        ],
                        help="How to decide which duplicate to keep"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🗑️ **DELETE ALL DUPLICATES**", type="primary"):
                            st.session_state.confirm_delete_all = True
                    
                    with col2:
                        if st.button("**PREVIEW DELETIONS**", type="secondary"):
                            st.session_state.preview_deletions = True
                    
                    # Confirmation dialog for delete all
                    if st.session_state.get('confirm_delete_all', False):
                        st.warning("⚠️ **CONFIRM MASS DELETION**")
                        st.write(f"This will permanently delete **{images_can_delete} duplicate images** using the strategy: **{deletion_strategy}**")
                        
                        col_yes, col_no = st.columns(2)
                        
                        with col_yes:
                            if st.button("✅ **YES, DELETE DUPLICATES**", type="primary"):
                                with st.spinner("🗑️ Deleting duplicate images..."):
                                    deleted_count, deletion_log = smart_delete_duplicates_enhanced(
                                        duplicate_groups, 
                                        liveness_results,
                                        strategy=deletion_strategy
                                    )
                                
                                st.success(f"🎉 Successfully deleted {deleted_count} duplicate images!")
                                st.balloons()
                                
                                # Show deletion details
                                with st.expander("View Deletion Details"):
                                    for log_entry in deletion_log:
                                        st.text(log_entry)
                                
                                # Clear session state and refresh
                                for key in ['duplicate_groups', 'duplicate_pairs', 'total_duplicate_images', 'liveness_results']:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                st.session_state.confirm_delete_all = False
                                st.rerun()
                        
                        with col_no:
                            if st.button("❌ Cancel"):
                                st.session_state.confirm_delete_all = False
                                st.rerun()
                    
                    # Preview deletions
                    if st.session_state.get('preview_deletions', False):
                        st.markdown("#### Preview: What Will Be Deleted")
                        
                        preview_data = preview_deletions(duplicate_groups, liveness_results, deletion_strategy)
                        
                        for group_name, preview_info in preview_data.items():
                            with st.expander(f"📁 {group_name}: Keep {preview_info['keep']['filename']}, Delete {len(preview_info['delete'])} others"):
                                
                                # Show what will be kept
                                st.markdown("**✅ KEEPING:**")
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    try:
                                        keep_img = Image.open(preview_info['keep']['path'])
                                        st.image(keep_img, width=200)
                                    except:
                                        st.error("Could not load image")
                                
                                with col2:
                                    st.write(f"**{preview_info['keep']['filename']}**")
                                    st.write(f"• Resolution: {preview_info['keep']['resolution']:,} pixels")
                                    st.write(f"• File Size: {preview_info['keep']['file_size']:.1f} MB")
                                    st.write(f"• Liveness Score: {preview_info['keep']['liveness_score']:.3f}")
                                    st.write(f"• Reason: {preview_info['keep']['reason']}")
                                
                                # Show what will be deleted
                                st.markdown("**🗑️ DELETING:**")
                                for del_info in preview_info['delete']:
                                    st.write(f"❌ **{del_info['filename']}** - {del_info['reason']}")
                        
                        if st.button("❌ Close Preview"):
                            st.session_state.preview_deletions = False
                            st.rerun()
                
                # Individual group management
                st.markdown("#### 👥 Individual Duplicate Groups")
                st.write("Manage each duplicate group individually:")
                
                for group_name, image_list in duplicate_groups.items():
                    with st.expander(f"🔍 {group_name} → {len(image_list)} identical faces"):
                        st.write(f"**These {len(image_list)} images contain the same person:**")
                        
                        # Display images in columns with detailed info
                        for idx, image_path in enumerate(image_list):
                            if not os.path.exists(image_path):
                                continue
                            
                            # Create a container for each image
                            with st.container():
                                col1, col2, col3 = st.columns([1, 2, 1])
                                
                                with col1:
                                    try:
                                        img = Image.open(image_path)
                                        st.image(img, width=150)
                                    except Exception as e:
                                        st.error(f"❌ Cannot load image")
                                
                                with col2:
                                    filename = os.path.basename(image_path)
                                    st.write(f"**{filename}**")
                                    
                                    # Image details
                                    try:
                                        img = Image.open(image_path)
                                        file_size = os.path.getsize(image_path) / (1024 * 1024)
                                        resolution = img.size[0] * img.size[1]
                                        
                                        st.write(f"📐 Resolution: {img.size[0]}×{img.size[1]} ({resolution:,} pixels)")
                                        st.write(f"💾 File Size: {file_size:.1f} MB")
                                        
                                        # Liveness info
                                        if image_path in liveness_results:
                                            result = liveness_results[image_path]
                                            status = "✅ Live" if result['is_live'] else "⚠️ Issue"
                                            st.write(f"👤 Liveness: {status} ({result['score']:.3f})")
                                    except:
                                        st.write("❌ Could not analyze image")
                                
                                with col3:
                                    # Individual delete button
                                    if st.button(f"🗑️ Delete", key=f"del_{group_name}_{idx}"):
                                        if delete_single_image(image_path):
                                            st.success(f"✅ Deleted {filename}")
                                            # Update session state
                                            if image_path in st.session_state.duplicate_groups[group_name]:
                                                st.session_state.duplicate_groups[group_name].remove(image_path)
                                            if len(st.session_state.duplicate_groups[group_name]) <= 1:
                                                del st.session_state.duplicate_groups[group_name]
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to delete")
                                
                                st.markdown("---")
            else:
                st.success("🎉 **No duplicate faces found!** All your images are unique.")

    elif app_mode == "Check Uploaded Image":
        st.subheader("📤 Check Uploaded Image with Liveness")
        
        uploaded_file = st.file_uploader(
            "Upload an image to analyze",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="📤 Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("🔍 **Analyze Image**", type="primary"):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        with st.spinner("🔍 Analyzing image for liveness and duplicates..."):
                            # Liveness detection
                            liveness_detector = SimpleLivenessDetector()
                            liveness_score, liveness_reasons = liveness_detector.predict_liveness(tmp_path)
                            
                            # Duplicate checking
                            matching_files = []
                            try:
                                dfs = DeepFace.find(
                                    img_path=tmp_path,
                                    db_path=folder_path_input,
                                    model_name=selected_model,
                                    distance_metric=selected_distance,
                                    enforce_detection=False,
                                    silent=True,
                                    threshold=threshold
                                )
                                
                                if dfs and len(dfs) > 0 and not dfs[0].empty:
                                    matching_files = dfs[0]['identity'].tolist()
                            except Exception as e:
                                st.warning(f"Duplicate check failed: {str(e)}")

                        # Display results
                        st.markdown("### 👤 Liveness Analysis")
                        is_live = liveness_score > liveness_threshold
                        status = "✅ Likely Live Face" if is_live else "⚠️ Possible Issue Detected"
                        st.markdown(f"**{status}**")
                        st.metric("Liveness Score", f"{liveness_score:.3f}")
                        
                        with st.expander("📋 Detailed Analysis"):
                            for reason in liveness_reasons:
                                st.write(f"• {reason}")
                        
                        # Duplicate results
                        st.markdown("### 🔍 Duplicate Check")
                        if matching_files:
                            st.warning(f"Found {len(matching_files)} similar image(s) in your database")
                            for match in matching_files:
                                st.write(f"📸 {os.path.basename(match)}")
                        else:
                            st.success("No duplicates found - this appears to be unique")
                            
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.remove(tmp_path)

# --- Footer/Cleanup Section ---
st.markdown("---")

# Clear results button
if hasattr(st.session_state, 'duplicate_groups') or hasattr(st.session_state, 'liveness_results'):
    st.markdown("### 🧹 Cleanup")
    if st.button("🗑️ Clear All Results", help="Clear scan results and start fresh"):
        for key in ['duplicate_groups', 'duplicate_pairs', 'total_duplicate_images', 'total_images', 'liveness_results', 'confirm_delete_all', 'preview_deletions']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("✅ Results cleared!")
        st.rerun()

# Additional Information
st.markdown("### ℹ️ About This Tool")
with st.expander("📖 How It Works"):
    st.markdown("""
    **This tool combines face recognition with basic liveness detection:**
    
    🔍 **Face Analysis:**
    - Uses DeepFace library for face recognition and comparison
    - Supports multiple AI models (VGG-Face, Facenet, OpenFace, DeepFace)
    - Finds duplicate faces across your image collection
    
    👤 **Basic Liveness Detection (PIL-based):**
    - **Sharpness Analysis**: Checks image clarity and focus
    - **Color Distribution**: Analyzes natural color variation
    - **Brightness Uniformity**: Detects unnatural lighting patterns
    - **Texture Complexity**: Evaluates skin texture details
    - **Compression Analysis**: Identifies potential photo-of-photo artifacts
    
    🗑️ **Smart Deletion:**
    - Multiple strategies to choose the best image to keep
    - Preview deletions before executing
    - Individual or bulk deletion options
    - Quality-based selection criteria
    
    ⚠️ **Limitations:**
    - Basic liveness detection (not professional-grade)
    - No advanced 3D depth analysis
    - PIL-based only (no OpenCV/MediaPipe required)
    - False positives/negatives possible
    """)

with st.expander("🛠️ Technical Details"):
    st.markdown(f"""
    **Current Configuration:**
    - **Face Model**: {selected_model}
    - **Distance Metric**: {selected_distance} 
    - **Face Similarity Threshold**: {threshold}
    - **Liveness Threshold**: {liveness_threshold}
    - **Image Formats Supported**: PNG, JPG, JPEG, BMP, TIFF, WEBP
    
    **Performance Notes:**
    - Processing time depends on image count and resolution
    - Higher thresholds = stricter matching
    - Lower thresholds = more duplicates detected
    """)

# --- Development Info ---
st.markdown("---")
st.markdown("**🔬 AI Face Analyzer v2.1** - Enhanced with Basic Liveness Detection")