import streamlit as st
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import cv2
from skimage import morphology

from core.signature_verification import (
    compare_boundary_signatures, clean_signature_advanced
)
from models.cyclegan import ResnetGenerator, transform_image_for_cyclegan, tensor_to_pil
from models.signature_detector import SignatureDetector
from utils.visualization import visualize_boundary_results

# Set page config
st.set_page_config(
    page_title="Signature Verification System",
    page_icon="✍️",
    layout="wide"
)

DETECTOR_PATH = r"/Users/saurabshrestha/Documents/SignatureVerification/signature/end-to-end/EndToEnd_Signature-Detection-Cleaning-Verification_System_using_YOLOv5-and-CycleGAN/models/drive-download-20250608T062128Z-1-001/Model_Artifacts/yolo_model/signatureyolo.pt"

@st.cache_resource
def get_signature_detector():
    return SignatureDetector(model_path=DETECTOR_PATH, confidence_threshold=0.5)

def resize_image_for_display(image, max_size=(400, 400)):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    ratio = min(max_size[0]/image.width, max_size[1]/image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def display_boundary_results(results):
    """Display boundary-based verification results"""
    st.subheader("Verification Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Feature Similarity", f"{results['similarities']['scalar_avg']:.3f}")
    
    with col2:
        st.metric("Boundary Similarity", f"{results['boundary_similarity']:.3f}")
        st.metric("Final Score", f"{results['final_score']:.3f}")
        st.metric("Adjusted Score", f"{results['adjusted_score']:.3f}")
    
    # Display verdict
    if results['adjusted_score'] > 0.75 and len(results['red_flags']) == 0:
        st.success("✅ STRONG MATCH")
    elif results['adjusted_score'] > 0.65 and len(results['red_flags']) <= 1:
        st.warning("⚠️ POSSIBLE MATCH")
    else:
        st.error("❌ NO MATCH")
    
    # Display red flags if any
    if results['red_flags']:
        st.error("Red Flags Detected:")
        for flag in results['red_flags']:
            st.write(f"• {flag}")
    
    # Display visualizations
    st.subheader("Signature Analysis")
    fig = visualize_boundary_results(results)
    st.pyplot(fig)

def display_texture_results(results):
    """Display texture-based verification results"""
    st.subheader("Texture Analysis Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Texture Similarity", f"{np.mean(list(results['texture_similarities'].values())):.3f}")
        st.metric("Gradient Similarity", f"{results['gradient_similarity']:.3f}")
    
    with col2:
        st.metric("Final Score", f"{results['final_score']:.3f}")
    
    # Display texture feature similarities
    st.subheader("Texture Feature Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(results['texture_similarities'].keys())
    similarities = list(results['texture_similarities'].values())
    
    colors = ['green' if s > 0.6 else 'orange' if s > 0.4 else 'red' for s in similarities]
    ax.bar(features, similarities, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Texture Feature Similarities")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Display verdict
    if results['final_score'] > 0.75:
        st.success("✅ STRONG MATCH")
    elif results['final_score'] > 0.65:
        st.warning("⚠️ POSSIBLE MATCH")
    else:
        st.error("❌ NO MATCH")

def display_combined_results(boundary_results, texture_results):
    """Display combined verification results"""
    st.subheader("Combined Analysis Results")
    
    # Calculate combined score
    weights = {
        'boundary': 0.6,
        'texture': 0.4
    }
    
    combined_score = (
        weights['boundary'] * boundary_results['final_score'] +
        weights['texture'] * texture_results['final_score']
    )
    
    # Create columns for results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Boundary Score", f"{boundary_results['final_score']:.3f}")
        st.metric("Texture Score", f"{texture_results['final_score']:.3f}")
    
    with col2:
        st.metric("Combined Score", f"{combined_score:.3f}")
    
    with col3:
        st.metric("Red Flags", len(boundary_results['red_flags']))
    
    tab1, tab2 = st.tabs(["Boundary Analysis", "Texture Analysis"])
    
    with tab1:
        fig = visualize_boundary_results(boundary_results)
        st.pyplot(fig)
    
    with tab2:
        display_texture_results(texture_results)
    
    # Display verdict
    if combined_score > 0.75 and len(boundary_results['red_flags']) == 0:
        st.success("✅ STRONG MATCH")
    elif combined_score > 0.65 and len(boundary_results['red_flags']) <= 1:
        st.warning("⚠️ POSSIBLE MATCH")
    else:
        st.error("❌ NO MATCH")
    
    # Display red flags if any
    if boundary_results['red_flags']:
        st.error("Red Flags Detected:")
        for flag in boundary_results['red_flags']:
            st.write(f"• {flag}")

st.title("✍️ Signature Verification System")
st.markdown("""
This application helps verify signatures using advanced computer vision techniques and CycleGAN for image cleaning.
Upload two signatures to compare them, or use the CycleGAN cleaning feature to enhance signature quality.
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature:", ["Signature Verification", "Signature Cleaning"])

if page == "Signature Verification":
    st.header("Signature Verification")
    
    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload First Signature")
        signature1 = st.file_uploader("Choose first signature image", type=['png', 'jpg', 'jpeg'])
        if signature1:
            img1 = resize_image_for_display(signature1.getvalue())
            st.image(img1, caption="First Signature", use_container_width=True)
    
    with col2:
        st.subheader("Upload Second Signature")
        signature2 = st.file_uploader("Choose second signature image", type=['png', 'jpg', 'jpeg'])
        if signature2:
            img2 = resize_image_for_display(signature2.getvalue())
            st.image(img2, caption="Second Signature", use_container_width=True)
    
    verification_method = st.radio(
        "Select verification method:",
        ["Boundary Analysis", "Texture Analysis", "Combined Analysis"]
    )
    
    # Verification button
    if signature1 and signature2:
        if st.button("Verify Signatures"):
            # Create temporary files
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            temp_path1 = temp_dir / "temp_sig1.jpg"
            temp_path2 = temp_dir / "temp_sig2.jpg"
            
            # Save uploaded files
            with open(temp_path1, "wb") as f:
                f.write(signature1.getvalue())
            with open(temp_path2, "wb") as f:
                f.write(signature2.getvalue())
            
            # Run verification
            with st.spinner("Analyzing signatures..."):
                if verification_method == "Boundary Analysis":
                    results = compare_boundary_signatures(str(temp_path1), str(temp_path2), debug=False)
                    display_boundary_results(results)

                os.remove(temp_path1)
                os.remove(temp_path2)

elif page == "Signature Cleaning":
    st.header("Signature Cleaning")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a signature image to clean", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        # Display original image
        st.subheader("Original Signature")
        img = resize_image_for_display(uploaded_file.getvalue())
        st.image(img, caption="Original Signature", use_container_width=True)
        
        # Create temporary file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "temp_sig.jpg"
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Add tabs for different features
        tab1, tab2 = st.tabs(["Signature Cleaning", "Digital Visualization"])
        
        with tab1:
            st.subheader("Select Cleaning Method")
            # Cleaning method selection
            cleaning_method = st.radio(
                "Choose a cleaning method:",
                ["CycleGAN", "Advanced Thresholding", "Denoising"],
                horizontal=True
            )
            
            # Model path input for CycleGAN
            if cleaning_method == "CycleGAN":
                model_path = st.text_input(
                    "Enter the path to your CycleGAN model",
                    value=r"/Users/saurabshrestha/Documents/SignatureVerification/signature/end-to-end/EndToEnd_Signature-Detection-Cleaning-Verification_System_using_YOLOv5-and-CycleGAN/models/drive-download-20250608T062128Z-1-001/Model_Artifacts/gan_model/latest_net_G.pth"
                )
            
            if st.button("Clean Signature"):
                with st.spinner("Cleaning signature..."):
                    output_path = temp_dir / "cleaned.jpg"
                    
                    if cleaning_method == "CycleGAN":
                        if not os.path.exists(model_path):
                            st.error("Please provide a valid model path")
                        else:
                            # Initialize CycleGAN model
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            netG = ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9).to(device)
                            
                            state_dict = torch.load(model_path, map_location=device)
                            netG.load_state_dict(state_dict)
                            netG.eval()
                            
                            input_tensor = transform_image_for_cyclegan(Image.open(temp_path)).to(device)
                            with torch.no_grad():
                                output_tensor = netG(input_tensor)
                            
                            output_image = tensor_to_pil(output_tensor)
                            output_image.save(str(output_path))
                    
                    elif cleaning_method == "Advanced Thresholding":
                        cleaned = clean_signature_advanced(str(temp_path), str(output_path))
                    
                    st.subheader("Cleaned Signature")
                    cleaned_img = resize_image_for_display(str(output_path))
                    st.image(cleaned_img, caption=f"Cleaned Signature ({cleaning_method})", use_container_width=True)
                    
                    # Clean up temporary files
                    os.remove(temp_path)
                    os.remove(output_path)
        
        with tab2:
            st.subheader("Digital Signature Analysis")
            
            if st.button("Analyze Signature"):
                with st.spinner("Analyzing signature..."):
                    # Read image
                    image = cv2.imread(str(temp_path))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Create figure for visualization
                    fig = plt.figure(figsize=(15, 10))
                    
                    # 1. Original and Edge Detection
                    plt.subplot(2, 3, 1)
                    plt.imshow(gray, cmap='gray')
                    plt.title('Original Signature')
                    plt.axis('off')
                    
                    # Edge detection
                    edges = cv2.Canny(gray, 50, 150)
                    plt.subplot(2, 3, 2)
                    plt.imshow(edges, cmap='gray')
                    plt.title('Edge Detection')
                    plt.axis('off')
                    
                    # 2. Contour Analysis
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour_img = image.copy()
                    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
                    plt.subplot(2, 3, 3)
                    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
                    plt.title('Contour Analysis')
                    plt.axis('off')
                    
                    # 3. Stroke Analysis
                    skeleton = morphology.skeletonize(gray > 0)
                    plt.subplot(2, 3, 4)
                    plt.imshow(skeleton, cmap='gray')
                    plt.title('Stroke Analysis')
                    plt.axis('off')
                    
                    # 4. Pressure Analysis (using gradient)
                    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
                    plt.subplot(2, 3, 5)
                    plt.imshow(np.abs(gradient), cmap='hot')
                    plt.title('Pressure Analysis')
                    plt.axis('off')
                    
                    # 5. Feature Statistics
                    plt.subplot(2, 3, 6)
                    plt.axis('off')
                    
                    # Calculate statistics
                    total_pixels = gray.size
                    signature_pixels = np.sum(gray < 128)  # Assuming dark signature
                    density = signature_pixels / total_pixels
                    
                    # Calculate stroke width
                    stroke_width = np.mean([cv2.contourArea(c) / cv2.arcLength(c, True) for c in contours if cv2.arcLength(c, True) > 0])
                    
                    # Calculate complexity
                    complexity = len(contours) / (gray.shape[0] * gray.shape[1])
                    
                    # Display statistics
                    stats_text = f"""
                    Signature Statistics:
                    
                    • Total Pixels: {total_pixels:,}
                    • Signature Pixels: {signature_pixels:,}
                    • Density: {density:.2%}
                    • Stroke Width: {stroke_width:.2f}
                    • Complexity: {complexity:.4f}
                    • Number of Strokes: {len(contours)}
                    """
                    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Signature Density", f"{density:.2%}")
                        st.metric("Stroke Width", f"{stroke_width:.2f}")
                    
                    with col2:
                        st.metric("Complexity", f"{complexity:.4f}")
                        st.metric("Number of Strokes", len(contours))
                    
                    with col3:
                        st.metric("Image Size", f"{gray.shape[1]}x{gray.shape[0]}")
                        st.metric("Aspect Ratio", f"{gray.shape[1]/gray.shape[0]:.2f}")
                    
                    # Clean up
                    plt.close(fig)
                    os.remove(temp_path)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 