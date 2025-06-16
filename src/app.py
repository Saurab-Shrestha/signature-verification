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
import pywt
from scipy.stats import pearsonr

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

DETECTOR_PATH = r"/Users/saurabshrestha/Downloads/cheques/signature_verification/models/signatureyolo.pt"

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
    if results['adjusted_score'] > 0.80 and len(results['red_flags']) == 0:
        st.success("✅ STRONG MATCH")
    elif results['adjusted_score'] > 0.74 and len(results['red_flags']) <= 1:
        st.warning("⚠️ POSSIBLE MATCH")
    else:
        st.error("❌ NO MATCH")
    
    if results['red_flags']:
        st.error("Red Flags Detected:")
        for flag in results['red_flags']:
            st.write(f"• {flag}")
    
    st.subheader("Signature Analysis")
    fig = visualize_boundary_results(results)
    st.pyplot(fig)

def apply_dwt(image, wavelet='db1', level=2):
    """Apply Discrete Wavelet Transform to the image"""
    # Ensure image is float32
    image = image.astype(np.float32)
    
    # Apply DWT
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Extract coefficients
    cA = coeffs[0]  # Approximation coefficients
    cH = coeffs[1][0]  # Horizontal detail coefficients
    cV = coeffs[1][1]  # Vertical detail coefficients
    cD = coeffs[1][2]  # Diagonal detail coefficients
    
    return cA, cH, cV, cD

def extract_swift_features(image, wavelet='db1', level=2):
    """Extract SWIFT features from the signature image"""
    # Apply DWT
    cA, cH, cV, cD = apply_dwt(image, wavelet, level)
    
    # Calculate energy features
    energy_cA = np.sum(cA ** 2)
    energy_cH = np.sum(cH ** 2)
    energy_cV = np.sum(cV ** 2)
    energy_cD = np.sum(cD ** 2)
    
    # Calculate entropy features
    def calculate_entropy(coeffs):
        hist, _ = np.histogram(coeffs.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    entropy_cA = calculate_entropy(cA)
    entropy_cH = calculate_entropy(cH)
    entropy_cV = calculate_entropy(cV)
    entropy_cD = calculate_entropy(cD)
    
    # Calculate mean and standard deviation
    mean_cA = np.mean(cA)
    std_cA = np.std(cA)
    mean_cH = np.mean(cH)
    std_cH = np.std(cH)
    mean_cV = np.mean(cV)
    std_cV = np.std(cV)
    mean_cD = np.mean(cD)
    std_cD = np.std(cD)
    
    # Calculate correlation between subbands
    corr_HV = np.corrcoef(cH.flatten(), cV.flatten())[0, 1]
    corr_HD = np.corrcoef(cH.flatten(), cD.flatten())[0, 1]
    corr_VD = np.corrcoef(cV.flatten(), cD.flatten())[0, 1]
    
    return {
        'energy': {
            'cA': energy_cA,
            'cH': energy_cH,
            'cV': energy_cV,
            'cD': energy_cD
        },
        'entropy': {
            'cA': entropy_cA,
            'cH': entropy_cH,
            'cV': entropy_cV,
            'cD': entropy_cD
        },
        'mean': {
            'cA': mean_cA,
            'cH': mean_cH,
            'cV': mean_cV,
            'cD': mean_cD
        },
        'std': {
            'cA': std_cA,
            'cH': std_cH,
            'cV': std_cV,
            'cD': std_cD
        },
        'correlation': {
            'HV': corr_HV,
            'HD': corr_HD,
            'VD': corr_VD
        }
    }

def calculate_entropy(coeffs):
    """Calculate entropy of coefficients"""
    hist, _ = np.histogram(coeffs.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def analyze_signature(image):
    """Analyze signature image and display results"""
    if image is not None:
        # Convert uploaded file to image
        image_bytes = image.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create figure for visualization
            fig = plt.figure(figsize=(15, 10))
            
            # 1. Original Image
            plt.subplot(2, 4, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            # 2. Binary Image
            plt.subplot(2, 4, 2)
            plt.imshow(binary, cmap='gray')
            plt.title('Binary Image')
            plt.axis('off')
            
            # 3. Contour Analysis
            contour_img = img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            plt.subplot(2, 4, 3)
            plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            plt.title('Contour Analysis')
            plt.axis('off')
            
            # 4. Pressure Analysis (Gradient)
            gradient = cv2.Laplacian(gray, cv2.CV_64F)
            plt.subplot(2, 4, 4)
            plt.imshow(np.abs(gradient), cmap='hot')
            plt.title('Pressure Analysis')
            plt.axis('off')
            
            # 5. DWT Analysis
            cA, cH, cV, cD = apply_dwt(binary)
            plt.subplot(2, 4, 5)
            plt.imshow(cA, cmap='gray')
            plt.title('DWT Approximation')
            plt.axis('off')
            
            plt.subplot(2, 4, 6)
            plt.imshow(np.abs(cH), cmap='hot')
            plt.title('DWT Horizontal Details')
            plt.axis('off')
            
            # 6. SWIFT Features
            swift_features = extract_swift_features(binary)
            
            # Plot SWIFT Energy features
            plt.subplot(2, 4, 7)
            energy_values = list(swift_features['energy'].values())
            energy_names = list(swift_features['energy'].keys())
            plt.bar(energy_names, energy_values)
            plt.title('SWIFT Energy Features')
            plt.xticks(rotation=45)
            
            # Plot SWIFT Entropy features
            plt.subplot(2, 4, 8)
            entropy_values = list(swift_features['entropy'].values())
            entropy_names = list(swift_features['entropy'].keys())
            plt.bar(entropy_names, entropy_values)
            plt.title('SWIFT Entropy Features')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Signature Density", f"{np.sum(binary > 0) / binary.size:.2%}")
                st.metric("Number of Strokes", str(len(contours)))
            
            with col2:
                st.metric("DWT Energy", f"{np.sum(cA**2):.2f}")
                st.metric("DWT Entropy", f"{calculate_entropy(cA):.2f}")
            
            with col3:
                st.metric("SWIFT Energy (cA)", f"{swift_features['energy']['cA']:.2f}")
                st.metric("SWIFT Entropy (cA)", f"{swift_features['entropy']['cA']:.2f}")
            
            with col4:
                st.metric("SWIFT Correlation (HV)", f"{swift_features['correlation']['HV']:.2f}")
                st.metric("SWIFT Mean (cA)", f"{swift_features['mean']['cA']:.2f}")

st.title("✍️ Signature Verification System")
st.markdown("""
This application helps verify signatures using advanced computer vision techniques and CycleGAN for image cleaning.
Upload two signatures to compare them, or use the CycleGAN cleaning feature to enhance signature quality.
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature:", ["Signature Verification", "Signature Cleaning"])

if page == "Signature Verification":
    st.header("Signature Verification")
    
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
        ["Boundary Analysis"]
    )
    
    if signature1 and signature2:
        if st.button("Verify Signatures"):
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            temp_path1 = temp_dir / "temp_sig1.jpg"
            temp_path2 = temp_dir / "temp_sig2.jpg"
            
            with open(temp_path1, "wb") as f:
                f.write(signature1.getvalue())
            with open(temp_path2, "wb") as f:
                f.write(signature2.getvalue())
            
            with st.spinner("Analyzing signatures..."):
                if verification_method == "Boundary Analysis":
                    results = compare_boundary_signatures(str(temp_path1), str(temp_path2), debug=False)
                    display_boundary_results(results)

                os.remove(temp_path1)
                os.remove(temp_path2)

elif page == "Signature Cleaning":
    st.header("Signature Cleaning")
    
    uploaded_file = st.file_uploader("Choose a signature image to clean", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        st.subheader("Original Signature")
        img = resize_image_for_display(uploaded_file.getvalue())
        st.image(img, caption="Original Signature", use_container_width=True)
        
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / "temp_sig.jpg"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        tab1, tab2 = st.tabs(["Signature Cleaning", "Digital Visualization"])
        
        with tab1:
            st.subheader("Select Cleaning Method")
            cleaning_method = st.radio(
                "Choose a cleaning method:",
                ["CycleGAN"],
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
                    analyze_signature(uploaded_file)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit") 