"""
Rice Leaf Disease Detection - Mobile App Demo (Streamlit)
----------------------------------------------------------
Interactive web-based mobile simulator for rice leaf disease detection.

Usage:
    streamlit run mobile_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile-like appearance
st.markdown("""
<style>
    .main {
        max-width: 600px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    .disease-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .result-healthy {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .result-diseased {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Disease information database
DISEASE_INFO = {
    'Bacterial Leaf Blight': {
        'severity': '🔴 High',
        'description': 'Caused by Xanthomonas oryzae. Causes water-soaked lesions that turn yellow to white.',
        'treatment': 'Use resistant varieties, apply copper-based bactericides, maintain field hygiene, avoid excessive nitrogen.'
    },
    'Brown Spot': {
        'severity': '🟡 Medium',
        'description': 'Fungal disease caused by Bipolaris oryzae. Creates brown spots with gray centers on leaves.',
        'treatment': 'Apply fungicides (e.g., mancozeb), ensure proper nutrition, avoid water stress, use disease-free seeds.'
    },
    'Healthy Rice Leaf': {
        'severity': '🟢 None',
        'description': 'Leaf is healthy with no visible disease symptoms. Continue regular monitoring.',
        'treatment': 'Continue regular monitoring and preventive care. Maintain proper irrigation and fertilization.'
    },
    'Leaf Blast': {
        'severity': '🔴 High',
        'description': 'Most destructive rice disease caused by Magnaporthe oryzae. Creates diamond-shaped lesions.',
        'treatment': 'Apply systemic fungicides (tricyclazole), use resistant varieties, manage nitrogen levels, improve field drainage.'
    },
    'Leaf scald': {
        'severity': '🟡 Medium',
        'description': 'Fungal disease caused by Microdochium oryzae. Causes scalded appearance with zonate patterns.',
        'treatment': 'Remove infected plants, apply fungicides, improve air circulation, avoid prolonged leaf wetness.'
    },
    'Sheath Blight': {
        'severity': '🟠 High',
        'description': 'Fungal disease caused by Rhizoctonia solani. Affects sheaths, leaves, and grains.',
        'treatment': 'Reduce plant density, apply fungicides (validamycin), maintain proper water management, avoid excessive nitrogen.'
    }
}


@st.cache_resource
def load_model():
    """Load the trained PyTorch model"""

    # Check for rice_disease_detector module first
    try:
        from rice_disease_detector import RiceLeafDiseaseDetector
    except ImportError:
        return None, None, "module_not_found"

    # Define possible model paths
    model_paths = [
        'rice_disease_model.pth',
        'models/rice_disease_model.pth',
        '../rice_disease_model.pth',
        os.path.join(os.getcwd(), 'rice_disease_model.pth'),
        os.path.join(os.path.dirname(__file__), 'rice_disease_model.pth')
    ]

    # Find model file
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        return None, None, "model_not_found"

    # Load model
    try:
        model = RiceLeafDiseaseDetector.load(model_path, device='cpu')
        classes = model.get_classes()
        return model, classes, model_path
    except Exception as e:
        return None, None, f"error: {str(e)}"


def predict_with_timing(model, image):
    """Run prediction and measure inference time"""

    # Save image temporarily
    temp_path = 'temp_image.jpg'
    image.save(temp_path)

    try:
        # Measure inference time
        start_time = time.time()
        result = model.predict(temp_path, return_probs=True)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result, inference_time

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None, 0


def main():
    # Header
    st.title("🌾 Rice Leaf Disease Detector")
    st.markdown("**AI-powered leaf disease diagnosis using Deep Learning**")
    st.markdown("---")

    # Load model
    with st.spinner("🔄 Loading model... Please wait..."):
        model, classes, status = load_model()

    # Handle loading errors
    if model is None:
        if status == "module_not_found":
            st.error("❌ Cannot import `rice_disease_detector` module!")
            st.markdown("""
            **Error:** Module 'rice_disease_detector' not found.
            
            **Solution:**
            1. Download `rice_disease_detector.py` from your project
            2. Place it in the **same folder** as `mobile_app.py`
            3. Restart the app
            
            **Required files in the same folder:**
            - `mobile_app.py` (this app)
            - `rice_disease_detector.py` (model wrapper)
            - `rice_disease_model.pth` (trained model)
            """)

        elif status == "model_not_found":
            st.error("❌ Model file not found!")
            st.markdown("""
            **The model file `rice_disease_model.pth` was not found.**
            
            **Searched in these locations:**
            - Current directory
            - `models/` folder
            - Parent directory
            
            **Solution:**
            1. Download `rice_disease_model.pth` from your project
            2. Place it in the **same folder** as `mobile_app.py`
            3. Restart the app
            """)
        else:
            st.error(f"❌ Error loading model: {status}")

        st.stop()

    # Show success message
    st.success(f"✅ Model loaded successfully from: `{status}`")

    # Display model info
    with st.expander("ℹ️ Model Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Classes", len(classes))
            st.metric("Architecture", "MobileNetV2")
        with col2:
            st.metric("Input Size", "224x224")
            st.metric("Device", "CPU")

        st.markdown("**Detectable Disease Classes:**")
        for i, cls in enumerate(classes, 1):
            st.write(f"{i}. {cls}")

    # File uploader
    st.markdown("### 📸 Upload or Capture Image")

    # Create two columns for upload options
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a rice leaf image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a rice leaf"
        )

    with col2:
        # Camera input (works on mobile devices)
        camera_photo = st.camera_input("Or take a photo")

    # Use camera photo if available, otherwise uploaded file
    image_source = camera_photo if camera_photo is not None else uploaded_file

    if image_source is not None:
        # Load image
        try:
            image = Image.open(image_source).convert('RGB')
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.stop()

        # Display image
        st.markdown("### 🖼️ Captured Image")
        # Use use_column_width for compatibility with older Streamlit versions
        st.image(image, use_column_width=True, caption="Original Image")

        # Image info
        st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")

        # Analyze button
        if st.button("🔍 Analyze Leaf", type="primary"):
            with st.spinner("🔄 Analyzing image... Please wait..."):
                # Run prediction
                result, inference_time = predict_with_timing(model, image)

                if result is None:
                    st.error("Failed to analyze image. Please try again.")
                    st.stop()

                # Get prediction details
                pred_class = result['predicted_class']
                confidence = result['confidence'] * 100
                all_probs = result['all_probabilities']

                # Success message
                st.success("✅ Analysis complete!")

                # Display results
                st.markdown("### 📊 Diagnosis Results")

                # Result card with disease info
                is_healthy = 'Healthy' in pred_class
                card_class = 'result-healthy' if is_healthy else 'result-diseased'

                # Get disease info (handle case variations)
                disease_key = pred_class
                if disease_key not in DISEASE_INFO:
                    # Try to find a matching key
                    for key in DISEASE_INFO.keys():
                        if key.lower() in disease_key.lower() or disease_key.lower() in key.lower():
                            disease_key = key
                            break

                disease_data = DISEASE_INFO.get(disease_key, {
                    'severity': '🟡 Unknown',
                    'description': 'Disease information not available.',
                    'treatment': 'Consult an agricultural expert.'
                })

                st.markdown(f"""
                <div class="disease-card {card_class}">
                    <h2 style="margin-top: 0;">{'✅ ' if is_healthy else '⚠️ '}{pred_class}</h2>
                    <h3>Confidence: {confidence:.1f}%</h3>
                    <p><strong>Severity:</strong> {disease_data['severity']}</p>
                    <p><strong>Description:</strong> {disease_data['description']}</p>
                    <p><strong>Recommended Treatment:</strong> {disease_data['treatment']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Performance metrics
                st.markdown("### ⚡ Performance Metrics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Inference Time", f"{inference_time:.1f} ms")

                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")

                with col3:
                    status_text = "Healthy" if is_healthy else "Diseased"
                    st.metric("Status", status_text)

                # Confidence interpretation
                st.markdown("### 🎯 Confidence Interpretation")
                if confidence >= 90:
                    st.success("✅ **Very High Confidence** - The model is very certain about this diagnosis.")
                elif confidence >= 70:
                    st.info("ℹ️ **High Confidence** - The model is confident about this diagnosis.")
                elif confidence >= 50:
                    st.warning("⚠️ **Moderate Confidence** - Consider taking additional photos or consulting an expert.")
                else:
                    st.error("❌ **Low Confidence** - Please take a clearer photo or consult an agricultural expert.")

                # All probabilities
                st.markdown("### 📊 All Class Probabilities")

                # Sort probabilities
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

                for class_name, prob in sorted_probs:
                    prob_percent = prob * 100

                    # Create columns for class name and percentage
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Highlight predicted class
                        if class_name == pred_class:
                            st.markdown(f"**{class_name}** ⭐ (Predicted)")
                        else:
                            st.markdown(f"{class_name}")

                    with col2:
                        st.markdown(f"**{prob_percent:.2f}%**")

                    # Progress bar
                    st.progress(prob, text="")
                    st.markdown("")  # Spacing

                # Download results button
                st.markdown("### 💾 Download Results")
                result_text = f"""Rice Leaf Disease Detection Results
=====================================

Image Analysis Results:
- Predicted Disease: {pred_class}
- Confidence: {confidence:.2f}%
- Inference Time: {inference_time:.2f} ms
- Status: {'Healthy' if is_healthy else 'Diseased'}

Disease Information:
- Severity: {disease_data['severity']}
- Description: {disease_data['description']}
- Treatment: {disease_data['treatment']}

All Class Probabilities:
"""
                for class_name, prob in sorted_probs:
                    result_text += f"- {class_name}: {prob*100:.2f}%\n"

                st.download_button(
                    label="📄 Download Report as Text",
                    data=result_text,
                    file_name="rice_disease_report.txt",
                    mime="text/plain"
                )

                # Tips
                st.markdown("### 💡 Tips for Better Results")
                st.info("""
                - 📷 **Take clear, well-lit photos** - Ensure good lighting conditions
                - 🎯 **Focus on diseased areas** - Capture symptoms clearly
                - 📐 **Capture the entire leaf** - Include context around symptoms
                - 🌞 **Avoid shadows and reflections** - Use diffused natural light
                - 📏 **Keep camera steady** - Avoid blurry images
                - 🔍 **Fill the frame** - Get close to the leaf
                - 📸 **Multiple angles** - Take photos from different angles for confirmation
                """)

    else:
        # Instructions when no image is loaded
        st.markdown("""
        ### 📱 How to Use This App
        
        1. **Upload** a photo of a rice leaf using the file uploader, or
        2. **Take** a photo directly using your device's camera
        3. Click the **"🔍 Analyze Leaf"** button
        4. View the diagnosis, treatment recommendations, and confidence scores
        
        ### 🌾 Detectable Diseases
        """)

        # Display disease list with info
        if classes:
            for i, disease in enumerate(classes, 1):
                disease_key = disease
                if disease_key not in DISEASE_INFO:
                    for key in DISEASE_INFO.keys():
                        if key.lower() in disease_key.lower():
                            disease_key = key
                            break

                severity = DISEASE_INFO.get(disease_key, {}).get('severity', '🟡 Unknown')
                description = DISEASE_INFO.get(disease_key, {}).get('description', 'No description available')

                with st.expander(f"{i}. {disease} - {severity}"):
                    st.write(f"**Description:** {description}")

        st.markdown("""
        ### ⚡ App Features
        
        - ✅ **Offline Inference** - Works without internet after model is loaded
        - ✅ **Fast Results** - Real-time diagnosis in milliseconds
        - ✅ **Mobile Friendly** - Works on smartphones and tablets
        - ✅ **Detailed Information** - Disease descriptions and treatment recommendations
        - ✅ **Confidence Scores** - Know how certain the model is
        - ✅ **All Probabilities** - See predictions for all disease classes
        - ✅ **Download Reports** - Save diagnosis results as text file
        
        ### 🔬 Model Details
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Architecture", "MobileNetV2")
            st.metric("Input Size", "224×224")
        with col2:
            st.metric("Disease Classes", len(classes) if classes else "N/A")
            st.metric("Inference Device", "CPU")

        st.markdown("""
        - **Training Dataset:** Rice Leaf Diseases Detection dataset
        - **Optimization:** Optimized for edge devices and mobile deployment
        - **Framework:** PyTorch with CPU inference
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 12px; padding: 20px;">
        <p>🤖 <strong>Powered by Deep Learning & Edge AI</strong> | 🌾 <strong>For Agricultural Use</strong></p>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for reference and educational purposes only.</p>
        <p>For critical agricultural decisions, always consult certified agricultural experts and plant pathologists.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()