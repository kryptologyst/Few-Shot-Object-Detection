"""Streamlit demo for few-shot object detection."""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import create_model
from src.utils import get_device, set_seed
from src.data import get_transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Few-Shot Object Detection Demo",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Few-Shot Object Detection Demo")
st.markdown("""
This demo showcases few-shot object detection models that can learn to detect objects 
with very few examples. Upload support images (examples) and query images to see 
how the model performs.
""")

# Sidebar for model selection
st.sidebar.header("Model Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["meta_rcnn", "fsrw", "tfa"],
    help="Choose the few-shot detection model to use"
)

num_classes = st.sidebar.slider(
    "Number of Classes",
    min_value=2,
    max_value=10,
    value=5,
    help="Number of object classes to detect"
)

support_shot = st.sidebar.slider(
    "Support Shots per Class",
    min_value=1,
    max_value=10,
    value=5,
    help="Number of support examples per class"
)

# Initialize model
@st.cache_resource
def load_model(model_name: str, num_classes: int, support_shot: int):
    """Load and cache the model."""
    device = get_device()
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        support_shot=support_shot
    )
    model.to(device)
    model.eval()
    return model, device

# Load model
try:
    model, device = load_model(model_name, num_classes, support_shot)
    st.sidebar.success(f"✅ Loaded {model_name.upper()} model")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Support Images")
    st.markdown("Upload example images for each class (few-shot learning)")
    
    # Support image upload
    support_images = {}
    support_labels = {}
    
    for class_id in range(num_classes):
        st.subheader(f"Class {class_id}")
        
        uploaded_files = st.file_uploader(
            f"Upload support images for Class {class_id}",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key=f"support_class_{class_id}"
        )
        
        if uploaded_files:
            # Display uploaded images
            cols = st.columns(min(len(uploaded_files), 3))
            for i, uploaded_file in enumerate(uploaded_files[:3]):  # Show max 3
                with cols[i % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Support {i+1}", use_column_width=True)
            
            # Store images and labels
            support_images[class_id] = uploaded_files[:support_shot]  # Limit to support_shot
            support_labels[class_id] = [class_id] * len(support_images[class_id])

with col2:
    st.header("🎯 Query Images")
    st.markdown("Upload images to classify using the support examples")
    
    # Query image upload
    query_files = st.file_uploader(
        "Upload query images to classify",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="query_images"
    )
    
    if query_files:
        # Display query images
        cols = st.columns(min(len(query_files), 3))
        for i, uploaded_file in enumerate(query_files):
            with cols[i % 3]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Query {i+1}", use_column_width=True)

# Process images and make predictions
if st.button("🚀 Run Few-Shot Detection", type="primary"):
    
    # Check if we have support and query images
    if not support_images or not any(support_images.values()):
        st.error("❌ Please upload support images for at least one class")
        st.stop()
    
    if not query_files:
        st.error("❌ Please upload query images to classify")
        st.stop()
    
    # Process support images
    st.header("🔄 Processing Images...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get transforms
        transform = get_transforms("val", 224)
        
        # Process support images
        all_support_images = []
        all_support_labels = []
        
        for class_id, images in support_images.items():
            if images:  # Only process classes with images
                for img_file in images:
                    image = Image.open(img_file).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0)
                    all_support_images.append(image_tensor)
                    all_support_labels.append(class_id)
        
        if not all_support_images:
            st.error("❌ No valid support images found")
            st.stop()
        
        # Stack support images
        support_tensor = torch.cat(all_support_images, dim=0).to(device)
        support_labels_tensor = torch.tensor(all_support_labels).to(device)
        
        progress_bar.progress(0.3)
        status_text.text("Processing support images... ✅")
        
        # Process query images
        query_results = []
        
        for i, query_file in enumerate(query_files):
            image = Image.open(query_file).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(support_tensor, support_tensor, support_labels_tensor)
                
                # Get prediction for this query image
                if "logits" in outputs:
                    logits = outputs["logits"]
                elif "roi_logits" in outputs:
                    logits = outputs["roi_logits"]
                else:
                    logits = outputs["proto_logits"]
                
                # Get prediction for the query image
                query_logits = logits[i:i+1] if i < logits.shape[0] else logits[-1:]
                prediction = torch.argmax(query_logits, dim=1).item()
                confidence = F.softmax(query_logits, dim=1).max().item()
                
                query_results.append({
                    'image': image,
                    'prediction': prediction,
                    'confidence': confidence,
                    'logits': query_logits.cpu().numpy()
                })
            
            progress_bar.progress(0.3 + 0.7 * (i + 1) / len(query_files))
            status_text.text(f"Processing query image {i+1}/{len(query_files)}...")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Display results
        st.header("📊 Results")
        
        # Create results grid
        cols = st.columns(min(len(query_results), 3))
        
        for i, result in enumerate(query_results):
            with cols[i % 3]:
                st.subheader(f"Query Image {i+1}")
                
                # Display image
                st.image(result['image'], caption="Query Image", use_column_width=True)
                
                # Display prediction
                pred_class = result['prediction']
                confidence = result['confidence']
                
                # Color code based on confidence
                if confidence > 0.8:
                    color = "🟢"
                elif confidence > 0.6:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"""
                **Predicted Class:** {pred_class}  
                **Confidence:** {color} {confidence:.3f}
                """)
                
                # Show class probabilities
                if 'logits' in result:
                    probs = F.softmax(torch.tensor(result['logits']), dim=1).numpy()[0]
                    
                    # Create probability bar chart
                    fig, ax = plt.subplots(figsize=(6, 4))
                    bars = ax.bar(range(len(probs)), probs, color='skyblue', alpha=0.7)
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Probability')
                    ax.set_title(f'Class Probabilities (Query {i+1})')
                    ax.set_xticks(range(len(probs)))
                    ax.set_xticklabels([f'Class {j}' for j in range(len(probs))])
                    
                    # Highlight predicted class
                    bars[pred_class].set_color('red')
                    
                    # Add probability values on bars
                    for j, (bar, prob) in enumerate(zip(bars, probs)):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        # Summary statistics
        st.header("📈 Summary")
        
        avg_confidence = np.mean([r['confidence'] for r in query_results])
        predictions = [r['prediction'] for r in query_results]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col2:
            st.metric("Total Predictions", len(query_results))
        
        with col3:
            unique_preds = len(set(predictions))
            st.metric("Unique Classes Predicted", unique_preds)
        
        # Class distribution
        st.subheader("Prediction Distribution")
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = list(pred_counts.keys())
        counts = list(pred_counts.values())
        
        bars = ax.bar([f"Class {c}" for c in classes], counts, color='lightcoral', alpha=0.7)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        logger.error(f"Error in demo: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
### About Few-Shot Object Detection

Few-shot object detection is a challenging computer vision task where models learn to detect 
objects with very few labeled examples. This demo showcases three different approaches:

- **Meta-RCNN**: Uses meta-learning with prototypical networks
- **FSRW**: Few-Shot Region-based detection with weighted attention  
- **TFA**: Two-Stage Fine-tuning Approach

The models can adapt to new object classes with just a few support examples, making them 
suitable for scenarios with limited labeled data.
""")

# Add some sample data generation
if st.sidebar.button("🎲 Generate Sample Data"):
    st.sidebar.info("""
    For testing purposes, you can use any images from your computer. 
    The model will attempt to classify them based on the support examples you provide.
    
    Try uploading:
    - Different types of objects (cars, animals, etc.)
    - Images with clear, distinct features
    - Similar images for support examples
    """)
