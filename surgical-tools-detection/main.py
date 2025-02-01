# app.py
import streamlit as st
import cv2
import numpy as np
import torch
import time

# Page config
st.set_page_config(
    page_title="Surgical Tools Detection",
    layout="wide"
)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False

# Title
st.title("Surgical Tools Detection Software")

# Define tool classes with generic names
TOOL_CLASSES = {
    0: "Tool1",
    1: "Tool2",
    2: "Tool3",
    3: "Tool4",
    4: "Tool5",
    5: "Tool6",
    6: "Tool7",
    7: "Tool8"
}

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.conf = 0.5  # Confidence threshold
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_frame(frame, model, reference_width_px, reference_width_cm):
    """Process a single frame"""
    try:
        # Convert frame to RGB (YOLOv5 expects RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = model(frame_rgb, size=640)
        
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Calculate pixel-to-cm ratio
        pixel_to_cm_ratio = reference_width_cm / reference_width_px
        
        # Process each detection
        for idx, detection in detections.iterrows():
            # Get coordinates
            x1, y1, x2, y2 = map(int, [
                detection['xmin'],
                detection['ymin'],
                detection['xmax'],
                detection['ymax']
            ])
            
            confidence = detection['confidence']
            class_id = int(detection['class'])
            
            if confidence > 0.5:  # Additional confidence threshold
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate measurements in cm
                width_px = x2 - x1
                height_px = y2 - y1
                width_cm = width_px * pixel_to_cm_ratio
                height_cm = height_px * pixel_to_cm_ratio
                
                # Add label with tool name and measurements
                tool_name = TOOL_CLASSES.get(class_id % len(TOOL_CLASSES))
                label = f"{tool_name} | Width: {width_cm:.2f} cm | Height: {height_cm:.2f} cm"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # Increased font size to 1.0
        
        return frame
    
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return frame

def main():
    # Model loading
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    # Camera selection
    camera_option = st.selectbox(
        'Select Camera',
        options=['Default Camera', 'External Camera'],
        index=0
    )
    
    camera_id = 0 if camera_option == 'Default Camera' else 1
    
    # Create placeholder for video feed
    stframe = st.empty()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button('Start Detection', key='start')
    with col2:
        stop_button = st.button('Stop Detection', key='stop')
    
    if start_button:
        st.session_state.running = True
    
    if stop_button:
        st.session_state.running = False
    
    if st.session_state.running:
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            st.error("Failed to open camera. Please check camera connection.")
            st.session_state.running = False
            return
        
        # Reference object
        reference_width_px = st.number_input('Enter the width of the reference object in pixels', min_value=1)
        reference_width_cm = st.number_input('Enter the width of the reference object in centimeters', min_value=0.1)
        
        if not reference_width_px or not reference_width_cm:
            st.error("Please enter valid reference object dimensions.")
            st.session_state.running = False
            return
        
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame")
                    break
                
                # Process frame
                processed_frame = process_frame(frame, model, reference_width_px, reference_width_cm)
                
                # Display frame
                stframe.image(processed_frame, channels="BGR")
                
                # Add small delay to reduce CPU usage
                time.sleep(0.1)
                
        finally:
            cap.release()
    
    if not st.session_state.running:
        # Display placeholder image or message when not running
        stframe.write("Click 'Start Detection' to begin")

if __name__ == '__main__':
    main()
