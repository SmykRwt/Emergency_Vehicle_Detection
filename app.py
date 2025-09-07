import streamlit as st
import cv2
from roboflow import Roboflow
from PIL import Image
import numpy as np
import tempfile
import os

# Set up Streamlit page
st.set_page_config(page_title="🚨 Emergency Vehicle Detection", layout="wide")
st.title("🚨 Emergency Vehicle Detection using Roboflow + Streamlit")

# Sidebar inputs
st.sidebar.title("🔧 Configuration")
api_key = st.sidebar.text_input("🔑 Enter your Roboflow API Key", type="password")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 10, 90, 20)
overlap_threshold = st.sidebar.slider("Overlap Threshold (%)", 10, 90, 50)

# Roboflow project details
project_name = "safety_vehicles-gk05s"
model_version = 1

if api_key:
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(project_name)
        model = project.version(model_version).model
        st.sidebar.success(f"✅ Model loaded: {project_name} (version {model_version})")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
        st.stop()

    # File upload
    uploaded_file = st.file_uploader("📤 Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file:
        file_type = uploaded_file.type

        # --- Image Handling ---
        if file_type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            st.write(f"Image Dimensions: {image.size}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                # Convert PIL to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                if image_cv is None or image_cv.size == 0:
                    st.error("❌ Invalid image format or corrupted image.")
                    st.stop()

                # Resize image to standard size
                image_cv = cv2.resize(image_cv, (640, 640))

                # Save temporary image
                cv2.imwrite(temp_img.name, image_cv)
                if not os.path.exists(temp_img.name) or os.path.getsize(temp_img.name) == 0:
                    st.error("❌ Failed to save temporary image.")
                    st.stop()

                try:
                    # Predict with adjustable parameters
                    prediction_result = model.predict(temp_img.name, confidence=confidence_threshold,
                                                      overlap=overlap_threshold)
                    prediction_json = prediction_result.json()
                    st.subheader("📌 Raw Prediction JSON:")
                    st.json(prediction_json)

                    # Count valid emergency vehicles
                    vehicle_count = 0
                    valid_predictions = []
                    for pred in prediction_json.get("predictions", []):
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        if x > 0 and y > 0 and w > 0 and h > 0 and x - w / 2 >= 0 and y - h / 2 >= 0 and x + w / 2 <= \
                                image_cv.shape[1] and y + h / 2 <= image_cv.shape[0]:
                            valid_predictions.append(pred)
                            vehicle_count += 1
                        else:
                            st.warning(f"⚠️ Invalid bounding box for {pred['class']}: x={x}, y={y}, w={w}, h={h}")

                    st.subheader("🚨 Number of Emergency Vehicles Detected:")
                    st.write(f"**{vehicle_count}** vehicle(s) detected.")

                    if vehicle_count == 0:
                        st.warning(
                            "⚠️ No valid emergency vehicles detected. Try lowering the confidence threshold or check model compatibility.")

                    # Manual rendering
                    image_cv = cv2.imread(temp_img.name)
                    for pred in valid_predictions:
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        label = pred["class"]
                        confidence = pred["confidence"]
                        cv2.rectangle(image_cv, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (0, 255, 0), 2)
                        cv2.putText(image_cv, f"{label} ({confidence:.2f})", (int(x - w / 2), int(y - h / 2 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    result_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="🧠 Prediction Result (Manual Rendering)", use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error during prediction or rendering: {e}")

        # --- Video Handling ---
        elif file_type == "video/mp4":
            stframe = st.empty()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            st.subheader("📼 Processing Video...")
            frame_count = 0
            total_vehicles = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if frame is valid
                if frame is None or frame.size == 0:
                    st.warning("⚠️ Skipped invalid frame.")
                    continue

                # Resize frame
                frame = cv2.resize(frame, (640, 640))

                # Save frame temporarily
                cv2.imwrite("temp_frame.jpg", frame)

                try:
                    prediction_result = model.predict("temp_frame.jpg", confidence=confidence_threshold,
                                                      overlap=overlap_threshold)
                    prediction_json = prediction_result.json()

                    # Count valid emergency vehicles
                    vehicle_count = 0
                    valid_predictions = []
                    for pred in prediction_json.get("predictions", []):
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        if x > 0 and y > 0 and w > 0 and h > 0 and x - w / 2 >= 0 and y - h / 2 >= 0 and x + w / 2 <= \
                                frame.shape[1] and y + h / 2 <= frame.shape[0]:
                            valid_predictions.append(pred)
                            vehicle_count += 1
                        else:
                            st.warning(f"⚠️ Invalid bounding box for {pred['class']}: x={x}, y={y}, w={w}, h={h}")
                    total_vehicles += vehicle_count

                    # Display vehicle count for current frame
                    stframe.write(f"Frame {frame_count}: **{vehicle_count}** vehicle(s) detected")

                    # Manual rendering
                    for pred in valid_predictions:
                        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                        label = pred["class"]
                        confidence = pred["confidence"]
                        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (int(x - w / 2), int(y - h / 2 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    result_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(result_rgb, use_container_width=True)
                    frame_count += 1

                except Exception as e:
                    st.warning(f"⚠️ Error on frame {frame_count}: {e}")
                    frame_count += 1
                    continue

            cap.release()
            st.success(f"✅ Video processing completed. Total vehicles detected: **{total_vehicles}**")

else:
    st.info("🔑 Please enter your Roboflow API key in the sidebar to get started.")