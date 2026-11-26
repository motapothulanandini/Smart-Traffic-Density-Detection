# 🚦 Smart Traffic Density Detection

## 📌 Project Description

The **Smart Traffic Density Detection** is an AI-based traffic management system designed to monitor traffic conditions and support dynamic traffic signal control. The system uses YOLO-based deep learning models to detect and classify vehicles from traffic videos and estimate traffic density across multiple lanes.

Based on the detected vehicle count and traffic density, the system dynamically determines green signal duration. It also detects emergency vehicles such as ambulances and gives them priority to help reduce waiting time. The project demonstrates how deep learning and computer vision can be applied to intelligent traffic monitoring and traffic signal management.

## ✨ Features

- 🚗 Vehicle Detection and Classification
- 📹 Process Traffic Videos
- 🛣️ Lane-wise Vehicle Counting
- 📊 Traffic Density Estimation
- 🚦 Dynamic Green Signal Timing
- 🚑 Ambulance Detection and Priority
- 🎥 Generate Annotated Output Video
- 🧠 AI-based Traffic Analysis

## 🛠️ Technologies Used

### Programming Language
- Python

### Deep Learning and Computer Vision
- YOLO
- Ultralytics
- OpenCV
- ByteTrack

### Tools
- Google Colab
- VS Code
- Git
- GitHub

## 🏗️ Project Structure

Smart-Traffic-Density-Detection/
│
├── dataset/
│   └── videos/
│
├── models/
│   ├── vehicle_model.pt
│   └── vehicle_ambulance_best.pt
│
├── src/
│   ├── vehicle_detection.py
│   └── test_controller.py
│
├── outputs/
│   └── output.mp4
│
└── README.md

## 🔄 System Workflow

1. The system takes a traffic video as input.
2. YOLO detects and classifies vehicles in the video.
3. Detected vehicles are tracked and counted across lanes.
4. Traffic density is estimated using the vehicle count.
5. The system determines the green signal duration based on traffic density.
6. If an ambulance is detected, the system gives it priority.
7. The processed video displays vehicle detection and traffic analysis results.

## 🎯 Objective

The main objective of this project is to develop an AI-based traffic monitoring and signal control prototype that uses vehicle detection and traffic density estimation to support efficient traffic management. It also aims to demonstrate emergency vehicle prioritization using deep learning and computer vision.

## 🚀 Future Enhancements

- 🌐 Real-time traffic monitoring using CCTV cameras
- 🚦 Integration with physical traffic signal controllers
- 🛣️ Support for more lanes and multiple intersections
- 📈 Traffic statistics and performance dashboard
- 🧠 Improved vehicle detection under different weather and lighting conditions
- 🚑 Enhanced emergency vehicle detection and prioritization

## 👩‍💻 Developed By

**Nandini Motapothula**

B.Tech – Computer Science and Engineering
