---

# Vehicle Detection and Counting System

This project implements a vehicle detection and counting system using a combination of object detection and tracking techniques. The system can detect vehicles using YOLOv8 or OpenCV methods and track vehicles using the Deep SORT algorithm in a video stream. It counts the number of vehicles entering and exiting a specified region of interest (ROI).

## Features
- Vehicle detection using YOLOv8 or OpenCV methods
- Vehicle tracking using the Deep SORT algorithm
- Counting the number of vehicles entering and exiting a specified ROI
- Real-time processing of video streams

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Ultralytics (for YOLO object detection)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/vehicle-detection.git
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model (`yolov8n.pt`) from the Ultralytics repository and place it in the project directory.

4. Run the main script:

   ```bash
   python main.py
   ```

## Usage

- When prompted, select the detection method (YOLOv8 or OpenCV contour).
- Press `Esc` to exit the program.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to add more details or sections to the README based on your project's specific requirements and features.
