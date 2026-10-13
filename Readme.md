# Hybrid Driver Drowsiness Detection System (AI / ML / DL)

An intelligent Advanced Driver Assistance System (ADAS) computer vision project that prevents vehicular collisions by detecting driver fatigue in real-time through a **Dual-Stream Hybrid Architecture** combining Facial Geometric Biometrics (EAR/MAR) and Deep Learning Spatial Classification (CNN).

---

## 🌟 Key Features

1. **Dual-Stream Hybrid Detection Engine**:
   * **Stream A (Geometric & Temporal Rules)**: Computes Eye Aspect Ratio (**EAR**) and Mouth Aspect Ratio (**MAR**) using 468 3D facial landmarks from MediaPipe FaceMesh. Employs temporal persistence counters (15 consecutive frames) to prevent false positives from standard human blinking (~100–400 ms).
   * **Stream B (Deep Learning Spatial CNN)**: Processes cropped, normalized facial ROI ($145 \times 145 \times 3$) through a Sequential Convolutional Neural Network trained to distinguish active from fatigued states.
   * **AI Decision & Fusion Engine**: Merges geometric and spatial confidence into a unified Fatigue Index:
     $$\text{Fatigue Index} = 0.60 \times \text{Score}_{\text{Geometric}} + 0.40 \times \text{Score}_{\text{CNN}}$$

2. **Adaptive Personalized Driver Calibration**:
   * Evaluates neutral open-eye shape over the first 30 frames (1–2 seconds) and dynamically computes individual thresholds:
     $$\text{Threshold}_{\text{personalized}} = 0.75 \times \text{EAR}_{\text{neutral}}$$

3. **End-to-End MediaPipe Preprocessing**:
   * Eliminates brittle OpenCV Haar Cascade classifiers, leveraging direct FaceMesh landmark extraction to prevent face-tracking failure under varying lighting and non-frontal head poses.

4. **Streamlit Real-Time HUD Deployment**:
   * Powered by `streamlit-webrtc` with low-latency video streaming, displaying live telemetry for EAR, MAR, CNN Confidence, Fatigue Index, and real-time status banners.

---

## 📁 Dataset

The project uses **Dataset2**, which contains 9,120 balanced images:
* `Active Subjects`: 4,560 images
* `Fatigue Subjects`: 4,560 images

Path: `../Dataset2` or `./Data/Dataset2`

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ahmedsaed/Driver-Drowsiness-Detection.git
   cd Driver-Drowsiness-Detection
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Training the Deep Learning Model
Run preprocessing and model training on Dataset2:
```bash
python run.py -p -t --train_dir "../Dataset2"
```
* `-p` / `--preprocess_data`: Extracts facial landmarks and crops ROI to `./Data/landmarks`.
* `-t` / `--train_model`: Trains the Sequential CNN model and saves checkpoints into `./Models/`.

### 2. Launching Real-Time Web Deployment
Launch the interactive Streamlit application:
```bash
streamlit run deploy.py
```
Open your browser at `http://localhost:8501`:
* **Real Time**: Live webcam stream with real-time fatigue telemetry and driver calibration.
* **Upload**: Analyze and generate annotated MP4 video predictions.
* **Train**: Train models directly from the web interface.
* **Logs**: Inspect live debugging logs.

---

## 🧠 Model Architecture

```python
model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.1),

    Conv2D(32, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.1),

    Conv2D(64, 10, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.1),

    Conv2D(128, 12, activation='relu'),
    BatchNormalization(),

    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
```

---

## 📚 Academic References & Literature Citations

1. **Albadawi, Y., Takruri, M., & Awad, M. (2022)**. *A Review of Recent Developments in Driver Drowsiness Detection Systems*. **Sensors**, 22(5), 2069. [doi:10.3390/s22052069](https://doi.org/10.3390/s22052069)
2. **Magán, E., Sesmero, M. P., Alonso-Weber, J. M., & Sanchis, A. (2022)**. *Driver Drowsiness Detection by Applying Deep Learning Techniques to Sequences of Images*. **Applied Sciences**, 12(3), 1145. [doi:10.3390/app12031145](https://doi.org/10.3390/app12031145)
3. **Hashemi, M., Mirrashid, A., & Shirazi, A. B. (2020)**. *Driver Safety Development: Real-Time Driver Drowsiness Detection System Based on Convolutional Neural Network*. **SN Computer Science**, 1(5), 1–10. [doi:10.1007/s42979-020-00277-2](https://doi.org/10.1007/s42979-020-00277-2)
4. **Zhao, Z., Zhou, N., Zhang, L., et al. (2020)**. *Driver Fatigue Detection Based on Convolutional Neural Networks Using EM-CNN*. **Computational Intelligence and Neuroscience**, 2020, 1–11. [doi:10.1155/2020/7254836](https://doi.org/10.1155/2020/7254836)
5. **Soukupová, T., & Čech, J. (2016)**. *Real-Time Eye Blink Detection Using Facial Landmarks*. In **Proceedings of the 21st Computer Vision Winter Workshop (CVWW)**, pp. 22–29.
6. **Dinges, D. F., & Grace, R. (1998)**. *PERCLOS: A Valid Psychophysiological Measure of Alertness as Assessed by Psychomotor Vigilance*. **US Department of Transportation**, Federal Highway Administration.
