# Static Sign Language Recognition System using CNN

This project is a **real-time static sign language recognition system** that detects hand gestures using a webcam and classifies them into alphabets and commonly used words.

---

## Features

- Real-time hand detection using **cvzone** and **OpenCV**  
- CNN-based gesture classification  
- Supports **alphabets** and **static words**  
- Displays custom sentences for specific gestures (like COMMENT)  
- Clean preprocessing with white background normalization  
- **Easily extendable:** you can add as many new data folders/gestures as you want and retrain the model  

---

## Supported Gestures

| Gesture | Meaning / Display Text |
|---------|----------------------|
| A       | A                    |
| B       | B                    |
| C       | C                    |
| YES     | YES                  |
| NO      | NO                   |
| GOOD    | GOOD                 |
| BAD     | BAD                  |
| ILY     | I Love You           |
| NAMASTE | Namaste              |
| COMMENT | Please use this model and comment your reviews |

> You can add more gesture folders inside the `Data/` directory. Each folder should contain images of that gesture. Update the CNN model to include new gestures during training.

---

## Tech Stack

- **Python**  
- **OpenCV**  
- **cvzone**  
- **TensorFlow / Keras**  
- **NumPy**  

---
