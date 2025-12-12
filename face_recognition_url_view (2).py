import cv2
import face_recognition
import numpy as np
import os
import requests
import base64
import re
from io import BytesIO
from PIL import Image
import tkinter as tk
from tkinter import simpledialog, messagebox

print(os.getcwd())

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def load_image(source):
    if not isinstance(source, str):
        raise ValueError("Source must be a string (URL, data:image..., or local path).")
    source = source.strip()

    
    if source.startswith("data:image"):
        try:
            base64_data = re.sub(r"^data:image/.+;base64,", "", source)
            image_bytes = base64.b64decode(base64_data)
            pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Failed to decode base64: {e}")
            return None

    
    if os.path.exists(source) and os.path.isfile(source):
        try:
            pil_img = Image.open(source).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Failed to open local file: {e}")
            return None

    
    if source.startswith("http://") or source.startswith("https://"):
        try:
            resp = requests.get(source, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request error: {e}")
            return None
        try:
            pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] PIL cannot identify image: {e}")
            return None

    print("[ERROR] Source not recognized.")
    return None

 
def load_known_faces(known_dir):
    known_encodings = []
    known_names = []

    if not os.path.isdir(known_dir):
        print(f"[WARNING] Known faces directory not found: {known_dir}")
        return known_encodings, known_names

    for filename in sorted(os.listdir(known_dir)):
        path = os.path.join(known_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if len(encs) > 0:
                known_encodings.append(encs[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"[WARN] No face in: {filename}")
        except Exception as e:
            print(f"[WARN] Failed {filename}: {e}")
    return known_encodings, known_names

 
def recognize_faces(source, known_dir):
    known_encodings, known_names = load_known_faces(known_dir)
    if len(known_encodings) == 0:
        messagebox.showwarning("No known faces", f"No known faces in:\n{known_dir}")
        return

    img = load_image(source)
    if img is None:
        messagebox.showerror("Load Error", "Could not load the image from source.")
        return

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    if len(face_locations) == 0:
        messagebox.showinfo("No faces", "No faces detected.")
        return

   
    screen_width = 800
    screen_height = 600
    img_height, img_width = img.shape[:2]
    scale = min(screen_width / img_width, screen_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))

     
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3

     
    face_locations_scaled = []
    for (top, right, bottom, left) in face_locations:
        top = int(top * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)
        right = int(right * scale)
        face_locations_scaled.append((top, right, bottom, left))

    
    for (top, right, bottom, left), face_encoding in zip(face_locations_scaled, face_encodings):
        name = "Unknown"
        if len(known_encodings) > 0:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

         
        cv2.rectangle(resized_img, (left, top), (right, bottom), (0, 255, 0), 2)

         
        (text_width, text_height), baseline = cv2.getTextSize(name, font, font_scale, font_thickness)

         
        margin = 10
        rect_top_left = (left, bottom + margin)
        rect_bottom_right = (left + text_width + 40, bottom + margin + text_height + 20)

         
        cv2.rectangle(resized_img, rect_top_left, rect_bottom_right, (0, 255, 0), cv2.FILLED)

         
        text_x = rect_top_left[0] + (rect_bottom_right[0] - rect_top_left[0] - text_width) // 2
        text_y = rect_top_left[1] + text_height + 5

         
        cv2.putText(resized_img, name, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(resized_img, name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    
    window_name = "Face Recognition Result"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, resized_img)
    cv2.resizeWindow(window_name, new_width, new_height)
    cv2.moveWindow(window_name, 100, 50)
    print("[INFO] Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    known_faces_dir = os.path.join(script_dir, "known_faces")

    root = tk.Tk()
    root.withdraw()

    prompt = "Enter your URL or local path:\n"
    user_input = simpledialog.askstring("Input", prompt)

    if user_input and user_input.strip():
        recognize_faces(user_input.strip(), known_faces_dir)
    else:
        print("No input entered. Exiting...")
