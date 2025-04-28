import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pygame


# 1. First Command
st.set_page_config(page_title="Gesture Keyboard", layout="wide")

# 2. Then inject CSS
with open('styles/style.css', encoding='utf-8') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Then inject HTML
with open('templates/header.html', encoding='utf-8') as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# 4. Now continue normally
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Keyboard layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['SPACE_____', 'BACK_____', 'ENTER_____']
]

key_width = 50
key_height = 50
keyboard_origin = (50, 250)
typed_text = ""
last_key_time = 0
key_cooldown = 1.0  # seconds
start_time = None
keystroke_log = []

# Sound
pygame.mixer.init()

def play_click_sound():
    pygame.mixer.music.load("click.mp3")
    pygame.mixer.music.play()

def draw_keyboard(frame):
    x, y = keyboard_origin
    colors = [(240, 240, 255), (220, 255, 220), (255, 240, 200), (220, 220, 255)]
    for row_index, row in enumerate(keys):
        for col_index, key in enumerate(row):
            is_special = key in ['SPACE_____', 'BACK_____', 'ENTER_____']
            width = key_width * 2 if is_special else key_width
            top_left = (x, y)
            bottom_right = (x + width, y + key_height)
            color = colors[row_index % len(colors)]
            cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), 2)
            text = "SPACE" if "SPACE" in key else "ENTER" if "ENTER" in key else "BACK" if "BACK" in key else key
            cv2.putText(frame, text, (x + 11, y + 41), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, text, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            x += width + 10
        x = keyboard_origin[0]
        y += key_height + 10

def handle_key_press(key):
    global typed_text, start_time, keystroke_log
    if not start_time:
        start_time = time.time()
    if "SPACE" in key:
        typed_text += " "
        keystroke_log.append((" ", time.time()))
    elif "BACK" in key:
        typed_text = typed_text[:-1]
        keystroke_log.append(("BACK", time.time()))
    elif "ENTER" in key:
        typed_text += "\n"
        keystroke_log.append(("\n", time.time()))
    else:
        typed_text += key
        keystroke_log.append((key, time.time()))

def check_key_press(landmark, frame):
    global last_key_time
    current_time = time.time()
    if current_time - last_key_time < key_cooldown:
        return None
    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
    x_offset, y_offset = keyboard_origin
    for row_index, row in enumerate(keys):
        for col_index, key in enumerate(row):
            width = key_width * 2 if key in ['SPACE_____', 'BACK_____', 'ENTER_____'] else key_width
            key_x = x_offset + sum((key_width * 2 + 10) if k in ['SPACE_____', 'BACK_____', 'ENTER_____'] else (key_width + 10) for k in row[:col_index])
            key_y = y_offset + row_index * (key_height + 10)
            if key_x <= x <= key_x + width and key_y <= y <= key_y + key_height:
                cv2.rectangle(frame, (key_x, key_y), (key_x + width, key_y + key_height), (0, 255, 0), -1)
                cv2.putText(frame, key, (key_x + 10, key_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                last_key_time = current_time
                play_click_sound()
                handle_key_press(key)
                return key
    return None

def custom_loading_spinner():
    spinner_html = """
    <div style="display: flex; justify-content: center; align-items: center; height: 150px;">
        <div style="font-size: 30px; animation: bounce 1s infinite;">🖐️</div>
    </div>

    <style>
    @keyframes bounce {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-20px); }
    }
    </style>
    """
    st.markdown(spinner_html, unsafe_allow_html=True)


if run:
    placeholder = st.empty()
    with placeholder.container():
        custom_loading_spinner()
        time.sleep(2)  # simulate 2 seconds loading

    placeholder.empty()  # Remove spinner after loading
    st.success('Webcam Initialized! 🎥')

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        draw_keyboard(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_tip = hand_landmarks.landmark[8]
                check_key_press(index_tip, frame)

        frame = cv2.putText(frame, f"Typed: {typed_text[-40:].replace(chr(10), ' ↵ ')}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

