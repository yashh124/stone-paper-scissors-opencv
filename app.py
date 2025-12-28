import cv2
import mediapipe as mp
import numpy as np
import random
import time

# ---------------- CONFIG ----------------
ROUND_TIME = 2.5   # seconds per round
STABLE_FRAMES = 8  # smoothing frames
# ----------------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

player_score = 0
computer_score = 0

gesture_buffer = []
last_round_time = time.time()
computer_choice = random.choice(["Rock", "Paper", "Scissors"])
round_result = ""

def get_gesture(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(hand_landmarks.landmark[tips[0]].x <
                   hand_landmarks.landmark[tips[0] - 1].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tips[i]].y <
                       hand_landmarks.landmark[tips[i] - 2].y)

    if fingers == [False, False, False, False, False]:
        return "Rock"
    elif fingers == [True, True, True, True, True]:
        return "Paper"
    elif fingers == [False, True, True, False, False]:
        return "Scissors"
    else:
        return None

def decide_winner(player, computer):
    if player == computer:
        return "Draw"
    if (player == "Rock" and computer == "Scissors") or \
       (player == "Paper" and computer == "Rock") or \
       (player == "Scissors" and computer == "Paper"):
        return "Player"
    return "Computer"

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    player_move = None

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(handLms)
            if gesture:
                gesture_buffer.append(gesture)
                if len(gesture_buffer) > STABLE_FRAMES:
                    gesture_buffer.pop(0)

    if gesture_buffer.count(gesture_buffer[-1]) > STABLE_FRAMES // 2:
        player_move = gesture_buffer[-1]

    # ---------- AUTO ROUND UPDATE ----------
    if time.time() - last_round_time > ROUND_TIME and player_move:
        computer_choice = random.choice(["Rock", "Paper", "Scissors"])
        winner = decide_winner(player_move, computer_choice)

        if winner == "Player":
            player_score += 1
            round_result = "YOU WIN!"
        elif winner == "Computer":
            computer_score += 1
            round_result = "COMPUTER WINS!"
        else:
            round_result = "DRAW!"

        last_round_time = time.time()
        gesture_buffer.clear()

    # ---------------- UI ----------------
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

    cv2.putText(frame, f"Player: {player_score}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Computer: {computer_score}", (350, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if player_move:
        cv2.putText(frame, f"You: {player_move}", (30, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"CPU: {computer_choice}", (350, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, round_result, (w//2 - 150, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    cv2.imshow("Rock Paper Scissors - CV", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
