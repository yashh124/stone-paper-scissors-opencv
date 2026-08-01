"""Rock Paper Scissors played with hand gestures.

Hold a gesture steady until the countdown fires, then the computer plays.

Gestures
    closed fist ............ Rock
    two fingers ............ Scissors
    open hand .............. Paper

Keys
    r  reset scores        ESC / q  quit

Run:  py -3.11 app.py
"""

import os
import random
import sys
import time
from collections import Counter, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvkit import FPS, bootstrap, draw_hud, open_camera  # noqa: E402

bootstrap()

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402

WINDOW = "Stone Paper Scissors"

STABLE_FRAMES = 12
ROUND_COOLDOWN = 2.0  # seconds before the next round can start

TIPS = [4, 8, 12, 16, 20]
BEATS = {"Rock": "Scissors", "Scissors": "Paper", "Paper": "Rock"}


def count_fingers(hand_landmarks, handedness):
    """Number of extended fingers, correct for either hand.

    The thumb comparison flips between hands; the original always used the
    right-hand rule so a left hand mis-counted by one and made "Rock" register
    as "Scissors".
    """
    lm = hand_landmarks.landmark
    fingers = 0

    if handedness == "Left":
        if lm[TIPS[0]].x > lm[TIPS[0] - 1].x:
            fingers += 1
    else:
        if lm[TIPS[0]].x < lm[TIPS[0] - 1].x:
            fingers += 1

    for i in range(1, 5):
        if lm[TIPS[i]].y < lm[TIPS[i] - 2].y:
            fingers += 1

    return fingers


def fingers_to_gesture(fingers):
    if fingers <= 1:
        return "Rock"
    if fingers in (2, 3):
        return "Scissors"
    if fingers >= 4:
        return "Paper"
    return None


def decide_winner(user, comp):
    if user == comp:
        return "Draw"
    return "User" if BEATS[user] == comp else "Computer"


def main():
    cap, width, height = open_camera(1280, 720)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    fps = FPS()

    gesture_buffer = deque(maxlen=STABLE_FRAMES)
    user_score = computer_score = 0
    last_play_time = 0.0
    computer_choice = None
    locked_user_move = None
    result_text = "Show a gesture to begin"

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[rps] dropped frame, retrying...")
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                label = "Right"
                if result.multi_handedness:
                    label = result.multi_handedness[0].classification[0].label

                gesture = fingers_to_gesture(count_fingers(hand_landmarks, label))
                if gesture:
                    gesture_buffer.append(gesture)

                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(120, 120, 120), thickness=1),
                )
            else:
                # Let the buffer decay when the hand leaves, so a stale gesture
                # cannot trigger a round.
                if gesture_buffer:
                    gesture_buffer.popleft()

            # Steady gesture = same value in most of the recent frames.
            steady_gesture = None
            if len(gesture_buffer) >= STABLE_FRAMES:
                move, count = Counter(gesture_buffer).most_common(1)[0]
                if count > STABLE_FRAMES * 0.7:
                    steady_gesture = move

            now = time.time()
            cooldown_left = max(0.0, ROUND_COOLDOWN - (now - last_play_time))

            if steady_gesture and cooldown_left == 0.0:
                computer_choice = random.choice(list(BEATS))
                locked_user_move = steady_gesture
                winner = decide_winner(locked_user_move, computer_choice)

                if winner == "User":
                    user_score += 1
                    result_text = "You Win!"
                elif winner == "Computer":
                    computer_score += 1
                    result_text = "Computer Wins!"
                else:
                    result_text = "Draw!"

                last_play_time = now
                gesture_buffer.clear()

            # -------- UI (sized from the real frame, not a constant) --------
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)

            live = steady_gesture or (gesture_buffer[-1] if gesture_buffer else None)
            cv2.putText(frame, f"You: {live or '--'}", (24, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"CPU: {computer_choice or '--'}", (24, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            (tw, _), _ = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(frame, result_text, ((w - tw) // 2, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.putText(frame, f"You {user_score}", (w - 190, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"CPU {computer_score}", (w - 190, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            if cooldown_left > 0:
                cv2.putText(frame, f"next round in {cooldown_left:.1f}s",
                            (w // 2 - 130, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 2, cv2.LINE_AA)

            fps.tick()
            draw_hud(frame, [f"FPS: {fps:.0f}   |   r reset   ESC/q quit"],
                     origin=(20, h - 20))

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                user_score = computer_score = 0
                computer_choice = locked_user_move = None
                result_text = "Scores reset"

            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
