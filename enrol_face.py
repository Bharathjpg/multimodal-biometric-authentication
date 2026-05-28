import cv2
import os

# ── Settings ──────────────────────────────────────────────────────────────────
FACE_DATA_DIR = "face_data"
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMAGE_COUNT   = 50   # number of photos to take per person

# ── Setup ─────────────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def enrol_user(user_name):
    save_dir = os.path.join(FACE_DATA_DIR, user_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print(f"\nEnrolling: {user_name}")
    print("Look at the camera. Press SPACE to take a photo.")
    print(f"We need {IMAGE_COUNT} photos. Press Q to quit early.\n")

    count = 0
    while count < IMAGE_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(80, 80))

        # Draw a rectangle around detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (139, 58, 47), 2)

        # Show counter on screen
        cv2.putText(frame, f"Photos taken: {count}/{IMAGE_COUNT}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (139, 58, 47), 2)
        cv2.putText(frame, "SPACE = take photo   Q = quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        cv2.imshow("Enrolment - " + user_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):   # spacebar
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_img = grey[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                face_img = cv2.equalizeHist(face_img)
                img_path = os.path.join(save_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                count += 1
                print(f"  Photo {count} saved.")
            else:
                print("  No face detected clearly. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == IMAGE_COUNT:
        print(f"\nDone! {count} photos saved for {user_name}.")
    else:
        print(f"\nStopped early. {count} photos saved for {user_name}.")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    name = input("Enter your name to enrol: ").strip()
    if name:
        enrol_user(name)
    else:
        print("No name entered.")