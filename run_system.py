import cv2
from ultralytics import YOLO
import math

def main():
    # --- CONFIGURATION ---
    # Set to 0 for Webcam, or put a filename like 'test_video.mp4'
    SOURCE = "C:\\Users\\naree\\Desktop\\UI\\backend\\sample_video_2.mp4"
    # SOURCE = "road_test_video.mp4" 
    
    # Load YOUR trained model
    # Make sure this matches the name of the file you just downloaded!
    model = YOLO('C:\\Users\\naree\\Desktop\\UI\\backend\\best.pt') 

    # Class names map (matches your data.yaml)
    classNames = ['Pothole', 'Crack', 'Open Manhole']

    # Start Video
    cap = cv2.VideoCapture(SOURCE)

    # --- FIX STARTS HERE ---
    # 1. Get the original video's FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. Calculate the delay required between frames (in milliseconds)
    # If video is 30fps, delay should be ~33ms
    delay = int(1000 / video_fps)
    print(f"Original Video FPS: {video_fps}. Setting delay to {delay}ms.")
    # --- FIX ENDS HERE ---
    
    # Set webcam resolution (optional, makes it run faster)
    # cap.set(3, 640)
    # cap.set(4, 480)

    print("System Active. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Video ended or camera error.")
            break

        # Run detection
        results = model(frame, stream=True, conf=0.2) # conf=0.5 means 50% confidence required

        # Draw our own custom boxes to make it look like a "System"
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 1. Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 2. Confidence & Class
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                # 3. Logic: Color Coding
                # Red for Manholes (Danger!), Orange for Potholes, Yellow for Cracks
                color = (0, 255, 255) # Default Yellow
                if currentClass == 'Open Manhole':
                    color = (0, 0, 255) # Red
                    cv2.putText(frame, "!!! DANGER !!!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif currentClass == 'Pothole':
                    color = (0, 165, 255) # Orange

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw Label
                cv2.putText(frame, f'{currentClass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the video
        cv2.imshow('Smart Road Hazard System', frame)

        # Exit on 'q'
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()