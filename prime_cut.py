import cv2
import os
import json

def crop(frame, x=75, y=225, w=500, h=225):
    '''
    Crops frame to show steak by default.
    '''
    return frame[y:y+h, x:x+w]

def find_steak_shape(cropped_frame):
    '''
    Find closest match in steak_crops.
    '''
    input_dir = 'steak_crops/'

    for steak_crop in os.listdir(input_dir):
        input_path = os.path.join(input_dir, steak_crop)
        template = cv2.imread(input_path)

        # Match current steak shape with all possible steak shapes
        result = cv2.matchTemplate(cropped_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # If steak is similar enough (return steak shape)
        if max_val > 0.9:
            steak = os.path.splitext(steak_crop)[0].split('_')[0]
            return steak[-1]

def run_prime_cut():
    # Load steak cut dictionary
    with open('steak_half_cuts.json', 'r') as f:
        cut_dict = json.load(f)

    video_source = 0  # Elgato

    # Open the video capture
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # Main loop
    while True:
        ret, frame = cap.read()

        cropped = crop(frame)
        steak = find_steak_shape(cropped)

        if steak in cut_dict:
            height, width, _ = frame.shape
            x1 = 75 + cut_dict[steak]
            x2 = int(x1 + width / 2)

            cv2.line(frame, (x1, 0), (x1, height), (0, 0, 255), 2)
            cv2.line(frame, (x2, 0), (x2, height), (0, 0, 255), 2)

        cv2.imshow('Prime Cut (press q to quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_prime_cut()