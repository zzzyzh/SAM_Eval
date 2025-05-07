import os
import json
from tqdm import tqdm

import cv2


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global current_mode
    if event == cv2.EVENT_LBUTTONDOWN and current_mode == "point":
        xy = "%d,%d" % (x, y)
        cv2.circle(image, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(image, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow(window, image)
        point.append((x,y))
        print(x, y)


def on_mouse(event, x, y, flags, param):
    global rect, drawing, current_mode

    if event == cv2.EVENT_LBUTTONDOWN and current_mode == "box":
        # current_mode = "box"
        rect[0], rect[1] = x, y
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP and current_mode == "box":
        rect[2], rect[3] = x, y
        drawing = False

        x_min, x_max = min(rect[0], rect[2]), max(rect[0], rect[2])
        y_min, y_max = min(rect[1], rect[3]), max(rect[1], rect[3])
        
        box.append([x_min, y_min, x_max, y_max])

        print("Selected Rectangle:")
        print(f"Top Left: ({x_min}, {y_min})")
        print(f"Bottom Right: ({x_max}, {y_max})")

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow(window, image)
        

if __name__ == "__main__":
    
    current_mode = "point"  # Initialize mode to point operation
    drawing = False
    
    data_path = '/home/yanzhonghao/data/ven/bhx_sammed'
    mode = 'train'
    json_file = open(os.path.join(data_path, mode, 'image2label_train.json'), "r")
    dataset = json.load(json_file)    
    image_paths = list(dataset.values())
    label_paths = list(dataset.keys())
    
    prompts = {}
    
    for i, lbl_path in enumerate(label_paths):
        im_name = image_paths[i].split('/')[-1].split('.')[0]
        lbl_name = lbl_path.split('/')[-1].split('.')[0]
        obj = lbl_path.split('/')[-1].split('_')[-2]
        print(f'Choose prompt of {obj} for {im_name}')
        
        image = cv2.imread(image_paths[i])
        window = f'{lbl_name}'
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_EVENT_LBUTTONDOWN)
        cv2.setMouseCallback(window, on_mouse)

        point = []  # Initialize a list to store coordinates
        box, rect = [], [0, 0, 0, 0]

        while True:
            cv2.imshow(window, image)
            key = cv2.waitKey(0) & 0xFF

            if key in [ord("q"), 27]:
                break  # Exit the loop when 'q' is pressed

            if key == ord("m"):
                current_mode = "point" if current_mode == "box" else "box"
                print("Switched mode:", current_mode)
                if current_mode == "point":
                    cv2.setMouseCallback(window, on_EVENT_LBUTTONDOWN)
                else:
                    cv2.setMouseCallback(window, on_mouse)

        prompts[lbl_name] = {'point': point, 'box': box}

        if key == 27:
            cv2.destroyWindow(window)
            break

    cv2.destroyAllWindows()

    with open('./prompts.json', 'w', newline='\n') as f:
        json.dump(prompts, f, indent=2)  # 换行显示

