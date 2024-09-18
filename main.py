import pyautogui
from PIL import ImageGrab
import time
import tkinter as tk
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model globally
model = YOLO('E:\ws\FFTool\runs\detect\train4\weights\best.pt')  # or your specific model path

# template_files = ['p1.jpg','p2.jpg', 'p3.jpg']  # Add all your template file names here
# templates = []
 
  
# for file in template_files:
#     template = cv2.imread(file)
#     image = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY )
#     cv2.imwrite(file, image)
        
# for file in template_files:
#     template = cv2.imread(file, 0)
#     if template is not None:
#         templates.append((file, template, template.shape[::-1]))

def get_pixel_color_at_mouse():
    x, y = pyautogui.position()
    screenshot = ImageGrab.grab(bbox=(x, y, x+1, y+1))
    color = screenshot.getpixel((0, 0))
    return color

def create_canvas():
    root = tk.Tk()
    root.title("Object Detector")
    
    # Set canvas size to 300x300
    canvas_width, canvas_height = 300, 300
    
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate position for the middle of the screen
    x = (screen_width - canvas_width) // 2
    y = (screen_height - canvas_height) // 2
    
    root.geometry(f"{canvas_width}x{canvas_height}+{x}+{y}")
    
    # Make the window transparent and always on top
    root.attributes('-alpha', 1)
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    
    # Use a system-specific transparent color
    transparent_color = 'systemTransparent' if root.tk.call('tk', 'windowingsystem') == 'aqua' else 'gray'
    root.config(bg=transparent_color)
    root.attributes('-transparentcolor', transparent_color)
    
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg=transparent_color, highlightthickness=0)
    canvas.pack()
    
    # Draw a red rectangle to mark the detection area
    canvas.create_rectangle(2, 2, canvas_width-2, canvas_height-2, outline="green", width=2)
    
    return root, canvas

def move_canvas(root, start_x, start_y, end_x, end_y, duration=0.1):
    steps = 10
    step_x = (end_x - start_x) / steps
    step_y = (end_y - start_y) / steps
    
    def animate(step):
        if step < steps:
            x = int(start_x + step * step_x)
            y = int(start_y + step * step_y)
            root.geometry(f"+{x}+{y}")
            root.after(duration // steps, animate, step + 1)
    
    animate(0)

def detect_object():
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Calculate the coordinates for the middle 300x300 area
    x1 = (screen_width - 300) // 2
    y1 = (screen_height - 300) // 2
    x2 = x1 + 300
    y2 = y1 + 300
    
    # Capture only the middle 300x300 area of the screen
    screenshot = np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2)))
    
    # Perform detection
    results = model(screenshot)
    
    detections = []
    
    # Process results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x, y, w, h = box.xywh[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            
            # Calculate center of the detected object
            center_x = int(x1 + x)
            center_y = int(y1 + y)
            
            # Get class name
            class_name = model.names[int(class_id)]
            
            detections.append((center_x, center_y, class_name, confidence))

    return detections if detections else None

if __name__ == "__main__":
    root, canvas = create_canvas()
    try:
        while True:
            result = detect_object()
            
            if result:
                for detection in result:
                    x, y, class_name, confidence = detection
                    # Calculate relative position within the 300x300 area
                    rel_x = x - ((root.winfo_screenwidth() - 300) // 2) 
                    rel_y = y - ((root.winfo_screenheight() - 300) // 2) 
                    # Clear previous detections
                    canvas.delete("detection")
                    # Draw a red dot at the detected position
                    canvas.create_oval(rel_x-5, rel_y-5, rel_x+5, rel_y+5, fill="red", outline="red", tags="detection")
                    print(f"Object detected at: ({x}, {y}), Class: {class_name}, Confidence: {confidence:.2f}")
            
            root.update()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if 