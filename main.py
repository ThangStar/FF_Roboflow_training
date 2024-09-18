import pyautogui
from PIL import ImageGrab
import tkinter as tk
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model globally
model = YOLO('E:/ws/FFTool/runs/detect/train5/weights/best.pt')  # or your specific model path

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
    
    # Perform detection with a higher confidence threshold and NMS
    results = model(screenshot, conf=0.5, iou=0.6)  # Adjust these values as needed
    
    detections = []
    
    # Process results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            
            # Lấy tên lớp
            class_name = model.names[int(class_id)]
            
            detections.append((int(x1), int(y1), int(x2), int(y2), class_name, confidence))

    return detections if detections else None

if __name__ == "__main__":
    root, canvas = create_canvas()
    try:
        while True:
            result = detect_object()
            
            if result:
                # Xóa các phát hiện trước đó
                canvas.delete("detection")
                for detection in result:
                    x1, y1, x2, y2, class_name, confidence = detection
                    # Vẽ hình chữ nhật bao quanh đối tượng
                    canvas.create_rectangle(x1, y1, x2, y2, 
                                            outline="red", width=2, tags="detection")
                    
                    # Hiển thị tên lớp và độ tin cậy
                    canvas.create_text((x1 + x2) / 2, y1 - 10,
                                       text=f"{class_name}: {confidence:.2f}",
                                       fill="red", font=("Arial", 8), tags="detection")
                    
                    print(f"Object detected at: ({x1}, {y1}, {x2}, {y2}), Class: {class_name}, Confidence: {confidence:.2f}")
            
            root.update()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if root:
            root.destroy()

