Mô hình được train trên Roboflow
https://universe.roboflow.com/nng-vn-thng/ff_detection/dataset/1

### 🤝 Chức năng chính

- [ ] Nhận diện nhân vật trong game freefire

### 🚀 Cách sử dụng
#### cài đặt
```bash
pip install keyring==8.7.0  
```
```bash
pip install --upgrade ultralytics
python -m venv yolo-env
yolo-env\Scripts\activate  
pip install yolo==0.3.1 keyring==8.7.0
pip install yolo --ignore-installed keyring
pip install torch torchvision   
```
#### Chỉnh sửa file data.yaml
```bash
train: FULL_PATH_TO_data.yaml to train/images
val: FULL_PATH_TO_data.yaml to valid/images
test: FULL_PATH_TO_data.yaml to test/images
```
#### Trainning process(epochs: số lần train, imgsz: kích thước ảnh)
```bash
yolo task=detect mode=train model=yolov8s.pt data="<FULL_PATH_TO_data.yaml>" epochs=100 imgsz=640 plots=True
```