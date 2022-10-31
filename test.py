import requests
import cv2
from PIL import Image
from io import BytesIO
import base64
import numpy as np

url = "http://172.29.13.24:35515/process/vg_api"
# font
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 1

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

def get_base64(file_name):
    img = Image.open(file_name) # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    return base64_str

def show_demo(file_path, text, output_path):
    image = Image.open(file_path)
    base64_str = get_base64(file_path)
    payload = "{\r\n    \"index\": \"a;lkalsd\",\r\n    \"data\": {\r\n        \"img\": \"" + base64_str + "\",\r\n        \"text\": \"" + text + "\"\r\n    }\r\n} "
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    response = response.json()
    start_point = response['data']['start_point']
    end_point = response['data']['end_point']
    print(start_point, end_point)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        img,
        (start_point[0], start_point[1]),
        (end_point[0], end_point[1]),
        (0, 255, 0),
        3
    )
    
    # Using cv2.putText() method
    img = cv2.putText(img, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(f'{output_path}/demo.jpg', img)

if __name__ == '__main__':
    
    show_demo('/TMTAI/Computer_Vision/backup/AttributeQA/film/black bow shirt.jpg',
              'black bow shirt',
              '/TMTAI/Computer_Vision/backup/AttributeQA/output_dataset_VG')