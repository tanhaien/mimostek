import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import shutil
from fastapi.responses import JSONResponse
import jwt

# Thông tin payload của JWT
payload = {
    "user_id": 123,
    "username": "exampleuser",
    "zone/area": "examplearea"
}

disease_id = ['BS6', 'EB1', 'FW9', 'GM11', 'HT12', 'LB2', 'MV7', 'PM4', 'SL5', 'TS3',  'VW10', 'YL8']

# # Mã hóa JWT
# secret_key = "your_secret_key"
# token = jwt.encode(payload, secret_key, algorithm="HS256")

app = FastAPI()
model_path = "models/model.tflite"

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Khai báo thư mục tĩnh để lưu trữ ảnh đã upload
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    # headers = request.headers
    # try:
    #     # Giải mã JWT
    #     decoded_payload = jwt.decode(token, secret_key, algorithms=["HS256"])
    # except jwt.ExpiredSignatureError:
    #     print("Token đã hết hạn")
    # except jwt.InvalidTokenError:
    #     print("Token không hợp lệ")
    # userID = decoded_payload #headers.get("userID")
        
     # Validate input
    if not file.filename.endswith((".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")):
        raise HTTPException(status_code=400, detail="Invalid file format")

    if file.size > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=413, detail="File size exceeds limit")

    # Lưu file vào thư mục 'static'
    with open(f"static/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Đường dẫn file ảnh
    image_path = f"static/{file.filename}"
    
    # Xử lý ảnh với model TFLite
    input_shape = input_details[0]['shape']
    try:
        img = Image.open(image_path).resize((input_shape[1], input_shape[2]))
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.uint8)
        img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

    except Exception as x:
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
    
    # Kết quả xử lý
    result = (np.array(output_data[0]))
    
    # Trong phương thức post upload
    # Tính argmax của biến result
    argmax_result = np.argmax(result)

    # Chuyển kết quả argmax thành int
    argmax_result_int = int(argmax_result)
    
    return JSONResponse(status_code=200, content={"result": disease_id[argmax_result_int], 
                                                  "image_path": image_path,
                                                  "user_id": payload["user_id"],
                                                  "username": payload["username"],
                                                  "zone_area": payload["zone/area"]})
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run('app2:app', host="0.0.0.0", port=1712, reload=True)
