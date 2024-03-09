from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive image file from the client
    file = request.files['image']
    
    # Read the image file
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur for noise reduction
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    # # Apply image denoising
    # img_denoised = cv2.fastNlMeansDenoising(img_blur, None, h=10, searchWindowSize=21, templateWindowSize=7)
    
    # # Apply image sharpening
    # kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # img_sharpened = cv2.filter2D(img_denoised, -1, kernel_sharpening)
    
    # Apply adaptive thresholding for binarization
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    
    # Convert processed image to bytes
    _, processed_img = cv2.imencode('.jpg', thresh)
    processed_img_bytes = processed_img.tobytes()
    
    # Send the processed image data back to the client
    return processed_img_bytes, 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
