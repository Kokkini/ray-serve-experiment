# File name: serve_quickstart_composed.py
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handle import RayServeHandle

import tensorflow as tf 
import numpy as np
from PIL import Image
from cloudpathlib import CloudPath
import time


@serve.deployment()
class Diseases:
    def __init__(self):

        # Download model from S3
        cp = CloudPath("s3://belletorus-triton-test-repo/model_repository/Diseases13_1K_16Jan23_EffB4_mixfl16_batch40_c20_fold4_363_712/")
        cp.download_to("diseases13_1")

        # Load model
        self.model = tf.keras.models.load_model("diseases13_1/1/model.savedmodel")
        # print(self.model.summary())
        print("Available GPUs: ", tf.config.list_physical_devices('GPU'))
        print("Diseases init")

    def predict(self, img: np.ndarray):
        start = time.time()
        model_output = self.model.predict({"input_3": img})
        print(f"Prediction time: {time.time() - start}", flush=True)
        return model_output

    async def __call__(self, http_request: Request):
        # read image file from http_request and convert to numpy array
        # image is sent in the form of --form 'image=@"/path/to/image.png"'

        request_form = await http_request.form()

        # print("form items:", len(request_form.items()), flush=True)
        # for field_name, image_data in request_form.items():
        #     print(f"{field_name}: {image_data}", flush=True)

        image = request_form['image']
        rep = int(request_form['rep'])
        print(f"image:{image.file}", flush=True)
        image = Image.open(image.file)
        image = image.resize((512, 512))

        # preprocess image
        image = np.array(image)
        print(image.shape, flush=True)
        image = image / 255.0
        images = np.array([image for i in range(rep)])
        print(f"batch shape: {images.shape}", flush=True)
        # image = np.expand_dims(image, 0)

        # predict
        prediction = self.predict(images)
        return prediction


app = Diseases.bind()