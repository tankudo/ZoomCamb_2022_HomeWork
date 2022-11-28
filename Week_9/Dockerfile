FROM svizor42/zoomcamp-dino-dragon-lambda:v2

COPY dino_dragon.tflite .

RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl?raw=true

COPY app.py .

CMD [ "app.handler" ]