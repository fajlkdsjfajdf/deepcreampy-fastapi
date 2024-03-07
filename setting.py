import onnxruntime as ort
import tensorflow as tf

def ort_gpu_check():
    print(f"onnxruntime使用设备: {ort.get_device()}")

def tf_gpu_check():
    print(f"tensorflow使用GPU: {tf.test.is_gpu_available()}")

def check():
    ort_gpu_check()
    tf_gpu_check()


if __name__ == "__main__":
    check()