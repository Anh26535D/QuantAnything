import os
import onnx
from ultralytics import YOLO
from onnx2keras.onnx2keras import onnx_to_keras

def sanitize_onnx_model(model):
    def rename(name):
        return name.replace('/', '_').replace(':', '_')
        
    for inp in model.graph.input:
        inp.name = rename(inp.name)
    for out in model.graph.output:
        out.name = rename(out.name)
    for init in model.graph.initializer:
        init.name = rename(init.name)
    for node in model.graph.node:
        if node.name:
            node.name = rename(node.name)
        inputs = [rename(i) for i in node.input]
        del node.input[:]
        node.input.extend(inputs)
        
        outputs = [rename(o) for o in node.output]
        del node.output[:]
        node.output.extend(outputs)
    for vi in model.graph.value_info:
        vi.name = rename(vi.name)
    return model

def main():
    print("--- Step 1: Loading YOLOv8n-cls model from Ultralytics ---")
    model = YOLO("yolov8n-cls.pt")
    
    print("\n--- Step 2: Exporting to ONNX ---")
    onnx_path = model.export(format="onnx", imgsz=224)
    print(f"ONNX model saved at: {onnx_path}")
    
    print("\n--- Step 3: Loading ONNX model ---")
    onnx_model = onnx.load(onnx_path)
    
    print("\n--- Step 3.5: Sanitizing ONNX model names for Keras 3.0 ---")
    onnx_model = sanitize_onnx_model(onnx_model)
    
    print("\n--- Step 4: Converting ONNX to Keras ---")
    try:
        keras_model = onnx_to_keras(
            onnx_model,
            input_names=['images'],
            input_shapes=[[3, 224, 224]],
            verbose=True,
            change_ordering=False
        )
        print("\n--- Success! Keras model converted successfully ---")
        keras_model.summary()
        
        # Save keras model structure
        keras_model.save("yolov8n-cls.keras")
        print("Keras model saved as yolov8n-cls.keras")
    except Exception as e:
        print("\n--- Error during ONNX to Keras conversion ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
