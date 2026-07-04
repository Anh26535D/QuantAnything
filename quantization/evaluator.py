import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from ultralytics import YOLO

# Add parent directory to path so we can import onnx2keras and quantization
sys.path.append("e:/phenikaa/quant_anything")

from onnx2keras.onnx2keras import onnx_to_keras
from quantization.quant_container import QuantContainer

def sanitize_onnx_model(model):
    """
    Sanitizes ONNX model tensor names by replacing '/' and ':' with '_' to comply with Keras 3.0.
    """
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

def load_dataset(dataset_dir, batch_size=16, limit_per_class=2, limit_classes=5):
    """
    Loads images from the structured dataset directory: dataset_dir/class-name/images
    If directory is missing or empty, yields dummy batches of size batch_size.
    To prevent RAM OOM, we support limit_per_class (default 2) and limit_classes (default 5).
    """
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found. Using dummy inputs for validation.")
        # Yield 3 dummy batches
        for _ in range(3):
            yield np.random.uniform(0.0, 1.0, (batch_size, 3, 224, 224)).astype(np.float32), None
        return
        
    image_paths = []
    labels = []
    class_names = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if limit_classes is not None:
        class_names = class_names[:limit_classes]
        
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        count = 0
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(class_path, fname))
                labels.append(class_to_idx[class_name])
                count += 1
                if count >= limit_per_class:
                    break
                
    if not image_paths:
        print(f"No valid images found in '{dataset_dir}'. Using dummy inputs for validation.")
        for _ in range(3):
            yield np.random.uniform(0.0, 1.0, (batch_size, 3, 224, 224)).astype(np.float32), None
        return
        
    print(f"Subsampled dataset loaded: {len(image_paths)} images across {len(class_names)} classes (limit={limit_per_class} per class).")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        batch_imgs = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((224, 224))
                img_arr = np.array(img).astype(np.float32) / 255.0
                # HWC to CHW
                img_arr = np.transpose(img_arr, (2, 0, 1))
                batch_imgs.append(img_arr)
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
                
        if batch_imgs:
            yield np.array(batch_imgs), np.array(batch_labels)

def run_evaluation(dataset_dir="dataset", quant_type="int8"):
    print("\n==================================================")
    print(f"Starting YOLOv8-cls Quantization Assessment ({quant_type})")
    print("==================================================")
    
    # 1. Download YOLOv8-cls nano
    print("\n--- Step 1: Loading model from Ultralytics ---")
    model = YOLO("yolov8n-cls.pt")
    
    # 2. Export to ONNX
    print("\n--- Step 2: Exporting model to ONNX ---")
    onnx_path = model.export(format="onnx", imgsz=224)
    print(f"ONNX model saved at: {onnx_path}")
    
    # 3. Load and sanitize ONNX model
    print("\n--- Step 3: Sanitizing ONNX for Keras 3 ---")
    onnx_model = onnx.load(onnx_path)
    sanitized_onnx_model = sanitize_onnx_model(onnx_model)
    
    # 4. Convert ONNX to Keras
    print("\n--- Step 4: Converting ONNX to Keras ---")
    keras_model = onnx_to_keras(
        sanitized_onnx_model,
        input_names=['images'],
        input_shapes=[[3, 224, 224]],
        verbose=False,
        change_ordering=False
    )
    print("Conversion successful.")
    
def evaluate_model(ort_session, keras_model, quant_graph, dataset_dir, batch_size=1):
    """
    Evaluates ONNX, Keras float, and Quantized models on the dataset,
    printing comparison logs and returning comparison metrics.
    """
    dataset_loader = load_dataset(dataset_dir, batch_size=batch_size)
    
    onnx_keras_errors = []
    keras_quant_errors = []
    kl_divergences = []
    
    y_true_list = []
    preds_onnx_list = []
    preds_keras_list = []
    preds_quant_list = []
    
    total_samples = 0
    has_labels = False
    
    from tqdm import tqdm
    
    progress_bar = tqdm(dataset_loader, desc="Evaluating batches", unit="batch")
    for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
        if y_batch is not None:
            has_labels = True
            y_true_list.extend(y_batch)
            
        # ONNX Inference
        ort_inputs = {ort_session.get_inputs()[0].name: x_batch}
        ort_outputs = ort_session.run(None, ort_inputs)
        out_onnx = ort_outputs[0]
        
        # Keras Inference
        out_keras = keras_model(tf.convert_to_tensor(x_batch)).numpy()
        
        # Quantized Inference
        out_quant = quant_graph.quantized_infer(x_batch)
        
        # Check alignment errors
        diff_onnx_keras = np.max(np.abs(out_onnx - out_keras))
        diff_keras_quant = np.max(np.abs(out_keras - out_quant))
        
        onnx_keras_errors.append(diff_onnx_keras)
        keras_quant_errors.append(diff_keras_quant)
        
        # Compute KL-Divergence KL(Keras || Quantized)
        eps = 1e-15
        p = np.clip(out_keras, eps, 1.0)
        q = np.clip(out_quant, eps, 1.0)
        kl = np.sum(p * np.log(p / q), axis=-1)
        kl_divergences.extend(kl)
        
        # Check classification accuracy
        batch_size_actual = x_batch.shape[0]
        total_samples += batch_size_actual
        
        # Argmax predictions
        preds_onnx = np.argmax(out_onnx, axis=1)
        preds_keras = np.argmax(out_keras, axis=1)
        preds_quant = np.argmax(out_quant, axis=1)
        
        preds_onnx_list.extend(preds_onnx)
        preds_keras_list.extend(preds_keras)
        preds_quant_list.extend(preds_quant)
            
        progress_bar.write(f"Batch {batch_idx + 1}: Max ONNX vs Keras diff = {diff_onnx_keras:.6f} | Max Keras vs Quantized diff = {diff_keras_quant:.6f} | KL-Div = {np.mean(kl):.6f}")
        
    print("\n================== Evaluation Results ==================")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Average ONNX vs Keras Conversion Error (Max Abs Diff): {np.mean(onnx_keras_errors):.6f}")
    print(f"Average Keras vs Quantized Quantization Error (Max Abs Diff): {np.mean(keras_quant_errors):.6f}")
    print(f"Average KL-Divergence KL(Keras || Quantized): {np.mean(kl_divergences):.6f}")
    
    if has_labels:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        # Compute metrics for ONNX
        acc_onnx = accuracy_score(y_true_list, preds_onnx_list)
        p_onnx, r_onnx, f_onnx, _ = precision_recall_fscore_support(y_true_list, preds_onnx_list, average='macro', zero_division=0)
        
        # Compute metrics for Keras
        acc_keras = accuracy_score(y_true_list, preds_keras_list)
        p_keras, r_keras, f_keras, _ = precision_recall_fscore_support(y_true_list, preds_keras_list, average='macro', zero_division=0)
        
        # Compute metrics for Quantized
        acc_quant = accuracy_score(y_true_list, preds_quant_list)
        p_quant, r_quant, f_quant, _ = precision_recall_fscore_support(y_true_list, preds_quant_list, average='macro', zero_division=0)
        
        kl_onnx = 0.0
        kl_keras = 0.0
        kl_quant = np.mean(kl_divergences)
        
        print("\nClassification Metrics Comparison (Macro-averaged):")
        print(f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'KL-Div':<10}")
        print("-" * 78)
        print(f"{'ONNX':<15} | {acc_onnx*100:<9.2f}% | {p_onnx*100:<9.2f}% | {r_onnx*100:<9.2f}% | {f_onnx*100:<9.2f}% | {kl_onnx:<10.6f}")
        print(f"{'Keras':<15} | {acc_keras*100:<9.2f}% | {p_keras*100:<9.2f}% | {r_keras*100:<9.2f}% | {f_keras*100:<9.2f}% | {kl_keras:<10.6f}")
        print(f"{f'Quantized ({quant_graph.quant_type.upper()})':<15} | {acc_quant*100:<9.2f}% | {p_quant*100:<9.2f}% | {r_quant*100:<9.2f}% | {f_quant*100:<9.2f}% | {kl_quant:<10.6f}")
    else:
        print("Note: Classification metrics (Accuracy, Precision, Recall, F1) were not calculated because labels were not available (dummy input mode).")
    print("========================================================")

def run_evaluation(dataset_dir="dataset", quant_type="int8"):
    print("\n==================================================")
    print(f"Starting YOLOv8-cls Quantization Assessment ({quant_type})")
    print("==================================================")
    
    # 1. Download YOLOv8-cls nano
    print("\n--- Step 1: Loading model from Ultralytics ---")
    model = YOLO("yolov8n-cls.pt")
    
    # 2. Export to ONNX
    print("\n--- Step 2: Exporting model to ONNX ---")
    onnx_path = model.export(format="onnx", imgsz=224)
    print(f"ONNX model saved at: {onnx_path}")
    
    # 3. Load and sanitize ONNX model
    print("\n--- Step 3: Sanitizing ONNX for Keras 3 ---")
    onnx_model = onnx.load(onnx_path)
    sanitized_onnx_model = sanitize_onnx_model(onnx_model)
    
    # 4. Convert ONNX to Keras
    print("\n--- Step 4: Converting ONNX to Keras ---")
    keras_model = onnx_to_keras(
        sanitized_onnx_model,
        input_names=['images'],
        input_shapes=[[3, 224, 224]],
        verbose=False,
        change_ordering=False
    )
    print("Conversion successful.")
    
    # 5. Initialize ONNX runtime session
    print("\n--- Step 5: Initializing ONNX Runtime Session ---")
    ort_session = ort.InferenceSession(onnx_path)
    
    # 6. Initialize QuantContainer and run calibration
    print("\n--- Step 6: Initializing QuantContainer and Calibrating ---")
    quant_graph = QuantContainer(keras_model, quant_type=quant_type)
    
    if os.path.exists(dataset_dir):
        print("Loading real dataset images for calibration...")
        # Load a small batch slice for calibration
        calib_loader = load_dataset(dataset_dir, batch_size=4, limit_per_class=2, limit_classes=5)
        calib_data = []
        for x_batch, _ in calib_loader:
            calib_data.append(x_batch)
    else:
        print("Dataset directory not found. Calibrating with dummy random inputs...")
        calib_data = [np.random.uniform(0.0, 1.0, (8, 3, 224, 224)).astype(np.float32) for _ in range(3)]
        
    quant_graph.calibrate(calib_data)
    
    # 7. Evaluate
    print("\n--- Step 7: Evaluating Models ---")
    evaluate_model(ort_session, keras_model, quant_graph, dataset_dir, batch_size=1)

if __name__ == "__main__":
    # Allow passing custom dataset folder and quantization type
    dataset_folder = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    quantization_type = sys.argv[2] if len(sys.argv) > 2 else "int8"
    run_evaluation(dataset_folder, quantization_type)
