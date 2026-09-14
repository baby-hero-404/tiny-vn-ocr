import os
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from onnxruntime.quantization import quantize_dynamic, QuantType
import warnings

warnings.filterwarnings("ignore")

def quantize():
    print("Loading original PyTorch model...")
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['cnn']['pretrained'] = False
    config['predictor']['beamsearch'] = False
    config['device'] = 'cpu'
    
    # We will use the weights from /tmp/weights.pth if available
    weights = '/tmp/vgg_seq2seq.pth'
    if not os.path.exists(weights):
        print("Downloading weights...")
        from vietocr.tool.utils import download_weights
        weights_path = download_weights(config['weights'])
        os.system(f"cp {weights_path} {weights}")
    else:
        config['weights'] = weights

    p = Predictor(config)
    model = p.model
    model.eval()

    # 1. Export CNN to ONNX
    cnn = model.cnn
    cnn_onnx_path = 'resources/models/vgg_cnn.onnx'
    cnn_int8_path = 'resources/models/vgg_cnn_int8.onnx'
    
    os.makedirs('resources/models', exist_ok=True)
    
    print("Exporting CNN to ONNX...")
    dummy_input = torch.randn(1, 3, 32, 256)
    try:
        torch.onnx.utils.export(
            cnn, dummy_input, cnn_onnx_path, opset_version=14,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 3: 'width'}, 'output': {0: 'batch_size', 1: 'width'}},
            export_params=True, do_constant_folding=True
        )
    except Exception as e:
        print(f"Warning: {e}")

    print("Quantizing CNN ONNX to INT8...")
    quantize_dynamic(cnn_onnx_path, cnn_int8_path, weight_type=QuantType.QUInt8)
    
    print(f"CNN ONNX size: {os.path.getsize(cnn_onnx_path)/1e6:.2f} MB")
    print(f"CNN INT8 size: {os.path.getsize(cnn_int8_path)/1e6:.2f} MB")

    # 2. Extract and Quantize Transformer (Seq2Seq)
    print("Quantizing Transformer (PyTorch)...")
    transformer = model.transformer
    transformer_int8 = torch.quantization.quantize_dynamic(
        transformer, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, dtype=torch.qint8
    )
    
    transformer_path = 'resources/models/transformer_int8.pth'
    torch.save(transformer_int8.state_dict(), transformer_path)
    print(f"Transformer INT8 size: {os.path.getsize(transformer_path)/1e6:.2f} MB")
    
    print("Quantization complete!")

if __name__ == "__main__":
    quantize()
