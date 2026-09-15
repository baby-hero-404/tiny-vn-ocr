import os
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from vietocr.tool.config import Cfg
from vietocr.model.vocab import Vocab
from vietocr.model.seqmodel.seq2seq import Seq2Seq, Encoder, Decoder, Attention

class ONNXPredictor:
    def __init__(self, config_name='vgg_seq2seq'):
        self.config = Cfg.load_config_from_name(config_name)
        self.config['device'] = 'cpu'
        
        self.vocab = Vocab(self.config['vocab'])
        
        # Load ONNX CNN
        onnx_path = 'resources/models/vgg_cnn_int8.onnx'
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Run scripts/quantize_vietocr.py first.")
        
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.cnn_session = ort.InferenceSession(onnx_path, options, providers=['CPUExecutionProvider'])
        
        # Optimize PyTorch CPU inference threads
        torch.set_num_threads(2)
        
        # Rebuild Transformer Architecture
        emb_dim = self.config['transformer']['decoder_embedded']
        dec_hid_dim = self.config['transformer']['decoder_hidden']
        enc_hid_dim = self.config['transformer']['encoder_hidden']
        dropout = self.config['transformer']['dropout']
        
        self.transformer = Seq2Seq(
            len(self.vocab),
            enc_hid_dim,
            dec_hid_dim,
            self.config['cnn']['hidden'],
            emb_dim,
            dropout
        )
        
        # Load Quantized Weights
        transformer_path = 'resources/models/transformer_int8.pth'
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Transformer model not found at {transformer_path}.")
            
        # We need to construct a dynamically quantized version of our transformer to load the state dict
        self.transformer = torch.quantization.quantize_dynamic(
            self.transformer, {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, dtype=torch.qint8
        )
        self.transformer.load_state_dict(torch.load(transformer_path, map_location='cpu', weights_only=False))
        self.transformer.eval()
        
    def preprocess_image(self, img: Image.Image) -> np.ndarray:
        if not hasattr(Image, 'ANTIALIAS'):
            Image.ANTIALIAS = Image.LANCZOS
        from vietocr.tool.translate import process_input
        # process_input returns torch.Tensor. We need to convert it to numpy for ONNX
        img_tensor = process_input(img, self.config['dataset']['image_height'], 
                                 self.config['dataset']['image_min_width'], 
                                 self.config['dataset']['image_max_width'])
        return img_tensor.numpy()
        
    def predict(self, img: Image.Image) -> str:
        # 1. Preprocess
        img_np = self.preprocess_image(img)
        
        # 2. CNN Feature Extraction (ONNX)
        ort_inputs = {self.cnn_session.get_inputs()[0].name: img_np}
        cnn_out = self.cnn_session.run(None, ort_inputs)[0] # Shape: [batch, hidden, height, width]
        
        # Post-process CNN out as in Vgg.forward()
        # cnn_out is numpy array. Convert to torch tensor
        conv = torch.from_numpy(cnn_out)
        
        # conv = conv.transpose(-1, -2)
        conv = conv.transpose(-1, -2)
        # conv = conv.flatten(2)
        conv = conv.flatten(2)
        # conv = conv.permute(-1, 0, 1)
        src = conv.permute(-1, 0, 1)
        
        # 3. Seq2Seq Decoder
        with torch.no_grad():
            outputs, hidden = self.transformer.encoder(src)
            
            # Autoregressive decoding (greedy)
            batch_size = src.shape[1]
            trg_indexes = [[self.vocab.go] for _ in range(batch_size)]
            
            max_seq_len = 64
            
            for i in range(max_seq_len):
                trg_tensor = torch.LongTensor([seq[-1] for seq in trg_indexes])
                
                output, hidden, _ = self.transformer.decoder(trg_tensor, hidden, outputs)
                top1 = output.argmax(1).tolist()
                
                all_ended = True
                for j in range(batch_size):
                    trg_indexes[j].append(top1[j])
                    if top1[j] != self.vocab.eos:
                        all_ended = False
                        
                if all_ended:
                    break
                    
            # Check for EOS and decode
            final_str = []
            for seq in trg_indexes:
                decoded = []
                for token in seq[1:]: # Skip SOS
                    if token == self.vocab.eos:
                        break
                    decoded.append(token)
                final_str.append(self.vocab.decode(decoded))
                
        return final_str[0] if len(final_str) == 1 else final_str
