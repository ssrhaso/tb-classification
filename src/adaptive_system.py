""" ADAPTIVE CNN-ViT MODULE, CONFIGURABLE, ROUTING DECISIONS TO BALANCE LATENCY & ACCURACY """

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models 
import time
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets.tb_dataset import TBDataset

# CONFIGURATION
CONFIG = {
    'cnn_checkpoint': './results/checkpoints/cnn_best_v3.pt',
    'vit_checkpoint': './results/checkpoints/vit_best_v3',
    'cnn_architecture': 'resnet18', # options: 'resnet18', 'resnet50'
    'vit_architecture': 'vit_b_16', 
    'threshold': 0.90,  # CONFIDENCE THRESHOLD FOR ROUTING
    'resolution': 224,  # IMAGE SIZE
    'test_csv': 'splits/test_split.csv',
    'data_dir': './src/datasets/tb_dataset_crops',
    'num_samples': None,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 2,
}






class AdaptiveClassifier:
    def __init__(
        self,
        cnn_checkpoint_path: str,
        vit_checkpoint_path: str,
        cnn_architecture: str = 'resnet18',
        vit_architecture: str = 'vit_b_16',
        num_classes: int = 2,
        
        threshold: float = 0.90,
        resolution: int = 224,
        device: str = 'cpu',
        input_mean: tuple = None,
        input_std: tuple = None,
        transform = None,
    ):
        
        self.device = device
        self.threshold = threshold
        self.resolution = resolution
        self.num_classes = num_classes
        
        
        # LOAD CNN MODEL
        
        # DEFAULT ImageNet NORMALIZATION VALUES IF NONE PROVIDED
        self.input_mean = input_mean if input_mean is not None else (0.485, 0.456, 0.406)
        self.input_std = input_std if input_std is not None else (0.229, 0.224, 0.225)        
        
        # LOAD CNN
        print("\nLOADING CNN model...")
        self.cnn = self._load_cnn(cnn_architecture, cnn_checkpoint_path, num_classes)
        self.cnn.to(self.device)
        self.cnn.eval()
        print("CNN model loaded")
        
        
        # LOAD VIT MODEL
        print("\nLOADING ViT model...")
        self.vit = self._load_vit(vit_architecture, vit_checkpoint_path, num_classes)
        self.vit.to(self.device)
        self.vit.eval()
        print("ViT model loaded")
        


    
        # SETUP PREPROCESSING
        # CNN (32x32):

        if transform is not None:
            self.transform = transform
        
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize( 
                    mean = self.input_mean,
                    std = self.input_std)
        ])
        
        print("TRANSFORMS for CNN and ViT set up")
        print("-"*60)
        print(" WARMING UP MODELS...")
        self._warmup() # WARMUP TO REDUCE INITIAL LATENCY (CACHING)
        print("-"*60)
        print(" MODELS WARMED UP")
        
    
    # WARMUP METHOD (TO REDUCE INITIAL LATENCY)
    def _warmup(
        self,
        num_runs: int = 5,
    ):
        """ WARMUP MODELS TO REDUCE INITIAL LATENCY , IMPROVE CONSISTENCY IN INFERENCE LABELLING """
        dummy_input = torch.randn(1, 3, self.resolution, self.resolution).to(self.device)
        
        with torch.no_grad(): # NO GRADIENTS NEEDED, FASTER
           for _ in range(num_runs):
                _ = self.cnn(dummy_input)
                _ = self.vit(dummy_input)
        
    
    
    
    
    def predict(
        self,
        image: Image.Image
    ) -> tuple:
        """ CNN + VIT ADAPTIVE INFERENCE , RETURNS PREDICTION, ROUTED MODEL, CONFIDENCE, METADATA """
        
        if isinstance(image, torch.Tensor):
            tensor_img = image 
        
        else:
            tensor_img = transforms.ToTensor()(image)
            
        # METADATA FOR ANALYSIS
        stats = {
            'cnn_latency' : 0.0,
            'vit_latency' : 0.0,
            'total_latency' : 0.0,
            'routed_to' : None,
            'cnn_confidence' : 0.0,
            'cnn_prediction' : None,
        }
        
        # CNN INFERENCE (ALWAYS RUNNING)
    
        cnn_pred, cnn_confidence, cnn_latency = self._predict_cnn(tensor_img)
        stats['cnn_latency'] = cnn_latency
        stats['cnn_confidence'] = cnn_confidence
        stats['cnn_prediction'] = cnn_pred
        
        
        # ROUTING DECISION
        
        if cnn_confidence >= self.threshold:
            
            # HIGH CONFIDENCE
            stats['routed_to'] = 'CNN'
            stats['total_latency'] = cnn_latency
            return (
                cnn_pred,
                'CNN',
                cnn_confidence,
                cnn_latency,
                stats
            )
            
        else:
            # LOW CONFIDENCE - ROUTE TO VIT
        
            vit_pred, vit_confidence, vit_latency = self._predict_vit(tensor_img)
            
            stats['vit_latency'] = vit_latency
            stats['routed_to'] = 'ViT'
            stats['total_latency'] = cnn_latency + vit_latency
            
            return (
                vit_pred,
                'ViT',
                vit_confidence,
                stats['total_latency'],
                stats
            )
            
            
            
       
    # HELPER FUNCTIONS:
         
    # MODEL LOADING HELPERS
    # CNN MODEL LOADER     
    def _load_cnn(
        self,
        architecture: str,
        checkpoint_path: str,
        num_classes: int
    ):
        """ LOAD CNN MODEL FROM CHECKPOINT """
        # INITIALIZE MODEL
        if architecture == 'resnet18':
            model = models.resnet18(weights = None)
            model.fc = nn.Linear(in_features = 512, out_features = num_classes)
        
        elif architecture == 'resnet50':
            model = models.resnet50(weights = None)
            model.fc = nn.Linear(in_features = 2048, out_features = num_classes)
        
        else:
            raise ValueError(f"Unsupported CNN architecture: {architecture}")
        
        # LOAD CHECKPOINT
        try:
            checkpoint = torch.load(checkpoint_path, map_location = self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
    
        except FileNotFoundError:
            raise FileNotFoundError(f"CNN checkpoint not found at: {checkpoint_path}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading CNN checkpoint: {e}")
        return model
    
    # VIT MODEL LOADER
    def _load_vit(
        self,
        architecture: str,
        checkpoint_path: str,
        num_classes: int
    ):
        """ LOAD VIT MODEL FROM CHECKPOINT """
        # INITIALIZE MODEL
        
        if os.path.isdir(checkpoint_path):
            # HUGGING FACE FORMAT
            from transformers import ViTForImageClassification, ViTConfig
            try:
                model = ViTForImageClassification.from_pretrained(
                    checkpoint_path,
                    num_labels = num_classes,
                    ignore_mismatched_sizes= True
                )
                return model
            except Exception as e:
                raise RuntimeError(f"Error loading ViT checkpoint from Hugging Face format: {e}")
        
        
        else:
            
            if architecture == 'vit_b_16':
                model = models.vit_b_16(weights = None)
            
            elif architecture == 'vit_b_32':
                model = models.vit_b_32(weights = None)
            else:
                raise ValueError(f"Unsupported ViT architecture: {architecture}")
            
            # MODIFY CLASSIFICATION HEAD
            model.heads = nn.Linear(model.hidden_dim, num_classes)
            
            # LOAD CHECKPOINT
            try:
                checkpoint = torch.load(checkpoint_path, map_location = self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
        
            except FileNotFoundError:
                raise FileNotFoundError(f"ViT checkpoint not found at: {checkpoint_path}")
            except RuntimeError as e:
                raise RuntimeError(f"Error loading ViT checkpoint: {e}")
            return model
        
    
            
            
    # PREDICTION HELPERS
    # CNN PREDICTION HELPER
    def _predict_cnn(
        self,
        image: Image.Image
    ):
        """ PREDICTION USING CNN MODEL """
        batched = image.unsqueeze(0)                # BATCH DIMENSION [3, 32, 32] -> [1, 3, 32, 32]
        cnn_input = batched.to(self.device)         # MOVE TO CPU/GPU
        
        # INFERENCE
        start = time.perf_counter()                                             # START TIME (INF 1)
        with torch.no_grad():
            # GET CONFIDENCE
            outputs =  self.cnn(cnn_input)  # FORWARD PASS
            probs = F.softmax(outputs, dim = 1) # SOFTMAX PROBABILITIES
            cnn_confidence, cnn_pred = torch.max(probs, dim = 1) # MAX CONFIDENCE & PREDICTION
            
        cnn_latency = (time.perf_counter() - start) * 1000                      # END TIME (INF 2) -> LATENCY MS
        
        return cnn_pred.item(), cnn_confidence.item(), cnn_latency # RETURN PRED, CONFIDENCE, LATENCY
  
    # VIT PREDICTION HELPER
    def _predict_vit(
        self,
        image: Image.Image
    ):
        """ PREDICTION USING VIT MODEL """
        batched = image.unsqueeze(0).to(self.device)           # BATCH DIMENSION [3, 224, 224] -> [1, 3, 224, 224]
    
            
        # INFERENCE
        start = time.perf_counter()                                                 # START TIME (INF 1)
        with torch.no_grad():
            # GET CONFIDENCE
            outputs =  self.vit(batched)    # FORWARD PASS
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            probs = F.softmax(logits, dim = 1) # SOFTMAX PROBABILITIES
            vit_confidence, vit_pred = torch.max(probs, dim = 1) # MAX CONFIDENCE & PREDICTION
            
        vit_latency = (time.perf_counter() - start) * 1000                          # END TIME (INF 2) -> LATENCY MS
            
        return vit_pred.item(), vit_confidence.item(), vit_latency # RETURN PRED, CONFIDENCE, LATENCY
    
    

# ENTRY (TESTING PURPOSES)
if __name__ == "__main__":
    # TEST ADAPTIVE CLASSIFIER
        
    # LOAD TEST DATASET
    
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG['resolution'], CONFIG['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize( 
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225))
    ])
    
    test_dataset = TBDataset(
        csv_file = CONFIG['test_csv'],
        root_dir = CONFIG['data_dir'],
        transform = test_transform, 
    )

    # CREATE ADAPTIVE CLASSIFIER
    adaptive = AdaptiveClassifier(
        cnn_checkpoint_path = CONFIG['cnn_checkpoint'],
        vit_checkpoint_path = CONFIG['vit_checkpoint'],
        cnn_architecture = CONFIG['cnn_architecture'],
        vit_architecture = CONFIG['vit_architecture'],
        threshold = CONFIG['threshold'],
        resolution = CONFIG['resolution'],
        device = CONFIG['device'],
        num_classes = CONFIG['num_classes'],
        transform = test_transform,
    )
    
    num_samples = CONFIG['num_samples'] if CONFIG['num_samples'] is not None else len(test_dataset)
    num_samples = min(num_samples, len(test_dataset))
    
    # TEST ON IMAGES        
    print("-"*70)
    print(f"\nTESTING ADAPTIVE CLASSIFIER")
    print("-"*70)

    # Configuration
    x = num_samples  # Number of images to test (change to 5, 100, or 1000) OR SELECT num_samples FOR FULL TEST SET

    # Tracking variables
    total_latency = 0.0
    cnn_count = 0
    vit_count = 0
    correct = 0
    cnn_latencies = []
    vit_latencies = []

    # Run predictions
    print(f"\nProcessing {x} test images...")
    for i in range(x):
        img, true_label = test_dataset[i]
        pred, model_used, confidence, latency, stats = adaptive.predict(img)

        
        # Track metrics
        total_latency += latency
        if isinstance(true_label, torch.Tensor):
            true_label = true_label.item()
        correct += (pred == true_label)

            
        cnn_count += (stats['routed_to'] == 'CNN')
        vit_count += (stats['routed_to'] == 'ViT')
        cnn_latencies.append(stats['cnn_latency'])
        if stats['routed_to'] == 'ViT':
            vit_latencies.append(stats['vit_latency'])
        
        # Progress indicator (every 100 images)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{x} images...")

    # Calculate metrics
    accuracy = 100 * correct / x
    avg_latency = total_latency / x
    avg_cnn_latency = sum(cnn_latencies) / len(cnn_latencies)
    avg_vit_latency = sum(vit_latencies) / len(vit_latencies) if vit_latencies else 0

    # Baselines ( ADJUST BASED ON ANALYSIS )
    vit_baseline_latency = 96.13  # ms
    vit_baseline_accuracy = 99.00  # %
    speedup = vit_baseline_latency / avg_latency

    # Print results
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print("="*70)

    print(f"\nOVERALL PERFORMANCE")
    print("-"*70)
    print(f"  Test Images:     {x}")
    print(f"  Correct:         {correct}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    print(f"  Avg Latency:     {avg_latency:.2f} ms")

    print(f"\nROUTING STATISTICS")
    print("-"*70)
    print(f"  CNN Usage:       {cnn_count} images ({100*cnn_count/x:.1f}%)")
    print(f"  ViT Usage:       {vit_count} images ({100*vit_count/x:.1f}%)")

    print(f"\nLATENCY BREAKDOWN")
    print("-"*70)
    print(f"  Avg CNN Latency: {avg_cnn_latency:.2f} ms (always runs)")
    print(f"  Avg ViT Latency: {avg_vit_latency:.2f} ms (when routed)")
    print(f"  Avg Total:       {avg_latency:.2f} ms")

    print(f"\nBASELINE COMPARISON")
    print("-"*70)
    print(f"  ViT-only Accuracy: {vit_baseline_accuracy:.2f}%")
    print(f"  ViT-only Latency:  {vit_baseline_latency:.2f} ms")
    print(f"  Adaptive Accuracy: {accuracy:.2f}%")
    print(f"  Adaptive Latency:  {avg_latency:.2f} ms")

    print(f"\nPERFORMANCE GAINS")
    print("-"*70)
    print(f"  Speedup vs ViT:  {speedup:.2f}x")
    print(f"  Accuracy Retention: {accuracy/vit_baseline_accuracy:.2%}")

    print(f"\n{'='*70}")

        
        

        
        
        

    