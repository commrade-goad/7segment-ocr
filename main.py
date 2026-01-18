import io
import os
import glob
from typing import Optional, Tuple, Union

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# ==========================================
# CRNN Setup (copied/adapted from your code)
# ==========================================

# Get Numeric Only
ALPHABET = "0123456789-."
BLANK = "<BLANK>"
itos = {i: ch for i, ch in enumerate(ALPHABET)}
itos[len(itos)] = BLANK
NUM_CLASSES = len(itos)


# Adapt Image Width Padding
def pad_to_width(img: Image.Image, target_width: int = 192) -> Image.Image:
    w, h = img.size
    if w >= target_width:
        return img.resize((target_width, h))
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left
    return ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=128)


# Pipeline Preprocessing (Grayscale -> Autocontrast -> Resize -> Pad -> Normalize)
transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(ImageOps.autocontrast),
    transforms.Resize(64),
    transforms.Lambda(lambda img: pad_to_width(img, target_width=192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# CRNN Module
class CRNN(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(
            input_size=192 * 16,  # C * H after CNN: C=192, H=16 (given input H=64 -> two pools -> 16)
            hidden_size=192,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=False
        )
        self.head = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        feats = self.cnn(x)  # (B, C, H, W)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 3, 1, 2).reshape(B, W, C * H)  # (B, W, C*H)
        feats = feats.permute(1, 0, 2)  # (W, B, C*H)
        out, _ = self.rnn(feats)
        logits = self.head(out)  # (W, B, num_classes)
        return logits


# Translate model outputs to text via greedy CTC decode
def ctc_decode(logits: torch.Tensor):
    probs = logits.softmax(-1)
    best = probs.argmax(-1)  # (W, B)
    results = []
    for b in range(best.shape[1]):
        seq = best[:, b].tolist()
        dedup = []
        prev = None
        for s in seq:
            if s == NUM_CLASSES - 1:  # blank
                prev = s
                continue
            if s != prev:
                dedup.append(s)
            prev = s
        text = "".join(itos[i] for i in dedup)
        results.append(text)
    return results


def _find_weights_path() -> str:
    # Prefer common names, else pick newest .pt/.pth in current directory
    preferred = ["model.pt", "model.pth", "weights.pt", "weights.pth", "crnn.pt", "crnn.pth"]
    for p in preferred:
        if os.path.exists(p):
            return p
    candidates = glob.glob("*.pt") + glob.glob("*.pth")
    if not candidates:
        raise FileNotFoundError("No .pt or .pth model file found in the current directory.")
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def _normalize_state(state: Union[nn.Module, dict]) -> Tuple[Optional[nn.Module], Optional[dict]]:
    """
    Normalize various checkpoint formats to either:
    - an nn.Module (return as (module, None))
    - a pure state_dict (return as (None, state_dict))

    Supported formats:
      - nn.Module
      - {"state_dict": ...}
      - {"model": <nn.Module>} or {"model": <state_dict>}   <-- your case
      - {"model_state_dict": ...}
      - plain state dict with prefixes like "module." or "model."
    """
    # Direct nn.Module saved
    if isinstance(state, nn.Module):
        return state, None

    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format: expected nn.Module or dict.")

    # Common wrappers
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        model_obj = state["model"]
        if isinstance(model_obj, nn.Module):
            return model_obj, None
        else:
            # In your file, 'model' is the actual state_dict
            state = model_obj

    if not isinstance(state, dict):
        raise RuntimeError("Unsupported checkpoint format inside wrapper; expected a dict state_dict.")

    # Strip prefixes if present (e.g., DDP "module." or Lightning "model.")
    def strip_prefix(d: dict, prefix: str) -> dict:
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}

    if any(k.startswith("model.") for k in state.keys()):
        state = strip_prefix(state, "model.")
    if any(k.startswith("module.") for k in state.keys()):
        state = strip_prefix(state, "module.")

    return None, state


@st.cache_resource(show_spinner=False)
def load_model_from_path(weights_path: str, device: torch.device) -> nn.Module:
    raw = torch.load(weights_path, map_location=device)
    module_obj, state_dict = _normalize_state(raw)

    if module_obj is not None:
        module_obj = module_obj.to(device)
        module_obj.eval()
        return module_obj

    # Load into fresh CRNN
    model = CRNN(num_classes=NUM_CLASSES, in_ch=1).to(device)
    model.eval()

    # Try strict=True first, then fallback to strict=False
    try:
        result = model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)

    # Report load status in the sidebar
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    if missing or unexpected:
        st.sidebar.warning(f"State dict load relaxed.\nMissing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")

    return model


def predict_image(img: Image.Image, model: nn.Module, device: torch.device) -> str:
    tensor = transform_pipeline(img)
    tensor = tensor.unsqueeze(0).to(device)  # (B=1, 1, H, W)
    with torch.no_grad():
        logits = model(tensor)  # (W, B=1, num_classes)
    text = ctc_decode(logits)[0]
    return text


# ==========================================
# Streamlit UI
# ==========================================

name = "CRNN OCR (7Segments digit OCR)"

st.set_page_config(page_title=name, page_icon="🗿", layout="centered")


def home_page():
    """Display the homepage with app description."""
    st.title("🗿 Welcome to 7-Segment LCD OCR")
    st.markdown("""
    ### About This Application
    
    This application uses a **CRNN (Convolutional Recurrent Neural Network)** model to perform 
    Optical Character Recognition (OCR) on 7-segment LCD displays.
    
    #### Features:
    - **Character Recognition**: Recognizes digits (0-9), decimal points (.), and minus signs (-)
    - **Deep Learning Model**: Utilizes a CRNN architecture with CTC decoding
    - **Preprocessing Pipeline**: Automatic grayscale conversion, contrast adjustment, and normalization
    - **Device Support**: Runs on CPU, CUDA (NVIDIA GPU), or MPS (Apple Silicon)
    
    #### How It Works:
    1. **Upload** an image containing a 7-segment LCD display
    2. The image is **preprocessed** (grayscale, autocontrast, resize, padding)
    3. The **CRNN model** processes the image through:
       - Convolutional layers for feature extraction
       - Recurrent layers (GRU) for sequence modeling
       - CTC decoding for text output
    4. The recognized **text is displayed** alongside the original and preprocessed images
    
    #### Getting Started:
    Navigate to the **"Model Interface"** page using the sidebar to start recognizing text from your images!
    
    ---
    
    **Supported Alphabet**: `0123456789-.`
    """)


def model_interface():
    """Display the OCR model interface for image upload and prediction."""
    st.title(name)
    st.caption("Upload an image and run OCR using a CRNN model with greedy CTC decoding. Alphabet: 0-9, '-', '.'")

    with st.sidebar:
        st.header("Model")
        device_choice = st.selectbox("Device", options=["Auto", "CPU", "CUDA", "MPS"], index=0)
        st.markdown("- The app automatically loads a .pt/.pth model from the current directory.")

    # Device selection
    if device_choice == "Auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    elif device_choice == "CUDA":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_choice == "MPS":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load model (cached) from current directory
    try:
        weights_path = _find_weights_path()
        model = load_model_from_path(weights_path, device)
        st.sidebar.success(f"Loaded weights: {weights_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading weights: {e}")
        st.stop()

    st.subheader("Upload Image")
    img_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"])

    if img_file:
        try:
            img = Image.open(img_file).convert("RGB")
        except Exception as e:
            st.error(f"Failed to open image: {e}")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)

        with st.spinner("Running OCR..."):
            try:
                text = predict_image(img, model, device)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        with col2:
            # Show preprocessed image
            pre_img = transform_pipeline.transforms[0](img)  # Grayscale
            pre_img = transform_pipeline.transforms[1](pre_img)  # Autocontrast
            pre_img = transform_pipeline.transforms[2](pre_img)  # Resize
            pre_img = transform_pipeline.transforms[3](pre_img)  # Pad to width
            st.image(pre_img, caption="Preprocessed (H≈64, padded to W≈192)", use_container_width=True, clamp=True)

        st.success(f"Predicted text: {text}")
    else:
        st.info("Upload an image to start OCR.")


# Navigation
page = st.sidebar.radio("Navigation", options=["Home", "Model Interface"])

if page == "Home":
    home_page()
elif page == "Model Interface":
    model_interface()


# Optional: simple function API similar to your snippet
def predict_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Note: This function requires model and device to be available in the scope
    # For API usage, you would need to load the model separately
    weights_path = _find_weights_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_path(weights_path, device)
    return predict_image(img, model, device)
