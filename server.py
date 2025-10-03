from flask import Flask, request, send_file, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
from PIL import Image
import tifffile
import io, base64

# ---------------------------
# Initialize Flask
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load trained model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create model architecture
model_flask = fcn_resnet50(pretrained=False)

# 2. Replace first conv layer for 12-channel input
orig_conv = model_flask.backbone.conv1
new_conv = nn.Conv2d(
    in_channels=12,
    out_channels=orig_conv.out_channels,
    kernel_size=orig_conv.kernel_size,
    stride=orig_conv.stride,
    padding=orig_conv.padding,
    bias=False
)
model_flask.backbone.conv1 = new_conv

# 3. Load trained weights
state_dict = torch.load("segmentation_model.pth", map_location=device)
model_flask.load_state_dict(state_dict, strict=False)  # ignore aux_classifier if present
model_flask.to(device)
model_flask.eval()

# ---------------------------
# Helper function: preprocess image
# ---------------------------
def preprocess_image(file, target_size=(128,128)):
    img = tifffile.imread(file).astype(np.float32)

    # Ensure 12 channels
    if img.ndim == 2:
        img = np.stack([img]*12, axis=-1)
    elif img.ndim == 3:
        if img.shape[-1] < 12:
            pad_width = 12 - img.shape[-1]
            img = np.concatenate([img, np.zeros((*img.shape[:2], pad_width), dtype=img.dtype)], axis=-1)
        elif img.shape[-1] > 12:
            img = img[..., :12]

    # Normalize each channel
    img_max = np.max(img, axis=(0,1), keepdims=True)
    img_max[img_max == 0] = 1
    img = img / img_max

    # Resize each channel
    img_resized = []
    for ch in range(12):
        ch_img = Image.fromarray((img[..., ch]*255).astype(np.uint8))
        ch_img = ch_img.resize(target_size, Image.Resampling.BILINEAR)
        img_resized.append(np.array(ch_img, dtype=np.float32)/255.0)
    img = np.stack(img_resized, axis=-1)

    # Convert to tensor
    img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).unsqueeze(0).float()  # [1,12,H,W]
    return img_tensor


# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def index():
    return """
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload a .tif Image for Segmentation</h1>
    <form action="/predict" method=post enctype=multipart/form-data>
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # --- Load original uploaded image (for display only) ---
    original_img = tifffile.imread(file).astype(np.float32)
    if original_img.ndim == 3 and original_img.shape[-1] > 3:
        original_img = original_img[..., :3]  # first 3 channels
    elif original_img.ndim == 2:
        original_img = np.stack([original_img]*3, axis=-1)

    orig_pil = Image.fromarray(
        (255 * (original_img / (original_img.max() + 1e-8))).astype(np.uint8)
    )

    # --- Preprocess for model ---
    file.seek(0)  # reset file pointer
    img_tensor = preprocess_image(file).to(device)

    # --- Forward pass ---
    with torch.no_grad():
        output = model_flask(img_tensor)['out']  # [1,num_classes,H,W]
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # --- Predicted mask ---
    pil_mask = Image.fromarray(pred_mask * 127)  # scaled for visibility
    buf = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    buf.seek(0)
    mask_bytes = buf.getvalue()

    # --- Overlay (mask blended on original image) ---
    mask_rgba = pil_mask.convert("RGBA")
    orig_rgba = orig_pil.convert("RGBA")

    overlay = Image.new("RGBA", mask_rgba.size, (255, 0, 0, 0))
    mask_array = np.array(pred_mask)
    overlay_array = np.array(overlay)
    overlay_array[mask_array > 0] = [255, 0, 0, 120]  # red transparent
    overlay = Image.fromarray(overlay_array, mode="RGBA")

    blended = Image.alpha_composite(orig_rgba, overlay)

    # --- Encode all images for HTML display ---
    def encode_img(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    encoded_orig = encode_img(orig_pil)
    encoded_mask = encode_img(pil_mask)
    encoded_overlay = encode_img(blended)

    html = f"""
    <!doctype html>
    <title>Prediction Result</title>
    <h1>Segmentation Result</h1>

    <h3>Original Image</h3>
    <img src="data:image/png;base64,{encoded_orig}" alt="Original" style="max-width:45%; margin-right:10px;">

    <h3>Predicted Mask</h3>
    <img src="data:image/png;base64,{encoded_mask}" alt="Mask" style="max-width:45%;"><br><br>

    <h3>Overlay (Mask on Image)</h3>
    <img src="data:image/png;base64,{encoded_overlay}" alt="Overlay" style="max-width:60%; border:2px solid #555;"><br><br>

    <a href="/download_mask" target="_blank">⬇️ Download Predicted Mask</a><br><br>
    <a href="/">🔙 Back</a>
    """

    global last_mask_bytes
    last_mask_bytes = mask_bytes

    return render_template_string(html)


@app.route("/download_mask")
def download_mask():
    if 'last_mask_bytes' not in globals():
        return "No mask available to download", 400
    return send_file(
        io.BytesIO(last_mask_bytes),
        mimetype='image/png',
        as_attachment=True,
        download_name='pred_mask.png'
    )


# ---------------------------
# Run Flask
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
