import os
import time
import io
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import models
import attacks
import configs
import datasets
import stickers

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adversarial Attacks & Resistant Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #8E9AAF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #CBCDD3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge-success {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }
    .status-badge-fail {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GTSRB Class Labels
# -----------------------------------------------------------------------------
GTSRB_CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles > 3.5t',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles > 3.5t prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed & passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing for vehicles > 3.5t'
}

# -----------------------------------------------------------------------------
# Model & Asset Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(model_type="baseline"):
    """Loads a trained CNNTrafficSignNet model from checkpoints or initializes architecture."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net = models.CNNTrafficSignNet().to(device)
    
    ckpt_dir = os.path.join(".", "checkpoints", "traffic_signs")
    if model_type == "baseline":
        ckpt_path = os.path.join(ckpt_dir, "CNN-TrafficSignNet.pt")
    elif model_type == "fgsm_robust":
        ckpt_path = os.path.join(ckpt_dir, "CNN-TrafficSignNet with FGSM adversarial training.pt")
    elif model_type == "pgd_robust":
        ckpt_path = os.path.join(ckpt_dir, "CNN-TrafficSignNet with PGD adversarial training.pt")
    else:
        ckpt_path = None

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt["trained_net"] if "trained_net" in ckpt else ckpt
            if isinstance(state_dict, dict):
                net.load_state_dict(state_dict)
            else:
                net = state_dict
            net.eval()
            return net, True, ckpt_path
        except Exception as e:
            st.warning(f"Could not load checkpoint ({e}), using initialized weights.")
    
    net.eval()
    return net, False, None

@st.cache_data
def get_sample_images():
    """Returns a dictionary of sample GTSRB images with known labels."""
    samples = {}
    test_csv_path = os.path.join(".", "data", "GTSRB", "testset", "test.csv")
    if os.path.exists(test_csv_path):
        df = pd.read_csv(test_csv_path)
        # Select distinct classes for preview
        selected_classes = [14, 13, 1, 12, 17, 23, 25, 38, 9]  # Stop, Yield, Speed 30, Priority, No Entry, etc.
        for class_id in selected_classes:
            rows = df[df['ClassId'] == class_id]
            if len(rows) > 0:
                rel_path = rows.iloc[0]['Filename']
                full_path = os.path.join(".", "data", "GTSRB", "testset", rel_path)
                if os.path.exists(full_path):
                    class_name = GTSRB_CLASSES.get(class_id, f"Class {class_id}")
                    samples[f"{class_name} (ID: {class_id})"] = {
                        "path": full_path,
                        "class_id": class_id,
                        "class_name": class_name
                    }
    return samples

def preprocess_image(pil_img):
    """Converts a PIL Image into model-ready tensor."""
    transform = configs.TrafficSigns_experiments_configs["data_transform"]
    tensor = transform(pil_img).unsqueeze(0)  # Shape: (1, 3, 32, 32)
    return tensor

def tensor_to_display_image(tensor):
    """Converts a normalized tensor back to a displayable NumPy RGB image [0, 1]."""
    # Undo normalization: mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669)
    t = tensor.detach().cpu().squeeze(0).clone()
    mean = torch.tensor([0.3403, 0.3121, 0.3214]).view(3, 1, 1)
    std = torch.tensor([0.2724, 0.2608, 0.2669]).view(3, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    np_img = t.permute(1, 2, 0).numpy()
    return np_img

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### **Adversarial AI Dashboard**")
    st.markdown("Exploring adversarial vulnerabilities and resistant learning mechanisms in Deep Learning.")
    st.markdown("---")
    
    selected_page = st.radio(
        "Navigation",
        [
            "🎯 Adversarial Attack Studio",
            "🛡️ Model Defense Comparison",
            "🏷️ Universal Sticker Attack",
            "📊 Experiment Reports & Analytics",
            "⚡ Live Experiment Runner"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### **System Info**")
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.caption(f"**Execution Device:** `{device_name}`")
    st.caption(f"**PyTorch Version:** `{torch.__version__}`")
    
    baseline_net, baseline_loaded, _ = load_trained_model("baseline")
    robust_net, robust_loaded, _ = load_trained_model("fgsm_robust")
    
    st.caption(f"**Baseline Checkpoint:** {'✅ Loaded' if baseline_loaded else '⚠️ Not Found'}")
    st.caption(f"**Robust Checkpoint:** {'✅ Loaded' if robust_loaded else '⚠️ Not Found'}")

# =============================================================================
# PAGE 1: ADVERSARIAL ATTACK STUDIO
# =============================================================================
if selected_page == "🎯 Adversarial Attack Studio":
    st.markdown('<div class="main-title">🎯 Adversarial Attack Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Craft real-time adversarial perturbations and observe how Deep Neural Networks are fooled by imperceptible noise.</div>', unsafe_allow_html=True)
    
    col_config, col_preview = st.columns([1, 2], gap="medium")
    
    with col_config:
        st.markdown("### ⚙️ **Attack Configuration**")
        
        # Model Selection
        model_choice = st.selectbox(
            "Target Model",
            ["Standard Baseline (CNNTrafficSignNet)", "Adversarially Trained (FGSM Robust)"],
            help="Select which network to attack."
        )
        active_net = baseline_net if "Baseline" in model_choice else robust_net
        
        # Attack Algorithm Selection
        attack_algo = st.selectbox(
            "Attack Method",
            ["FGSM (Fast Gradient Sign Method)", "PGD (Projected Gradient Descent)", "Momentum-FGSM"],
            help="Choose the adversarial optimization algorithm."
        )
        
        # Hyperparameters
        st.markdown("#### **Hyperparameters**")
        eps = st.slider(r"Epsilon (Perturbation Budget $\epsilon$)", min_value=0.01, max_value=0.50, value=0.22, step=0.01,
                        help="Controls the maximum pixel-wise magnitude of the adversarial perturbation.")
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        if "FGSM" in attack_algo and "Momentum" not in attack_algo:
            hp = {"epsilon": eps}
            attack_obj = attacks.FGSM(active_net, loss_fn, hp)
            
        elif "PGD" in attack_algo:
            alpha = st.slider("Step Size (Alpha $\\alpha$)", min_value=0.005, max_value=0.05, value=0.01, step=0.005)
            steps = st.slider("Optimization Steps", min_value=5, max_value=50, value=25, step=5)
            hp = {"epsilon": eps, "alpha": alpha, "steps": steps}
            attack_obj = attacks.PGD(active_net, loss_fn, hp)
            
        else:  # Momentum-FGSM
            alpha = st.slider("Step Size (Alpha $\\alpha$)", min_value=0.005, max_value=0.05, value=0.01, step=0.005)
            steps = st.slider("Optimization Steps", min_value=5, max_value=50, value=20, step=5)
            momentum = st.slider("Momentum Decay ($\\mu$)", min_value=0.1, max_value=0.99, value=0.9, step=0.05)
            hp = {"epsilon": eps, "alpha": alpha, "steps": steps, "momentum": momentum}
            attack_obj = attacks.MomentumFGSM(active_net, loss_fn, hp)
            
        st.markdown("---")
        st.markdown("### 🖼️ **Input Image**")
        input_source = st.radio("Input Source", ["Preset Sample Signs", "Upload Custom Image"], horizontal=True)
        
        selected_pil = None
        true_label_id = None
        true_label_name = "Unknown"
        
        if input_source == "Preset Sample Signs":
            sample_dict = get_sample_images()
            if sample_dict:
                chosen_sample_key = st.selectbox("Pick a Traffic Sign", list(sample_dict.keys()))
                sample_info = sample_dict[chosen_sample_key]
                selected_pil = Image.open(sample_info["path"])
                true_label_id = sample_info["class_id"]
                true_label_name = sample_info["class_name"]
            else:
                st.info("Sample testset images not found in data/GTSRB/testset. Please upload an image.")
        else:
            uploaded_file = st.file_uploader("Upload Image (PNG/JPG/PPM)", type=["png", "jpg", "jpeg", "ppm"])
            if uploaded_file is not None:
                selected_pil = Image.open(uploaded_file).convert('RGB')
                true_label_id = st.number_input("Optional True Label ID (0-42)", min_value=0, max_value=42, value=14)
                true_label_name = GTSRB_CLASSES.get(true_label_id, f"Class {true_label_id}")

    with col_preview:
        if selected_pil is not None:
            # 1. Forward Pass on Clean Input
            input_tensor = preprocess_image(selected_pil)
            model_device = next(active_net.parameters()).device  # GPU if the model is on the GPU
            input_tensor = input_tensor.to(model_device)
            with torch.no_grad():
                clean_logits = active_net(input_tensor)
                clean_probs = torch.softmax(clean_logits, dim=1).squeeze(0)
                clean_pred_id = int(torch.argmax(clean_probs).item())
                clean_pred_conf = float(clean_probs[clean_pred_id].item())
            
            # 2. Generate Adversarial Example
            target_y = torch.tensor([true_label_id if true_label_id is not None else clean_pred_id]).to(model_device)
            
            t0 = time.time()
            adv_tensor = attack_obj.perturb(input_tensor.clone(), target_y)
            attack_time = time.time() - t0
            
            # 3. Forward Pass on Adversarial Input
            with torch.no_grad():
                adv_logits = active_net(adv_tensor)
                adv_probs = torch.softmax(adv_logits, dim=1).squeeze(0)
                adv_pred_id = int(torch.argmax(adv_probs).item())
                adv_pred_conf = float(adv_probs[adv_pred_id].item())
            
            # Metrics Calculations
            clean_display = tensor_to_display_image(input_tensor)
            adv_display = tensor_to_display_image(adv_tensor)
            perturbation = np.abs(adv_display - clean_display)
            pert_amplified = np.clip(perturbation * 8.0, 0.0, 1.0)
            
            l_inf = float(np.max(perturbation))
            l_2 = float(np.linalg.norm(perturbation))
            is_attack_successful = (adv_pred_id != (true_label_id if true_label_id is not None else clean_pred_id))
            
            # Metric Banner
            st.markdown("### 📊 **Attack Outcome**")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{"🎯 Success" if is_attack_successful else "🛡️ Resisted"}</div><div class="metric-label">Attack Status</div></div>', unsafe_allow_html=True)
            with m_col2:
                conf_change = (clean_pred_conf - (float(adv_probs[clean_pred_id].item()))) * 100
                st.markdown(f'<div class="metric-box"><div class="metric-val">{conf_change:.1f}%</div><div class="metric-label">Confidence Drop</div></div>', unsafe_allow_html=True)
            with m_col3:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{l_inf:.3f}</div><div class="metric-label">L-Infinity (Max Shift)</div></div>', unsafe_allow_html=True)
            with m_col4:
                st.markdown(f'<div class="metric-box"><div class="metric-val">{attack_time*1000:.1f} ms</div><div class="metric-label">Attack Latency</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Image Comparison Columns
            img_col1, img_col2, img_col3 = st.columns(3)
            
            with img_col1:
                st.markdown("##### **1. Clean Input**")
                st.image(clean_display, use_container_width=True, clamp=True)
                pred_label_clean = GTSRB_CLASSES.get(clean_pred_id, f"Class {clean_pred_id}")
                st.markdown(f"**Prediction:** `{pred_label_clean}`")
                st.markdown(f"**Confidence:** `{clean_pred_conf*100:.2f}%`")
                if true_label_id is not None:
                    st.caption(f"Ground Truth: **{true_label_name}**")
                    
            with img_col2:
                st.markdown("##### **2. Perturbation (8× Amplified)**")
                st.image(pert_amplified, use_container_width=True, clamp=True)
                st.markdown(f"**Algorithm:** `{attack_algo.split('(')[0].strip()}`")
                st.markdown(f"**Epsilon ($\\epsilon$):** `{eps}`")
                st.caption(f"L2 Norm: `{l_2:.4f}`")
                
            with img_col3:
                st.markdown("##### **3. Adversarial Example**")
                st.image(adv_display, use_container_width=True, clamp=True)
                pred_label_adv = GTSRB_CLASSES.get(adv_pred_id, f"Class {adv_pred_id}")
                st.markdown(f"**Prediction:** `{pred_label_adv}`")
                st.markdown(f"**Confidence:** `{adv_pred_conf*100:.2f}%`")
                badge_html = '<span class="status-badge-fail">🚨 FOOLED MODEL</span>' if is_attack_successful else '<span class="status-badge-success">🛡️ MODEL DEFENDED</span>'
                st.markdown(badge_html, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📈 **Prediction Probability Distribution (Top-5 Classes)**")
            
            # Top-5 Chart
            clean_top5 = torch.topk(clean_probs.detach().cpu(), 5)  # .cpu() so .numpy() works on GPU
            adv_top5 = torch.topk(adv_probs.detach().cpu(), 5)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.caption("Clean Image Probabilities")
                top5_clean_names = [GTSRB_CLASSES.get(int(idx), f"Class {idx}") for idx in clean_top5.indices]
                fig_clean = go.Figure(go.Bar(
                    x=clean_top5.values.numpy() * 100,
                    y=top5_clean_names,
                    orientation='h',
                    marker=dict(color='#4ECDC4')
                ))
                fig_clean.update_layout(xaxis_title="Confidence (%)", yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=20), height=240)
                st.plotly_chart(fig_clean, use_container_width=True)
                
            with chart_col2:
                st.caption("Adversarial Image Probabilities")
                top5_adv_names = [GTSRB_CLASSES.get(int(idx), f"Class {idx}") for idx in adv_top5.indices]
                bar_colors = ['#FF6B6B' if idx != (true_label_id if true_label_id is not None else clean_pred_id) else '#4ECDC4' for idx in adv_top5.indices]
                fig_adv = go.Figure(go.Bar(
                    x=adv_top5.values.numpy() * 100,
                    y=top5_adv_names,
                    orientation='h',
                    marker=dict(color=bar_colors)
                ))
                fig_adv.update_layout(xaxis_title="Confidence (%)", yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=20), height=240)
                st.plotly_chart(fig_adv, use_container_width=True)

        else:
            st.info("Select a sample sign or upload an image to begin the attack demonstration.")

# =============================================================================
# PAGE 2: MODEL DEFENSE COMPARISON
# =============================================================================
elif selected_page == "🛡️ Model Defense Comparison":
    st.markdown('<div class="main-title">🛡️ Model Defense Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Direct head-to-head comparison: Standard Baseline vs Adversarially Trained Model under the same attack.</div>', unsafe_allow_html=True)
    
    sample_dict = get_sample_images()
    if not sample_dict:
        st.warning("GTSRB test samples not found. Please ensure data/GTSRB/testset exists.")
    else:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])
        with ctrl_col1:
            chosen_sample = st.selectbox("Select Traffic Sign Test Sample", list(sample_dict.keys()), key="comp_sample")
        with ctrl_col2:
            comp_attack_type = st.selectbox("Attack Type", ["PGD Attack (Strong)", "FGSM Attack (Fast)"])
        with ctrl_col3:
            comp_eps = st.slider("Attack Epsilon ($\\epsilon$)", 0.05, 0.40, 0.22, 0.01)
            
        sample_data = sample_dict[chosen_sample]
        pil_img = Image.open(sample_data["path"])
        ground_truth_id = sample_data["class_id"]
        ground_truth_name = sample_data["class_name"]
        
        comp_device = next(baseline_net.parameters()).device  # GPU if the models are on the GPU
        tensor_in = preprocess_image(pil_img).to(comp_device)
        loss_fn = torch.nn.CrossEntropyLoss()
        y_target = torch.tensor([ground_truth_id]).to(comp_device)
        
        # 1. Attack Baseline Model
        if "PGD" in comp_attack_type:
            atk_baseline = attacks.PGD(baseline_net, loss_fn, {"epsilon": comp_eps, "alpha": 0.01, "steps": 30})
            atk_robust = attacks.PGD(robust_net, loss_fn, {"epsilon": comp_eps, "alpha": 0.01, "steps": 30})
        else:
            atk_baseline = attacks.FGSM(baseline_net, loss_fn, {"epsilon": comp_eps})
            atk_robust = attacks.FGSM(robust_net, loss_fn, {"epsilon": comp_eps})
            
        adv_tensor_baseline = atk_baseline.perturb(tensor_in.clone(), y_target)
        adv_tensor_robust = atk_robust.perturb(tensor_in.clone(), y_target)
        
        # Forward passes
        with torch.no_grad():
            clean_base_probs = torch.softmax(baseline_net(tensor_in), dim=1).squeeze(0)
            adv_base_probs = torch.softmax(baseline_net(adv_tensor_baseline), dim=1).squeeze(0)
            
            clean_rob_probs = torch.softmax(robust_net(tensor_in), dim=1).squeeze(0)
            adv_rob_probs = torch.softmax(robust_net(adv_tensor_robust), dim=1).squeeze(0)
            
        pred_base_clean = int(torch.argmax(clean_base_probs).item())
        pred_base_adv = int(torch.argmax(adv_base_probs).item())
        
        pred_rob_clean = int(torch.argmax(clean_rob_probs).item())
        pred_rob_adv = int(torch.argmax(adv_rob_probs).item())
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"**Ground Truth Label:** {ground_truth_name} (Class ID: {ground_truth_id})")
        
        col_base, col_rob = st.columns(2, gap="large")
        
        with col_base:
            st.markdown("### 🔴 **Standard Model (Without Defense)**")
            st.markdown("Trained strictly on standard empirical risk minimization.")
            
            b_sub1, b_sub2 = st.columns(2)
            with b_sub1:
                st.caption("Clean Input")
                st.image(tensor_to_display_image(tensor_in), use_container_width=True)
                st.markdown(f"Pred: **{GTSRB_CLASSES.get(pred_base_clean, f'Class {pred_base_clean}')}**")
                st.markdown(f"Conf: `{clean_base_probs[pred_base_clean]*100:.1f}%`")
            with b_sub2:
                st.caption(f"After {comp_attack_type.split()[0]} Attack")
                st.image(tensor_to_display_image(adv_tensor_baseline), use_container_width=True)
                st.markdown(f"Pred: **{GTSRB_CLASSES.get(pred_base_adv, f'Class {pred_base_adv}')}**")
                st.markdown(f"Conf: `{adv_base_probs[pred_base_adv]*100:.1f}%`")
                
            if pred_base_adv != ground_truth_id:
                st.error(f"❌ **Model Fooled!** Misclassified as: {GTSRB_CLASSES.get(pred_base_adv, str(pred_base_adv))}")
            else:
                st.success("✅ Maintained correct classification.")
                
        with col_rob:
            st.markdown("### 🟢 **Adversarially Trained Model (Robust)**")
            st.markdown("Trained using resistant learning / min-max adversarial formulation.")
            
            r_sub1, r_sub2 = st.columns(2)
            with r_sub1:
                st.caption("Clean Input")
                st.image(tensor_to_display_image(tensor_in), use_container_width=True)
                st.markdown(f"Pred: **{GTSRB_CLASSES.get(pred_rob_clean, f'Class {pred_rob_clean}')}**")
                st.markdown(f"Conf: `{clean_rob_probs[pred_rob_clean]*100:.1f}%`")
            with r_sub2:
                st.caption(f"After {comp_attack_type.split()[0]} Attack")
                st.image(tensor_to_display_image(adv_tensor_robust), use_container_width=True)
                st.markdown(f"Pred: **{GTSRB_CLASSES.get(pred_rob_adv, f'Class {pred_rob_adv}')}**")
                st.markdown(f"Conf: `{adv_rob_probs[pred_rob_adv]*100:.1f}%`")
                
            if pred_rob_adv == ground_truth_id:
                st.success(f"🛡️ **Defense Successful!** Correctly classified as: {GTSRB_CLASSES.get(pred_rob_adv, str(pred_rob_adv))}")
            else:
                st.warning(f"⚠️ Perturbation overwhelmed defense. Predicted: {GTSRB_CLASSES.get(pred_rob_adv, str(pred_rob_adv))}")

# =============================================================================
# PAGE: UNIVERSAL STICKER ATTACK
# =============================================================================
elif selected_page == "🏷️ Universal Sticker Attack":
    st.markdown('<div class="main-title">🏷️ Universal Sticker Attack</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Place simple black / white stickers on a sign and watch the prediction change. '
                'The same sticker location is applied to every sign, so no per-image perturbation is needed '
                '(after Etim &amp; Szefer, 2025).</div>', unsafe_allow_html=True)

    sticker_samples = get_sample_images()
    if not sticker_samples:
        st.warning("GTSRB test samples not found. Please ensure data/GTSRB/testset exists.")
    else:
        stk_mean = (0.3403, 0.3121, 0.3214)
        stk_std = (0.2724, 0.2608, 0.2669)
        pgd_net, pgd_loaded, _ = load_trained_model("pgd_robust")
        stk_models = {"Standard Baseline": baseline_net, "FGSM Adversarially Trained": robust_net}
        if pgd_loaded:
            stk_models["PGD Adversarially Trained"] = pgd_net
        stk_default_hp = {"sizes": [(6, 6), (8, 8), (10, 10), (12, 12)], "region": (6, 6, 26, 26), "stride": 2}
        stk_hp = dict(getattr(configs, "sticker_hps", {}).get("traffic_signs", stk_default_hp))

        col_cfg, col_view = st.columns([1, 2], gap="medium")

        with col_cfg:
            st.markdown("### ⚙️ **Sticker Configuration**")
            stk_model_name = st.selectbox("Target Model", list(stk_models.keys()), key="stk_model")
            stk_net = stk_models[stk_model_name]
            stk_device = next(stk_net.parameters()).device
            stk_sign_key = st.selectbox("Traffic Sign", list(sticker_samples.keys()), key="stk_sign")
            stk_variant = st.selectbox("Stickers", ["One black", "One white", "Black + White"], key="stk_variant")
            stk_colors = {"One black": ["black"], "One white": ["white"], "Black + White": ["black", "white"]}[stk_variant]
            stk_mode = st.radio("Placement", ["Manual", "Auto-search (universal)"], key="stk_mode")

            stk_placements = []
            if stk_mode == "Manual":
                for i, stk_color in enumerate(stk_colors):
                    st.markdown(f"**Sticker {i + 1} ({stk_color})**")
                    stk_h = st.slider("Height (px)", 2, 16, 8, key=f"stk_h{i}")
                    stk_w = st.slider("Width (px)", 2, 16, 8, key=f"stk_w{i}")
                    stk_top = st.slider("Top (px)", 0, 32 - stk_h, min((32 - stk_h) // 2 + 8 * i, 32 - stk_h))
                    stk_left = st.slider("Left (px)", 0, 32 - stk_w, (32 - stk_w) // 2)
                    stk_placements.append((stk_color, stk_top, stk_left, stk_h, stk_w))
            else:
                st.caption("Searches the best location over all preset signs. Only signs the model already "
                           "classifies correctly are used for the search.")
                stk_state_key = f"stk_auto::{stk_model_name}::{stk_variant}"
                if st.button("🔍 Search universal sticker location", key="stk_search"):
                    with st.spinner("Searching sticker location..."):
                        search_xs = torch.cat([preprocess_image(Image.open(v["path"]).convert("RGB"))
                                               for v in sticker_samples.values()]).to(stk_device)
                        search_ys = torch.tensor([v["class_id"] for v in sticker_samples.values()]).to(stk_device)
                        searcher = stickers.UniversalSticker(stk_net, None, {**stk_hp, "colors": stk_colors},
                                                             mean=stk_mean, std=stk_std)
                        st.session_state[stk_state_key] = searcher.fit([(search_xs, search_ys)],
                                                                       max_images=len(search_xs), device=stk_device)
                stk_placements = st.session_state.get(stk_state_key, [])
                if not stk_placements:
                    st.info("Press the button to find the sticker location.")

        with col_view:
            if not stk_placements:
                st.info("Configure the stickers (or run the search) to see the result.")
            else:
                stk_attack = stickers.UniversalSticker(stk_net, None, {"colors": stk_colors},
                                                       mean=stk_mean, std=stk_std)
                sign_info = sticker_samples[stk_sign_key]
                clean_t = preprocess_image(Image.open(sign_info["path"]).convert("RGB")).to(stk_device)
                sticker_t = stk_attack.apply(clean_t, stk_placements)
                with torch.no_grad():
                    clean_p = torch.softmax(stk_net(clean_t), dim=1).squeeze(0).cpu()
                    stick_p = torch.softmax(stk_net(sticker_t), dim=1).squeeze(0).cpu()
                clean_id, stick_id = int(clean_p.argmax()), int(stick_p.argmax())
                clean_name = GTSRB_CLASSES.get(clean_id, f"Class {clean_id}")
                stick_name = GTSRB_CLASSES.get(stick_id, f"Class {stick_id}")

                st.markdown(f"**Ground truth:** {sign_info['class_name']} (ID {sign_info['class_id']})")
                img_a, img_b = st.columns(2)
                with img_a:
                    st.markdown("##### **Clean Sign**")
                    st.image(tensor_to_display_image(clean_t), use_container_width=True, clamp=True)
                    st.markdown(f"**Prediction:** `{clean_name}`")
                    st.markdown(f"**Confidence:** `{float(clean_p[clean_id]) * 100:.1f}%`")
                with img_b:
                    st.markdown("##### **With Sticker(s)**")
                    st.image(tensor_to_display_image(sticker_t), use_container_width=True, clamp=True)
                    st.markdown(f"**Prediction:** `{stick_name}`")
                    st.markdown(f"**Confidence:** `{float(stick_p[stick_id]) * 100:.1f}%`")

                if stick_id != sign_info["class_id"]:
                    st.error(f"❌ **Model fooled!** Predicted: {stick_name}")
                else:
                    st.success("✅ The model still classifies this sign correctly.")

                # the same stickers on every preset sign (this is what makes the attack universal)
                st.markdown("#### 🌍 **Same stickers on every preset sign**")
                all_items = list(sticker_samples.items())
                all_xs = torch.cat([preprocess_image(Image.open(v["path"]).convert("RGB"))
                                    for _, v in all_items]).to(stk_device)
                all_true = [v["class_id"] for _, v in all_items]
                with torch.no_grad():
                    all_clean = stk_net(all_xs).argmax(dim=1).cpu().tolist()
                    all_stick = stk_net(stk_attack.apply(all_xs, stk_placements)).argmax(dim=1).cpu().tolist()
                rows, fooled, usable = [], 0, 0
                for (name, _), true_id, c_id, s_id in zip(all_items, all_true, all_clean, all_stick):
                    if c_id == true_id:
                        usable += 1
                        fooled += int(s_id != true_id)
                    rows.append({"Sign": name,
                                 "Clean prediction": GTSRB_CLASSES.get(c_id, str(c_id)),
                                 "With sticker": GTSRB_CLASSES.get(s_id, str(s_id)),
                                 "Fooled": "yes" if (c_id == true_id and s_id != true_id) else "no"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(f"Fooled {fooled} of {usable} signs that the model classifies correctly without stickers. "
                           "This is a digital simulation on 32x32 crops, not a physical-world test.")

# =============================================================================
# PAGE 3: EXPERIMENT REPORTS & ANALYTICS
# =============================================================================
elif selected_page == "📊 Experiment Reports & Analytics":
    st.markdown('<div class="main-title">📊 Experiment Reports & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprehensive evaluation metrics from the paper experiments across GTSRB and MNIST.</div>', unsafe_allow_html=True)
    
    tab_overview, tab_plots, tab_logs = st.tabs(["📋 Resistance Summary Table", "🖼️ Attack Visualizations", "📜 Raw Execution Logs"])
    
    with tab_overview:
        st.markdown("### 🔬 **Paper Empirical Findings Summary**")
        st.markdown("Comparison of model accuracy under various training regimes and adversarial attacks ($\\epsilon=0.22$ for GTSRB, $\\epsilon=0.3$ for MNIST).")
        
        gtsrb_data = {
            "Training Method": ["Standard Training (Natural)", "FGSM Adversarial Training", "PGD Adversarial Training"],
            "Clean Test Accuracy": ["97.8%", "94.6%", "95.1%"],
            "Accuracy under FGSM Attack": ["17.3%", "82.4%", "87.6%"],
            "Accuracy under PGD Attack": ["5.5%", "64.1%", "84.3%"],
            "Robustness Assessment": ["Highly Vulnerable", "Partially Robust", "SOTA Resilient"]
        }
        df_gtsrb = pd.DataFrame(gtsrb_data)
        st.dataframe(df_gtsrb, use_container_width=True, hide_index=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Comparison Bar Chart
        st.markdown("#### **Interactive Robustness Breakdown**")
        fig_bar = go.Figure(data=[
            go.Bar(name='Clean Test Acc', x=df_gtsrb['Training Method'], y=[97.8, 94.6, 95.1], marker_color='#4ECDC4'),
            go.Bar(name='Under FGSM Attack', x=df_gtsrb['Training Method'], y=[17.3, 82.4, 87.6], marker_color='#FFA07A'),
            go.Bar(name='Under PGD Attack', x=df_gtsrb['Training Method'], y=[5.5, 64.1, 84.3], marker_color='#FF6B6B')
        ])
        fig_bar.update_layout(barmode='group', yaxis_title='Accuracy (%)', height=360, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab_plots:
        st.markdown("### 🖼️ **Saved Attack Visualizations**")
        plots_dir = os.path.join(".", "results_folder", "traffic_signs", "plots")
        ex_plots_dir = os.path.join(".", "an_example_of_execution_results", "traffic_signs", "plots")
        
        active_plots_dir = plots_dir if (os.path.exists(plots_dir) and len(os.listdir(plots_dir)) > 0) else ex_plots_dir
        
        if os.path.exists(active_plots_dir):
            plot_files = [f for f in os.listdir(active_plots_dir) if f.endswith('.png')]
            if plot_files:
                selected_plot = st.selectbox("Select Generated Visualization", plot_files)
                full_plot_path = os.path.join(active_plots_dir, selected_plot)
                st.image(full_plot_path, caption=selected_plot, use_container_width=True)
            else:
                st.info("No plot images found in results directory yet.")
        else:
            st.info("Plots directory not found. Run experiments to generate figures.")
            
    with tab_logs:
        st.markdown("### 📜 **Experiment Logs**")
        log_path = os.path.join(".", "results_folder", "traffic_signs", "log.txt")
        ex_log_path = os.path.join(".", "an_example_of_execution_results", "traffic_signs", "log.txt")
        
        chosen_log = log_path if os.path.exists(log_path) else ex_log_path
        if os.path.exists(chosen_log):
            with open(chosen_log, "r") as f:
                log_content = f.read()
            st.text_area(f"Log File ({chosen_log})", log_content, height=450)
        else:
            st.info("No log file found.")

# =============================================================================
# PAGE 4: LIVE EXPERIMENT RUNNER
# =============================================================================
elif selected_page == "⚡ Live Experiment Runner":
    st.markdown('<div class="main-title">⚡ Live Experiment Runner</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Trigger model training and adversarial validation directly from the browser.</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        exp_dataset = st.selectbox("Select Dataset", ["traffic_signs (GTSRB)", "MNIST"])
        exp_mode = st.selectbox("Experiment Mode", [
            "Experiment 1: Train Standard Model & Attack",
            "Experiment 2: Train Adversarially Robust Models",
            "Experiment 3: Capacity & Robustness Analysis"
        ])
    with c2:
        st.markdown("#### **Execution Command**")
        ds_arg = "traffic_signs" if "traffic" in exp_dataset else "MNIST"
        cmd = f".\\venv\\Scripts\\python.exe experiments.py --dataset-name {ds_arg}"
        st.code(cmd, language="powershell")
        
    st.markdown("---")
    st.markdown("### 📝 **Live Terminal Log Monitor**")
    
    log_file_target = os.path.join(".", "results_folder", ds_arg, "log.txt")
    if os.path.exists(log_file_target):
        with open(log_file_target, "r") as f:
            lines = f.readlines()
        st.caption(f"Showing last 50 lines of `{log_file_target}`:")
        st.code("".join(lines[-50:]), language="text")
        if st.button("🔄 Refresh Live Logs"):
            st.rerun()
    else:
        st.info(f"Log file `{log_file_target}` will be generated once training starts.")