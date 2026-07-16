"""
python batok.py --image batok.png
python batok.py --video path/ke/video.mp4
python batok.py --webcam
python batok.py --webcam --camera-id 1 --infer-interval 5 --save-video
python batok.py --video path/ke/video.mp4 --save-video --max-frames 900
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2

CONFIG = {
    "model_id": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",

    "camera_height_m": 2.0,          # meter
    "horizontal_distance_m": 2.0,    # meter

    "roi": (0.08, 0.15, 0.88, 0.99),

    "specific_gravity_ton_per_m3": 0.46,  # 460 kg/m^3

    "outlier_clip_percentile": 98,

    "camera_horizontal_fov_deg": 60.0,   
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    print(f"[1/5] Loading pretrained model: {CONFIG['model_id']} ...")
    processor = AutoImageProcessor.from_pretrained(CONFIG["model_id"])
    model = AutoModelForDepthEstimation.from_pretrained(CONFIG["model_id"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"      Model loaded. Running on: {device}")
    return processor, model, device


def run_depth_inference(image_path, processor, model, device):
    print(f"[2/5] Membaca gambar: {image_path}")
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    print("[3/5] Menjalankan inference depth estimation (AI)...")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return image, depth_resized


def apply_roi(depth_map, roi):
    h, w = depth_map.shape
    x1, y1, x2, y2 = roi
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    return depth_map[y1:y2, x1:x2], (x1, y1, x2, y2)


def estimate_volume_and_tonnage(depth_map_roi, verbose=True):
    if verbose:
        print("[4/5] Menghitung estimasi volume & tonase...")

    h_cam = CONFIG["camera_height_m"]
    s = CONFIG["horizontal_distance_m"]

    ground_depth = float(np.sqrt(h_cam ** 2 + s ** 2))
    tilt_angle_rad = float(np.arctan2(h_cam, s))

    if verbose:
        print(f"      Jarak miring kamera->dasar : {ground_depth:.3f} m")
        print(f"      Sudut kemiringan kamera    : {np.degrees(tilt_angle_rad):.1f} deg")

    height_map = (ground_depth - depth_map_roi) * np.sin(tilt_angle_rad)
    height_map = np.clip(height_map, 0, None)

    clip_pct = CONFIG.get("outlier_clip_percentile", 100)
    if clip_pct < 100 and np.any(height_map > 0):
        cap = np.percentile(height_map, clip_pct)
        height_map = np.clip(height_map, 0, cap)

    h, w = depth_map_roi.shape
    fov_rad = np.radians(CONFIG["camera_horizontal_fov_deg"])
    real_width_at_ground = 2 * ground_depth * np.tan(fov_rad / 2)
    m_per_pixel_x = real_width_at_ground / w
    m_per_pixel_y = m_per_pixel_x
    pixel_area_m2 = m_per_pixel_x * m_per_pixel_y

    volume_m3 = float(np.sum(height_map) * pixel_area_m2)
    tonnage = volume_m3 * CONFIG["specific_gravity_ton_per_m3"]

    max_height_m = float(np.max(height_map))
    mean_height_m = float(np.mean(height_map[height_map > 0])) if np.any(height_map > 0) else 0.0

    return {
        "volume_m3": volume_m3,
        "tonnage_ton": tonnage,
        "max_height_m": max_height_m,
        "mean_height_m": mean_height_m,
        "height_map": height_map,
        "pixel_area_m2": pixel_area_m2,
        "ground_depth_m": ground_depth,
        "tilt_angle_deg": float(np.degrees(tilt_angle_rad)),
    }


def save_visualizations(image, depth_map, roi_box, height_map, out_prefix):
    print("[5/5] Menyimpan hasil visualisasi...")

    lo, hi = np.percentile(depth_map, [2, 98])
    depth_clipped = np.clip(depth_map, lo, hi)
    depth_norm = (hi - depth_clipped) / (hi - lo + 1e-8)
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    Image.fromarray(depth_color_rgb).save(f"{out_prefix}_depthmap.png")

    img_np = np.array(image).copy()
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 3)
    Image.fromarray(img_np).save(f"{out_prefix}_roi.png")

    hlo, hhi = np.percentile(height_map, [2, 98])
    height_clipped = np.clip(height_map, hlo, hhi)
    hm_norm = (height_clipped - hlo) / (hhi - hlo + 1e-8)
    hm_color = cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    hm_color_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    Image.fromarray(hm_color_rgb).save(f"{out_prefix}_heightmap.png")

    return [
        f"{out_prefix}_depthmap.png",
        f"{out_prefix}_roi.png",
        f"{out_prefix}_heightmap.png",
    ]


def run_depth_inference_pil(image_pil, processor, model, device):
    orig_w, orig_h = image_pil.size
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return depth_resized


def estimate_from_pil(image_pil, processor, model, device):
    depth_map = run_depth_inference_pil(image_pil, processor, model, device)
    depth_roi, roi_box = apply_roi(depth_map, CONFIG["roi"])
    result = estimate_volume_and_tonnage(depth_roi, verbose=False)
    return depth_map, roi_box, result


def make_depth_heatmap(depth_map):
    lo, hi = np.percentile(depth_map, [2, 98])
    depth_clipped = np.clip(depth_map, lo, hi)
    depth_norm = (hi - depth_clipped) / (hi - lo + 1e-8)  
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)  


def make_height_heatmap(height_map):
    hlo, hhi = np.percentile(height_map, [2, 98])
    height_clipped = np.clip(height_map, hlo, hhi)
    hm_norm = (height_clipped - hlo) / (hhi - hlo + 1e-8)
    return cv2.applyColorMap((hm_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)  


def draw_overlay(frame_bgr, roi_box, result, extra_text=None):
    out = frame_bgr.copy()
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)

    lines = [
        f"Volume : {result['volume_m3']:,.2f} m3",
        f"Tonase : {result['tonnage_ton']:,.2f} ton",
        f"H max  : {result['max_height_m']:,.2f} m",
        f"H rata : {result['mean_height_m']:,.2f} m",
    ]
    if extra_text:
        lines.append(extra_text)

    for i, line in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def print_result(result, prefix=""):
    print(f"{prefix} Volume: {result['volume_m3']:,.2f} m^3 | "
          f"Tonase: {result['tonnage_ton']:,.2f} ton | "
          f"H max: {result['max_height_m']:,.2f} m | "
          f"H rata: {result['mean_height_m']:,.2f} m")


def run_image_mode(image_path, processor, model, device):
    image, depth_map = run_depth_inference(image_path, processor, model, device)
    depth_roi, roi_box = apply_roi(depth_map, CONFIG["roi"])
    result = estimate_volume_and_tonnage(depth_roi)

    out_prefix = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0])
    saved_files = save_visualizations(image, depth_map, roi_box, result["height_map"], out_prefix)

    print("\n" + "=" * 60)
    print(" HASIL ESTIMASI TUMPUKAN BATOK KELAPA")
    print("=" * 60)
    print(f" Jarak miring kamera->dasar : {result['ground_depth_m']:.2f} m")
    print(f" Sudut kemiringan kamera    : {result['tilt_angle_deg']:.1f} deg")
    print(f" Estimasi Volume            : {result['volume_m3']:,.2f} m^3")
    print(f" Estimasi Tonase            : {result['tonnage_ton']:,.2f} ton")
    print(f" Tinggi Maksimum            : {result['max_height_m']:,.2f} m")
    print(f" Tinggi Rata-rata           : {result['mean_height_m']:,.2f} m")
    print(f" Densitas dipakai           : {CONFIG['specific_gravity_ton_per_m3']} t/m^3")
    print("=" * 60)
    print("\n File hasil visualisasi disimpan di:")
    for f in saved_files:
        print(f"   - {f}")


def run_video_mode(video_path, processor, model, device, infer_interval, save_video, max_frames):
    print(f"[VIDEO] Membuka video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps * infer_interval)))
    print(f"        FPS video: {fps:.1f} | inference tiap {frame_interval} frame (~{infer_interval:.1f}s)")

    out_prefix = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(video_path))[0])
    csv_path = f"{out_prefix}_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_s", "volume_m3", "tonnage_ton", "max_height_m", "mean_height_m"])

    writer = None
    if save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(f"{out_prefix}_annotated.mp4", fourcc, fps, (w, h))

    frame_idx = 0
    last_result = None
    last_roi_box = None
    all_volumes = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        if frame_idx % frame_interval == 0:
            image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            depth_map, roi_box, result = estimate_from_pil(image_pil, processor, model, device)
            last_result, last_roi_box = result, roi_box
            timestamp_s = frame_idx / fps
            csv_writer.writerow([f"{timestamp_s:.2f}", result["volume_m3"], result["tonnage_ton"],
                                  result["max_height_m"], result["mean_height_m"]])
            all_volumes.append(result["volume_m3"])
            print_result(result, prefix=f"[t={timestamp_s:6.1f}s]")

        if writer is not None and last_result is not None:
            writer.write(draw_overlay(frame_bgr, last_roi_box, last_result))

        frame_idx += 1

    cap.release()
    csv_file.close()
    if writer is not None:
        writer.release()

    print(f"\n Log CSV disimpan di : {csv_path}")
    if writer is not None:
        print(f" Video anotasi disimpan : {out_prefix}_annotated.mp4")


def run_webcam_mode(camera_id, processor, model, device, infer_interval, save_video):
    print(f"[WEBCAM] Membuka kamera id={camera_id} ...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka webcam id={camera_id}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_prefix = os.path.join(OUTPUT_DIR, "webcam_session")
    csv_path = f"{out_prefix}_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp_s", "volume_m3", "tonnage_ton", "max_height_m", "mean_height_m"])

    writer = None
    if save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(f"{out_prefix}_annotated.mp4", fourcc, fps, (w, h))

    print("        Tekan 'q' di window untuk berhenti.")
    print(f"        Inference AI tiap ~{infer_interval:.1f} detik (model berat, tidak realtime penuh).")

    t_start = time.time()
    last_infer_time = -infer_interval  
    last_result = None
    last_roi_box = None

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("        Gagal membaca frame dari webcam, berhenti.")
                break

            now = time.time() - t_start
            if now - last_infer_time >= infer_interval:
                image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                depth_map, roi_box, result = estimate_from_pil(image_pil, processor, model, device)
                last_result, last_roi_box = result, roi_box
                last_infer_time = now
                csv_writer.writerow([f"{now:.2f}", result["volume_m3"], result["tonnage_ton"],
                                      result["max_height_m"], result["mean_height_m"]])
                print_result(result, prefix=f"[t={now:6.1f}s]")

            if last_result is not None:
                display = draw_overlay(frame_bgr, last_roi_box, last_result, extra_text="'q' = keluar")
            else:
                display = frame_bgr

            if writer is not None:
                writer.write(display)

            cv2.imshow("Estimasi Stockpile Batok Kelapa (realtime)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        if writer is not None:
            writer.release()

    print(f"\n Log CSV disimpan di : {csv_path}")
    if writer is not None:
        print(f" Video anotasi disimpan : {out_prefix}_annotated.mp4")


def main():
    parser = argparse.ArgumentParser(description="Estimator volume/tonase tumpukan batok kelapa (gambar/video/webcam)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path ke foto tumpukan batok")
    group.add_argument("--video", help="Path ke file video tumpukan batok")
    group.add_argument("--webcam", action="store_true", help="Gunakan webcam realtime")

    parser.add_argument("--camera-id", type=int, default=0, help="ID webcam (default 0)")
    parser.add_argument("--infer-interval", type=float, default=3.0,
                         help="Jeda antar inference AI dalam detik, untuk mode video/webcam (default 3.0)")
    parser.add_argument("--save-video", action="store_true",
                         help="Simpan video hasil anotasi ke folder Output (mode video/webcam)")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Batasi jumlah frame yang diproses untuk mode video panjang")
    args = parser.parse_args()

    processor, model, device = load_model()

    if args.image:
        run_image_mode(args.image, processor, model, device)
    elif args.video:
        run_video_mode(args.video, processor, model, device, args.infer_interval, args.save_video, args.max_frames)
    elif args.webcam:
        run_webcam_mode(args.camera_id, processor, model, device, args.infer_interval, args.save_video)


if __name__ == "__main__":
    main()
