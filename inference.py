import argparse
import io
import json
import os
import subprocess
import sys
import urllib.request
import warnings

try:
    import torch
    from PIL import Image
    from torchvision import transforms
except ImportError:
    _venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
    if os.path.exists(_venv_py) and os.path.normcase(sys.executable) != os.path.normcase(_venv_py):
        sys.exit(subprocess.call([_venv_py] + sys.argv))
    raise

warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(
    model_path="runs/effv2b0/model_submission.pt",
    class_map_path="runs/effv2b0/class_map.json",
    device="cpu",
):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model = torch.jit.optimize_for_inference(model)
    with open(class_map_path, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    return model, idx_to_class


def open_image_any(image_source):
    src = image_source.strip().strip('"').strip("'")
    if src.startswith("file:///"):
        src = src[len("file:///"):]
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    if os.path.exists(src) and os.path.isfile(src):
        return Image.open(src).convert("RGB")

    # Masaüstü/İndirilenler klasörü ve gizli dosya uzantıları için akıllı arama
    home = os.path.expanduser("~")
    base_name = os.path.basename(src)
    stem = os.path.splitext(base_name)[0].lower()
    search_dirs = [
        os.path.dirname(src),
        os.getcwd(),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Downloads"),
        home,
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "Masaüstü"),
    ]
    img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif")
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        exact = os.path.join(d, base_name)
        if os.path.isfile(exact):
            return Image.open(exact).convert("RGB")
        try:
            for fn in os.listdir(d):
                fn_low = fn.lower()
                fn_stem = os.path.splitext(fn_low)[0]
                is_match = fn_low == base_name.lower() or fn_stem in (stem, base_name.lower())
                if is_match and fn_low.endswith(img_exts):
                    cand = os.path.join(d, fn)
                    if os.path.isfile(cand):
                        return Image.open(cand).convert("RGB")
        except OSError:
            pass

    raise FileNotFoundError(f"Görsel bulunamadı: '{image_source}'")


def _crop_rel(img, cx, cy, size=224):
    w, h = img.size
    left = max(0, min(w - size, int(round(cx * w - size / 2))))
    top = max(0, min(h - size, int(round(cy * h - size / 2))))
    return img.crop((left, top, left + size, top + size))


def predict_image(image_path, model, idx_to_class, device="cpu", topk=3):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    img = open_image_any(image_path)

    candidate_crops = []
    # 1) Standart merkez kırpmalar (tekil çiçek ve telefon çekimleri için)
    for sz in (256, 320, 384, 448, 512):
        candidate_crops.append(transforms.CenterCrop(224)(transforms.Resize(sz)(img)))

    # 2) Sepet / buket / uzak çekimler için yüksek yakınlaştırmalı çiçek kuşağı taraması
    for sz in (640, 800):
        resized = transforms.Resize(sz)(img)
        for cx, cy in ((0.30, 0.42), (0.50, 0.40), (0.70, 0.42), (0.50, 0.52)):
            candidate_crops.append(_crop_rel(resized, cx, cy, 224))

    views = []
    for c in candidate_crops:
        c_flip = c.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        views.append(norm(to_tensor(c)))
        views.append(norm(to_tensor(c_flip)))

    x = torch.stack(views).to(device)
    with torch.inference_mode():
        raw_logits = model(x)
        # Sadece rengi benzeyen çiçek dışı nesnelerin zayıf aktivasyonla yüksek yüzde almasını
        # önleyen 103. sanal "Çiçek Dışı (OOD)" enerji kapısı
        ood_gate = torch.full((raw_logits.size(0), 1), 3.5, device=raw_logits.device)
        gated = torch.cat([raw_logits, ood_gate], dim=1) / 0.80
        probs = torch.softmax(gated, dim=1)[:, :raw_logits.size(1)]
        # Her bölgenin düz ve yatay aynalanmış tahmin ortalaması
        region_probs = 0.5 * (probs[0::2] + probs[1::2])
        core_probs = region_probs[:5]
        best_region_idx = int(region_probs.max(dim=1).values.argmax().item())
        best_probs = 0.55 * region_probs[best_region_idx] + 0.45 * core_probs.mean(dim=0)
        confs, pred_idxs = best_probs.topk(topk)

    results = []
    for conf, idx in zip(confs, pred_idxs):
        class_id = idx_to_class[str(int(idx.item()))]
        results.append((class_id, float(conf.item()) * 100.0))
    return results


def main():
    parser = argparse.ArgumentParser(description="Hocaya Teslim Edilecek Saf PyTorch Tahmin Scripti")
    parser.add_argument("pos_image", nargs="*", help="Tahmin edilecek görselin yolu (opsiyonel)")
    parser.add_argument("--image", "-i", type=str, default="", help="Görselin yolu veya URL'si")
    parser.add_argument("--model", type=str, default="runs/effv2b0/model_submission.pt", help="Model dosyası")
    parser.add_argument("--class-map", type=str, default="runs/effv2b0/class_map.json", help="Sınıf haritası")
    parser.add_argument("--cat-names", type=str, default="cat_to_name.json", help="Çiçek isimleri JSON")
    parser.add_argument("--threshold", type=float, default=25.0, help="Düşük güven uyarı eşiği (%)")
    parser.add_argument("--mid-threshold", type=float, default=50.0, help="Orta güven bilgi eşiği (%)")
    args = parser.parse_args()

    img_input = args.image if args.image else (" ".join(args.pos_image) if args.pos_image else "")
    if not img_input:
        parser.error("Lütfen bir görsel yolu belirtin: python inference.py --image <gorsel.jpg>")

    cat_to_name = {}
    if os.path.exists(args.cat_names):
        with open(args.cat_names, "r", encoding="utf-8") as f:
            cat_to_name = json.load(f)

    model, idx_to_class = load_model(args.model, args.class_map)
    results = predict_image(img_input, model, idx_to_class, topk=3)

    best_class_id, best_conf = results[0]
    best_name = cat_to_name.get(best_class_id, "")
    name_str = f" ({best_name.title()})" if best_name else ""
    if best_conf < args.threshold:
        print(
            f"[UYARI] Düşük Güven (%{best_conf:.2f} < %{args.threshold:.0f}) — "
            f"Görsel bir çiçek olmayabilir veya 102 sınıf dışındadır!"
        )
    elif best_conf < args.mid_threshold:
        print(
            f"[BİLGİ] Orta Düzey Güven (%{best_conf:.2f}) — Görsel zor bir açıdan çekilmiş olabilir "
            f"veya 102 sınıf dışındaki benzer bir çiçek türü olabilir (en yakın benzer gösteriliyor):"
        )
    print(f"Tahmin Edilen Sınıf: {best_class_id}{name_str} | Güven: %{best_conf:.2f}")

    if len(results) > 1:
        top3_str = " | ".join(
            f"#{i + 1} Sınıf {cid} ({cat_to_name.get(cid, '?').title()}): %{c:.1f}"
            for i, (cid, c) in enumerate(results)
        )
        print(f"Top-3 Aday: {top3_str}")


if __name__ == "__main__":
    main()
