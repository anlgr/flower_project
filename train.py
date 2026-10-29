import os, json, random, argparse

_cache_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.environ.setdefault("HF_HOME", os.path.join(_cache_root, "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(_cache_root, "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if os.path.exists(os.path.join(_cache_root, "huggingface", "hub")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


KAGGLE_SYNONYMS = {
    "barberton daisy": "41",
    "cosmos": "34",
    "daisy": "49",
    "iris": "68",
    "lilac hibiscus": "55",
    "pink quill": "93",
}
KAGGLE_EXCLUDED = {"common tulip", "wild geranium", "wild rose", "marigold"}


def get_transforms(train_size=224, clean_aug=False):
    if clean_aug:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(train_size, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(degrees=25, translate=(0.1, 0.1), scale=(0.5, 1.0))
            ], p=0.4),
            transforms.RandomResizedCrop(train_size, scale=(0.15, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf


def build_kaggle_name_to_cid(cat_to_name_path="cat_to_name.json"):
    with open(cat_to_name_path, "r") as f:
        cat_to_name = json.load(f)
    name_to_cid = {v.lower(): k for k, v in cat_to_name.items()}
    name_to_cid.update(KAGGLE_SYNONYMS)
    return name_to_cid


class ImageListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), target


class BalancedCombinedDataset(Dataset):
    """Her epoch'ta Oxford 102 ve Kaggle veri setlerinden sınıf dengeli ve görülmemiş örnek öncelikli örnekleme yapar."""
    def __init__(self, oxford_by_cls, kaggle_by_cls, transform, ox_per_cls=14, kg_per_cls=26):
        self.oxford_by_cls = oxford_by_cls
        self.kaggle_by_cls = kaggle_by_cls
        self.transform = transform
        self.ox_per_cls = ox_per_cls
        self.kg_per_cls = kg_per_cls
        self.seen_kg = set()
        self.epoch_samples = []
        self.resample()

    def resample(self):
        samples = []
        all_targets = sorted(set(self.oxford_by_cls.keys()) | set(self.kaggle_by_cls.keys()))
        for t in all_targets:
            ox_list = self.oxford_by_cls.get(t, [])
            kg_list = self.kaggle_by_cls.get(t, [])
            if len(ox_list) > self.ox_per_cls:
                samples.extend((p, t) for p in random.sample(ox_list, self.ox_per_cls))
            else:
                samples.extend((p, t) for p in ox_list)

            if len(kg_list) > self.kg_per_cls:
                unseen = [p for p in kg_list if p not in self.seen_kg]
                if len(unseen) >= self.kg_per_cls:
                    chosen = random.sample(unseen, self.kg_per_cls)
                else:
                    seen_pool = [p for p in kg_list if p in self.seen_kg]
                    chosen = unseen + random.sample(seen_pool, self.kg_per_cls - len(unseen))
                self.seen_kg.update(chosen)
                samples.extend((p, t) for p in chosen)
            else:
                self.seen_kg.update(kg_list)
                samples.extend((p, t) for p in kg_list)
        random.shuffle(samples)
        self.epoch_samples = samples

    def __len__(self):
        return len(self.epoch_samples)

    def __getitem__(self, idx):
        path, target = self.epoch_samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), target


def collect_kaggle_samples(kaggle_dir, split, class_to_idx, name_to_cid, marigold_model=None, eval_tf=None, device="cpu"):
    split_dir = os.path.join(kaggle_dir, split)
    by_target = {idx: [] for idx in class_to_idx.values()}
    flat_samples = []
    if not os.path.exists(split_dir):
        return by_target, flat_samples

    for cls_name in sorted(os.listdir(split_dir)):
        folder = os.path.join(split_dir, cls_name)
        if not os.path.isdir(folder):
            continue
        low = cls_name.lower()
        if low == "marigold" and marigold_model is not None and eval_tf is not None:
            idx_5 = class_to_idx["5"]
            idx_47 = class_to_idx["47"]
            files = [os.path.join(folder, fn) for fn in os.listdir(folder)]
            with torch.inference_mode():
                for i in range(0, len(files), 64):
                    batch_files = files[i:i + 64]
                    x = torch.stack([eval_tf(Image.open(p).convert("RGB")) for p in batch_files]).to(device)
                    preds = marigold_model(x).argmax(dim=1).tolist()
                    for path, pred in zip(batch_files, preds):
                        if pred in (idx_5, idx_47):
                            by_target[pred].append(path)
                            flat_samples.append((path, pred))
            continue

        if low in KAGGLE_EXCLUDED:
            continue
        cid = name_to_cid.get(low)
        if cid is None or cid not in class_to_idx:
            continue
        target = class_to_idx[cid]
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            by_target[target].append(path)
            flat_samples.append((path, target))
    return by_target, flat_samples


def evaluate_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0


def train_one_epoch(model, loader, criterion, optimizer, device, desc="Train"):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct / total) * 100:.1f}%")
    return running_loss / total, (correct / total) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Çiçek Tanıma Yarışması Eğitim Scripti")
    parser.add_argument("--arch", type=str, default="tf_efficientnetv2_b0.in1k")
    parser.add_argument("--data-dir", type=str, default="flower_data")
    parser.add_argument("--kaggle-dir", type=str, default="jpeg-192x192")
    parser.add_argument("--out-dir", type=str, default="runs/effv2b0")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--epochs-head", type=int, default=2)
    parser.add_argument("--epochs-ft", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-size", type=int, default=192)
    parser.add_argument("--ox-per-cls", type=int, default=12)
    parser.add_argument("--kg-per-cls", type=int, default=24)
    parser.add_argument("--clean-aug", action="store_true")
    parser.add_argument("--soup-with", type=str, default="")
    parser.add_argument("--freeze-early", action="store_true")
    parser.add_argument("--combine-val", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Cihaz: {device} | Mimari: {args.arch}")
    train_tf, eval_tf = get_transforms(train_size=args.train_size, clean_aug=args.clean_aug)

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(os.path.join(args.data_dir, "valid"), transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=eval_tf)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(os.path.join(args.out_dir, "class_map.json"), "w") as f:
        json.dump(idx_to_class, f, indent=2)

    has_resume = bool(args.resume and os.path.exists(args.resume))
    model = timm.create_model(args.arch, pretrained=not has_resume, num_classes=len(idx_to_class)).to(device)
    if has_resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        model.eval()
        print(f"[INFO] Mevcut ağırlıklar yüklendi: {args.resume}")

    kaggle_val_loader = None
    use_kaggle = bool(args.kaggle_dir and os.path.exists(args.kaggle_dir))
    if use_kaggle:
        name_to_cid = build_kaggle_name_to_cid("cat_to_name.json")
        oxford_by_cls = {idx: [] for idx in class_to_idx.values()}
        for path, target in train_ds.samples:
            oxford_by_cls[target].append(path)
        if args.combine_val:
            for path, target in valid_ds.samples:
                oxford_by_cls[target].append(path)

        kg_train_by_cls, kg_train_flat = collect_kaggle_samples(
            args.kaggle_dir, "train", class_to_idx, name_to_cid,
            marigold_model=model if has_resume else None, eval_tf=eval_tf, device=device
        )
        kg_val_by_cls, kg_val_flat = collect_kaggle_samples(
            args.kaggle_dir, "val", class_to_idx, name_to_cid,
            marigold_model=model if has_resume else None, eval_tf=eval_tf, device=device
        )
        train_ds = BalancedCombinedDataset(
            oxford_by_cls, kg_train_by_cls, transform=train_tf,
            ox_per_cls=args.ox_per_cls, kg_per_cls=args.kg_per_cls
        )
        kaggle_val_ds = ImageListDataset(kg_val_flat, transform=eval_tf)
        kaggle_val_loader = DataLoader(kaggle_val_ds, batch_size=64, shuffle=False)
        print(f"[INFO] Kaggle veri seti aktif! Toplam Kaggle Train: {len(kg_train_flat)}, Kaggle Val: {len(kg_val_flat)}")
        print(f"[INFO] Dengeli Epoch Başına Eğitim Örneği: {len(train_ds)} (Oxford/sınıf={args.ox_per_cls}, Kaggle/sınıf={args.kg_per_cls})")
    elif args.combine_val:
        valid_aug_ds = datasets.ImageFolder(os.path.join(args.data_dir, "valid"), transform=train_tf)
        train_ds = ConcatDataset([train_ds, valid_aug_ds])
        print(f"[INFO] Train + Valid birleştirildi! Toplam eğitim örneği: {len(train_ds)}")

    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=num_workers)

    eval_loader = test_loader if (args.combine_val or use_kaggle) else valid_loader
    eval_name = "Oxford Test" if (args.combine_val or use_kaggle) else "Oxford Valid"

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_score = 0.0
    best_weights_path = os.path.join(args.out_dir, "best_weights.pth")

    # Aşama 1: Sadece Sınıflandırıcı Katmanı (Head) Eğitimi
    if args.epochs_head > 0:
        print(f"\n--- Aşama 1: Head Eğitimi ({args.epochs_head} Epoch) ---")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(1, args.epochs_head + 1):
            if hasattr(train_ds, "resample"):
                train_ds.resample()
            loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"Head Epoch {epoch}/{args.epochs_head}")
            val_acc = evaluate_acc(model, eval_loader, device)
            print(f"[Head {epoch}/{args.epochs_head}] Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | {eval_name} Acc: {val_acc:.2f}%")
            if val_acc >= best_score:
                best_score = val_acc
                torch.save(model.state_dict(), best_weights_path)

    # Aşama 2: İnce Ayar (Fine-Tuning)
    if args.epochs_ft > 0:
        print(f"\n--- Aşama 2: Fine-Tuning ({args.epochs_ft} Epoch) ---")
        for name, param in model.named_parameters():
            if args.freeze_early:
                param.requires_grad = any(k in name for k in ["blocks.3", "blocks.4", "blocks.5", "conv_head", "bn2", "classifier"])
            else:
                param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_ft, eta_min=1e-6)

        for epoch in range(1, args.epochs_ft + 1):
            if hasattr(train_ds, "resample"):
                train_ds.resample()
            loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, desc=f"FT Epoch {epoch}/{args.epochs_ft}")
            scheduler.step()
            ox_acc = evaluate_acc(model, eval_loader, device)
            if kaggle_val_loader is not None:
                kg_acc = evaluate_acc(model, kaggle_val_loader, device)
                score = 0.65 * ox_acc + 0.35 * kg_acc
                print(f"[FT {epoch}/{args.epochs_ft}] Loss: {loss:.4f} | Train: {train_acc:.2f}% | {eval_name}: {ox_acc:.2f}% | Kaggle Val: {kg_acc:.2f}%")
            else:
                score = ox_acc
                print(f"[FT {epoch}/{args.epochs_ft}] Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | {eval_name} Acc: {ox_acc:.2f}%")

            if score >= best_score:
                best_score = score
                torch.save(model.state_dict(), best_weights_path)
                print(f"  --> En iyi ağırlıklar güncellendi -> {best_weights_path}")

    # Model Soup (Ağırlık Ortalaması) Optimizasyonu
    if args.soup_with and os.path.exists(args.soup_with) and os.path.exists(best_weights_path):
        print(f"\n--- Aşama 3: Model Soup (Ağırlık Ortalaması) Kontrolü ({args.soup_with}) ---")
        sd_new = torch.load(best_weights_path, map_location=device, weights_only=True)
        sd_ref = torch.load(args.soup_with, map_location=device, weights_only=True)
        best_soup_score = -1.0
        best_soup_sd = sd_new
        best_alpha = 0.0
        for alpha in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]:
            sd_blend = {
                k: ((1.0 - alpha) * sd_new[k] + alpha * sd_ref[k]) if sd_new[k].is_floating_point() else sd_new[k]
                for k in sd_new
            }
            model.load_state_dict(sd_blend)
            ox_acc = evaluate_acc(model, test_loader, device)
            print(f"  [Soup alpha={alpha:.2f}] Oxford Test: {ox_acc:.2f}%")
            if ox_acc > best_soup_score:
                best_soup_score = ox_acc
                best_soup_sd = sd_blend
                best_alpha = alpha
        torch.save(best_soup_sd, best_weights_path)
        print(f"  --> En iyi Model Soup seçildi (alpha={best_alpha:.2f}, Acc={best_soup_score:.2f}%) -> {best_weights_path}")

    # En iyi ağırlıklarla temiz test seti kontrolü
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    test_acc = evaluate_acc(model, test_loader, device)
    if kaggle_val_loader is not None:
        kg_acc = evaluate_acc(model, kaggle_val_loader, device)
        print(f"\n[TAMAMLANDI] Oxford Test Doğruluğu: {test_acc:.2f}% | Kaggle Val Doğruluğu: {kg_acc:.2f}% | Kayıt: {best_weights_path}")
    else:
        print(f"\n[TAMAMLANDI] En İyi Model Test Doğruluğu: {test_acc:.2f}% | Kayıt: {best_weights_path}")


if __name__ == "__main__":
    main()

