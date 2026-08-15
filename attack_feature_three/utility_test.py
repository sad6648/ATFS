"""
Utility Test: Evaluate face detection and recognition on original vs.
ATFS-protected images.

This script tests whether ATFS-protected images retain benign utility for
standard discriminative face tasks (detection and recognition), which operate
on different feature spaces than the generative models ATFS targets.

Usage:
    python utility_test.py --original_dir ./data/img --protected_dir ./output

Requirements:
    facenet-pytorch (already in requirements.txt)
    MTCNN for face detection
    InceptionResnetV1 (VGGFace2) for face recognition
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from facenet_pytorch import MTCNN, InceptionResnetV1


def compute_cosine_similarity(tensor_a, tensor_b):
    """Compute cosine similarity between two embedding tensors."""
    return float(
        torch.nn.functional.cosine_similarity(
            tensor_a.unsqueeze(0), tensor_b.unsqueeze(0)
        ).item()
    )


def load_image(path):
    """Load an image and convert to RGB."""
    return Image.open(path).convert("RGB")


def get_image_files(directory):
    """Get sorted list of image files in a directory."""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith(valid_extensions)
    ]
    files.sort()
    return files


def match_files(original_dir, protected_dir):
    """Match image files between original and protected directories by name."""
    orig_files = get_image_files(original_dir)
    prot_files = get_image_files(protected_dir)

    # Try exact name match first
    matched = []
    for orig in orig_files:
        if orig in prot_files:
            matched.append((orig, orig))
        else:
            # Try matching by index if names differ
            orig_idx = orig_files.index(orig)
            if orig_idx < len(prot_files):
                matched.append((orig, prot_files[orig_idx]))

    return matched


def run_utility_test(original_dir, protected_dir, device="cpu", output_file="utility_results.json"):
    """Run face detection and recognition tests on original vs protected images."""
    print("[Start] Utility Test: MTCNN detection + FaceNet recognition")
    print(f"  Original dir:  {original_dir}")
    print(f"  Protected dir: {protected_dir}")
    print(f"  Device: {device}")

    # Initialize models
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        post_process=True,
        device=device,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Match files
    matched = match_files(original_dir, protected_dir)
    if not matched:
        print("[Error] No matching image pairs found.")
        return None

    print(f"  Matched pairs: {len(matched)}")

    # Results storage
    results = {
        "total_pairs": len(matched),
        "detection": {
            "original_detected": 0,
            "protected_detected": 0,
            "original_rate": 0.0,
            "protected_rate": 0.0,
        },
        "recognition": {
            "cosine_similarities": [],
            "mean_cosine": 0.0,
            "std_cosine": 0.0,
            "min_cosine": 0.0,
            "max_cosine": 0.0,
        },
        "per_image": [],
    }

    for orig_name, prot_name in tqdm(matched, desc="Testing"):
        orig_path = os.path.join(original_dir, orig_name)
        prot_path = os.path.join(protected_dir, prot_name)

        orig_img = load_image(orig_path)
        prot_img = load_image(prot_path)

        entry = {"original": orig_name, "protected": prot_name}

        # --- Face Detection ---
        orig_faces = mtcnn(orig_img)
        prot_faces = mtcnn(prot_img)

        orig_detected = orig_faces is not None
        prot_detected = prot_faces is not None

        if orig_detected:
            results["detection"]["original_detected"] += 1
        if prot_detected:
            results["detection"]["protected_detected"] += 1

        entry["orig_detected"] = orig_detected
        entry["prot_detected"] = prot_detected

        # --- Face Recognition ---
        if orig_detected and prot_detected:
            # Handle single face (tensor shape: [3, 160, 160])
            if orig_faces.dim() == 3:
                orig_faces = orig_faces.unsqueeze(0)
            if prot_faces.dim() == 3:
                prot_faces = prot_faces.unsqueeze(0)

            orig_faces = orig_faces.to(device)
            prot_faces = prot_faces.to(device)

            with torch.no_grad():
                orig_embedding = resnet(orig_faces)
                prot_embedding = resnet(prot_faces)

            cos_sim = compute_cosine_similarity(
                orig_embedding.squeeze(0), prot_embedding.squeeze(0)
            )
            results["recognition"]["cosine_similarities"].append(cos_sim)
            entry["cosine_similarity"] = cos_sim

        results["per_image"].append(entry)

    # --- Summary ---
    total = results["total_pairs"]
    results["detection"]["original_rate"] = (
        results["detection"]["original_detected"] / total if total > 0 else 0.0
    )
    results["detection"]["protected_rate"] = (
        results["detection"]["protected_detected"] / total if total > 0 else 0.0
    )

    cos_sims = results["recognition"]["cosine_similarities"]
    if cos_sims:
        results["recognition"]["mean_cosine"] = float(np.mean(cos_sims))
        results["recognition"]["std_cosine"] = float(np.std(cos_sims))
        results["recognition"]["min_cosine"] = float(np.min(cos_sims))
        results["recognition"]["max_cosine"] = float(np.max(cos_sims))

    # --- Print Results ---
    print("\n" + "=" * 50)
    print("Utility Test Results")
    print("=" * 50)
    print(f"Total image pairs: {total}")
    print()
    print("Face Detection (MTCNN):")
    print(
        f"  Original:  {results['detection']['original_detected']}/{total} "
        f"({results['detection']['original_rate']:.4f})"
    )
    print(
        f"  Protected: {results['detection']['protected_detected']}/{total} "
        f"({results['detection']['protected_rate']:.4f})"
    )
    print()
    print("Face Recognition (FaceNet / VGGFace2):")
    if cos_sims:
        print(f"  Mean cosine similarity: {results['recognition']['mean_cosine']:.4f}")
        print(f"  Std:                    {results['recognition']['std_cosine']:.4f}")
        print(f"  Min:                    {results['recognition']['min_cosine']:.4f}")
        print(f"  Max:                    {results['recognition']['max_cosine']:.4f}")
    else:
        print("  No valid pairs for recognition test.")
    print("=" * 50)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")
    print("[Done] Utility Test complete.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility test: face detection and recognition on original vs ATFS-protected images"
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory of original (clean) images",
    )
    parser.add_argument(
        "--protected_dir",
        type=str,
        required=True,
        help="Directory of ATFS-protected images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="utility_results.json",
        help="Output JSON file for results (default: utility_results.json)",
    )
    args = parser.parse_args()

    run_utility_test(
        original_dir=args.original_dir,
        protected_dir=args.protected_dir,
        device=args.device,
        output_file=args.output_file,
    )
