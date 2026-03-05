import os

print("\n=== SCANNING FOR SAFETENSORS FILES ===\n")

search_roots = ["/workspace", "/runpod-volume", "/app/models"]

for root in search_roots:
    if not os.path.exists(root):
        print(f"[SKIP] {root} does not exist")
        continue
    print(f"\n[SCAN] {root}")
    print("-" * 50)
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip huge irrelevant dirs
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "node_modules")]
        for fname in filenames:
            if fname.endswith(".safetensors") or fname.endswith(".ckpt") or fname.endswith(".bin"):
                full = os.path.join(dirpath, fname)
                size_gb = os.path.getsize(full) / (1024**3)
                print(f"  {full}  ({size_gb:.2f} GB)")

print("\n=== DIRECTORY TREE (top 3 levels of /workspace) ===\n")
for dirpath, dirnames, filenames in os.walk("/workspace"):
    depth = dirpath.replace("/workspace", "").count(os.sep)
    if depth > 3:
        dirnames[:] = []
        continue
    indent = "  " * depth
    print(f"{indent}{os.path.basename(dirpath)}/")
    if depth == 3:
        dirnames[:] = []

print("\n=== ENV VARS ===\n")
for k, v in sorted(os.environ.items()):
    if any(x in k.upper() for x in ["PATH", "HOME", "CACHE", "MODEL", "HF", "COMFY", "RUNPOD"]):
        print(f"  {k}={v}")
