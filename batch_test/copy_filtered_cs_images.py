import os
import shutil
import pandas as pd

# ---------------------------------------------------------
# MODIFY THESE PATHS
# ---------------------------------------------------------
clean_csv = "batch_test/test_cs_images.csv"   # your manually cleaned CSV
src_folder = "batch_test/test_cs_images"        # folder where images currently exist
dst_folder = "batch_test/batch_test_images"           # new folder where you want to copy the images
# ---------------------------------------------------------

# Create new folder
os.makedirs(dst_folder, exist_ok=True)

# Read your cleaned CSV
df = pd.read_csv(clean_csv)

print(f"Total images listed in CSV: {len(df)}")

# Copy each image
missing = []
copied = 0

for fname in df["img_fName"]:      # column name MUST match your Excel column
    src_path = os.path.join(src_folder, fname)
    dst_path = os.path.join(dst_folder, fname)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied += 1
    else:
        missing.append(fname)

print(f"\n✔ Copied: {copied} images")
print(f"⚠ Missing: {len(missing)} images")

if missing:
    print("List of missing files:")
    for m in missing:
        print(" -", m)

# OPTIONAL — Create new CSV with updated paths
df["filepath"] = df["img_fName"].apply(lambda x: os.path.join(dst_folder, x))
df.to_csv("batch_test_images_list.csv", index=False)
print("\n📄 Saved updated CSV → batch_test_images_list.csv")
