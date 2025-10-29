import scipy.io

file_path = r"C:\Users\GAYATHRI\Downloads\AFIB2.mat"  # update path if needed
mat = scipy.io.loadmat(file_path)

print("🔍 Keys in this .mat file:")
for k, v in mat.items():
    if not k.startswith("__"):
        try:
            print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}")
        except Exception:
            print(f"{k}: {type(v)}")
