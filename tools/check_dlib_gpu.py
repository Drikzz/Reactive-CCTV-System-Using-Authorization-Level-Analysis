import os
try:
    import dlib
except Exception as e:
    dlib = None
    print("dlib import failed:", e)
try:
    import torch
except Exception as e:
    torch = None
    print("torch import failed:", e)

print("PWD:", os.getcwd())
if dlib is not None:
    print("dlib DLIB_USE_CUDA:", getattr(dlib, "DLIB_USE_CUDA", None))
    print("dlib.cuda available:", hasattr(dlib, "cuda"))
    try:
        n = dlib.cuda.get_num_devices() if hasattr(dlib, "cuda") else 0
        print("dlib.cuda.get_num_devices():", n)
    except Exception as e:
        print("dlib.cuda query error:", e)
else:
    print("dlib: not available")

if torch is not None:
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("torch.cuda.current_device():", torch.cuda.current_device())
        try:
            print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
        except Exception:
            pass
else:
    print("torch: not available")

# quick test: attempt small dlib GPU op if available (non-destructive)
if dlib is not None and hasattr(dlib, "cuda") and dlib.cuda.get_num_devices() > 0:
    try:
        print("Attempting a trivial dlib.cuda operation...")
        _ = dlib.cuda.get_device_name(0)
        print("dlib.cuda.get_device_name(0) ok")
    except Exception as e:
        print("dlib.cuda test failed:", e)