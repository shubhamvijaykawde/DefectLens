"""Print the versions and capabilities Phase 0b requires. Exit 1 if a hard requirement fails."""

from __future__ import annotations

import sys


def main() -> int:
    ok = True
    import torch
    import torchvision

    print(f"torch        {torch.__version__}")
    print(f"torchvision  {torchvision.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA avail   {cuda}")
    if cuda:
        print(f"GPU          {torch.cuda.get_device_name(0)}")
    ok &= cuda

    try:
        import onnxscript

        print(f"onnxscript   {onnxscript.__version__}")
    except ImportError as exc:
        print(f"onnxscript   MISSING ({exc})")
        ok = False

    import onnxruntime as ort

    providers = ort.get_available_providers()
    print(f"ORT          {ort.__version__}  providers={providers}")
    ok &= "CUDAExecutionProvider" in providers

    import anomalib
    import transformers

    print(f"anomalib     {anomalib.__version__}")
    print(f"transformers {transformers.__version__}")
    print("RESULT       " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
