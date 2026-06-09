"""Isolate the spconv SIGFPE: does a single strided SparseConv2d crash, and does
forcing ConvAlgo.Native fix it? Run on the GPU node.

    python tools/_spconv_algo_probe.py
"""
import torch
import spconv.pytorch as spconv
from spconv.core import ConvAlgo

dev = torch.device("cuda")
H = W = 64
n = 500
torch.manual_seed(0)
lin = torch.randperm(H * W)[:n]
x = (lin % W).int()
y = (lin // W).int()
indices = torch.stack([torch.zeros(n, dtype=torch.int32), y, x], dim=1).cuda()
feats = torch.randn(n, 8, device=dev)

def run(algo, tag):
    st = spconv.SparseConvTensor(feats, indices, [H, W], batch_size=1)
    conv = spconv.SparseConv2d(8, 16, kernel_size=3, stride=2, padding=1,
                               bias=False, indice_key="sp", algo=algo).to(dev)
    out = conv(st)
    torch.cuda.synchronize()
    print(f"[{tag}] OK  out_sites={out.features.shape[0]}")

print("spconv:", spconv.__version__, "| gpu:", torch.cuda.get_device_name(0),
      "| cap:", torch.cuda.get_device_capability(0))
for algo, tag in [(ConvAlgo.Native, "Native"),
                  (ConvAlgo.MaskImplicitGemm, "MaskImplicitGemm"),
                  (ConvAlgo.MaskSplitImplicitGemm, "MaskSplitImplicitGemm")]:
    try:
        run(algo, tag)
    except Exception as e:
        print(f"[{tag}] raised {type(e).__name__}: {e}")
    # NOTE: a SIGFPE will hard-crash the process here (not catchable); the last
    # printed [tag] line tells you which algo died.
