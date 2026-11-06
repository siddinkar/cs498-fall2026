"""Fused confidence accumulation preserving FP64 multiply/add then FP32 store."""
import triton
import triton.language as tl


@triton.jit
def _accumulate(acc,samples,rows,weights,N:tl.constexpr,C:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=index<N*C;row=index//C;channel=index%C
    target=tl.load(rows+row,row<N,0)*C+channel
    previous=tl.load(acc+target,valid,0).to(tl.float64)
    sample=tl.load(samples+index,valid,0).to(tl.float64)
    weight=tl.load(weights+row,row<N,0)
    # Do not contract the multiply and add: NumPy/PyTorch round them separately.
    result=previous+sample*weight
    tl.store(acc+target,result.to(tl.float32),valid)


def accumulate(acc,samples,rows,weights):
    if not samples.is_contiguous() or not acc.is_contiguous():raise ValueError('Accumulation inputs must be contiguous')
    if samples.shape[0]!=len(rows) or weights.numel()!=len(rows):raise ValueError('Accumulation row count mismatch')
    if not len(rows):return
    _accumulate[(triton.cdiv(samples.numel(),1024),)](acc,samples,rows,weights,len(rows),samples.shape[1],1024,enable_fp_fusion=False)
