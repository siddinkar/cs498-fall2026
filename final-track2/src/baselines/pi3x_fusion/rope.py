"""Exact fused fallback RoPE for the fixed calibrated image grid.

Preserves native BF16/FP32 product rounding and separate addition. The cosine
and sine tables are computed by Pi3X's own helper; no frequency approximation.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(X,P,CO,SI,Y,H:tl.constexpr,T:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,P0:tl.constexpr,P1:tl.constexpr,P2:tl.constexpr,MAX_POS:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<N
 c=i%D;t=i//D%T;h=i//(D*T)%H;b=i//(D*T*H)
 axis=c//(D//2);local=c%(D//2);second=local>=D//4
 partner=c+tl.where(second,-D//4,D//4)
 off=b*S0+h*S1+t*S2
 a=tl.load(X+off+c*S3,valid,0);v=tl.load(X+off+partner*S3,valid,0)
 pos=tl.load(P+b*P0+t*P1+axis*P2,valid,0)
 tl.device_assert(((pos>=0)&(pos<MAX_POS))|~valid,"Pi3X RoPE position exceeds fixed image grid")
 co=tl.load(CO+pos*(D//2)+local,valid,0);si=tl.load(SI+pos*(D//2)+local,valid,0)
 dtype=a.dtype
 first=(a.to(tl.float32)*co.to(tl.float32)).to(dtype).to(tl.float32)
 signed=tl.where(second,v.to(tl.float32),-v.to(tl.float32))
 last=(signed*si.to(tl.float32)).to(dtype).to(tl.float32)
 result=first+last;tl.store(Y+i,result.to(dtype),valid)
def apply(native,tokens,positions,max_position):
 out=torch.empty(tokens.shape,device=tokens.device,dtype=tokens.dtype)
 cos,sin=native.get_cos_sin(tokens.shape[-1]//2,max_position,tokens.device,tokens.dtype)
 rope_kernel[(triton.cdiv(tokens.numel(),1024),)](tokens,positions,cos,sin,out,*tokens.shape[1:],tokens.numel(),*tokens.stride(),*positions.stride(),max_position,1024,enable_fp_fusion=False,debug=True)
 return out


def install(model,resolution):
    native=model.rope
    if not hasattr(native,'get_cos_sin'):return False
    max_position=resolution//14+1
    # PositionGetter builds the same nonnegative 14-pixel image grid in every
    # block; patch positions are offset by one; register/metric tokens use zero. No per-block GPU -> CPU max sync.
    native.forward=lambda tokens,positions:apply(native,tokens,positions,max_position)
    return True
