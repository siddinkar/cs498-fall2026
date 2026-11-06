"""Lossless NumPy conditioning archives with zero-decompression feature reads.

ZIP_STORED keeps conditioning.npz compatible with np.load and allows large NPY
members to be memory mapped. Older compressed archives remain readable.
"""
import os
from pathlib import Path
import struct
import zipfile
import numpy as np


def save_conditioning(path,**arrays):
    path=Path(path);temporary=path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez(temporary,**arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_feature(path,key):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        member=archive.getinfo(key+'.npy')
        if member.compress_type==zipfile.ZIP_STORED:
            with path.open('rb') as stream:
                stream.seek(member.header_offset+26)
                name_bytes,extra_bytes=struct.unpack('<HH',stream.read(4))
                stream.seek(member.header_offset+30+name_bytes+extra_bytes)
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError(f'Unsupported NPY version: {version}')
                if dtype.hasobject:raise ValueError('Object feature arrays are forbidden')
                return np.memmap(path,mode='c',offset=stream.tell(),dtype=dtype,shape=shape,order='F' if fortran else 'C')
    with np.load(path,allow_pickle=False) as data:return data[key]
