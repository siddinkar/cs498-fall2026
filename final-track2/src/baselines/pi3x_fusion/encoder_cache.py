"""Reuse independent RGB backbone features for the unchanged Pi3X anchor views."""
import torch


class AnchorEncoderCache(torch.nn.Module):
    def __init__(self,native,anchors):
        super().__init__();self.native=native;self.anchors=set(anchors);self.cached={};self.indices=[]

    def forward(self,images,**kwargs):
        if len(images)!=len(self.indices):raise ValueError('Pi3X frame IDs and backbone batch differ')
        pending=[(row,identity) for row,identity in enumerate(self.indices) if identity not in self.cached]
        fresh={}
        if pending:
            values=self.native(images[[row for row,_ in pending]],**kwargs)['x_norm_patchtokens']
            for row,(_,identity) in enumerate(pending):
                fresh[identity]=values[row]
                if identity in self.anchors:self.cached[identity]=values[row].clone()
        return {'x_norm_patchtokens':torch.stack([self.cached[identity] if identity in self.cached else fresh[identity] for identity in self.indices])}
