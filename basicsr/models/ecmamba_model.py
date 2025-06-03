import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class ECMambaModel(SRModel):
    """MambaIR model for image restoration."""

    # test by partitioning
    def test(self):
        
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out = self.net_g_ema(self.lq)
                
        else:
            self.net_g.eval()
            with torch.no_grad():
                out = self.net_g(self.lq)
        self.output = out
