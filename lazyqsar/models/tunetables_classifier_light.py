from tunetables_light.scripts.transformer_prediction_interface import TuneTablesClassifierLight as TuneTablesClassifierLightBase

class TuneTablesClassifierLight(TuneTablesClassifierLightBase):
    def __init__(self, lr=0.1, epoch=7, device="cpu"):
        super().__init__(lr=lr, epoch=epoch, device=device)