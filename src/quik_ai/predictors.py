from quik_ai import tuning

class Predictor(tuning.Tunable):
    def __init__(self, name, numeric=True, drop=False, **kwargs):
        super().__init__(name, **kwargs)
        self.numeric = numeric
        self.drop = drop
    
    
    def get_parameters(self, hp):
        config = super().get_parameters(hp)
        config.update({
            'drop' : self._get_hp(None, 'drop', hp)
        })
        return config