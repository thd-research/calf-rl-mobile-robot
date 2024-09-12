from regelum import ScipyOptimizerConfig

class TestScipyOptimizerConfig(ScipyOptimizerConfig):
    def __init__(self):
        super().__init__(
            kind="numeric",
            opt_method="SLSQP",
            opt_options={
                'maxiter': 40, 
                'maxfev': 60, 
                'disp': False, 
                'adaptive': True, 
                'xatol': 1e-3, 
                'fatol': 1e-3
                },
            config_options=None
        )
