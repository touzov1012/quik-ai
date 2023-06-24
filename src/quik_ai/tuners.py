import numpy as np

try:
    import sklearn
    import sklearn.gaussian_process
except ImportError:  # pragma: no cover
    sklearn = None  # pragma: no cover

try:
    import scipy
    import scipy.optimize
except ImportError:  # pragma: no cover
    scipy = None  # pragma: no cover

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import tuner as tuner_module

from keras_tuner.tuners import hyperband as hyperband_module
from keras_tuner.tuners import bayesian as bayesian_module

class BOHBOracle(hyperband_module.HyperbandOracle):

    def __init__(
        self,
        objective=None,
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        num_initial_points=None,
        random_chance=0.2,
        alpha=1e-3,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=8,
    ):
        super().__init__(
            objective=objective,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        self.num_initial_points = num_initial_points
        self.random_chance = random_chance
        self.alpha = alpha
        self.beta = beta
        self._random_state = np.random.RandomState(self.seed)
        self.gpr = self._make_gpr()
    
    def _make_gpr(self):
        return sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=sklearn.gaussian_process.kernels.Matern(nu=2.5),
            n_restarts_optimizer=20,
            normalize_y=True,
            alpha=self.alpha,
            random_state=self.seed,
        )
    
    def _nonfixed_space(self):
        return [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed)
        ]
    
    def _process_candidate_x(self, x):
        x = np.nan_to_num(x, posinf=0, neginf=0)
        x = [min(max(xi,0.0),1.0) for xi in x]
        return x
    
    def _vectorize_trials(self, candidates):
        x = []
        y = []
        ongoing_trials = set(self.ongoing_trials.values())
        for trial in candidates:
            # Create a vector representation of each Trial's hyperparameters.
            trial_hps = trial.hyperparameters
            vector = []
            for hp in self._nonfixed_space():
                # For hyperparameters not present in the trial (either added after
                # the trial or inactive in the trial), set to default value.
                if (
                    trial_hps.is_active(hp)  # inactive
                    and hp.name in trial_hps.values  # added after the trial
                ):
                    trial_value = trial_hps.values[hp.name]
                else:
                    trial_value = hp.default

                # Embed an HP value into the continuous space [0, 1].
                prob = hp.value_to_prob(trial_value)
                vector.append(prob)

            if trial in ongoing_trials:
                # "Hallucinate" the results of ongoing trials. This ensures that
                # repeat trials are not selected when running distributed.
                x_h = np.array(vector).reshape((1, -1))
                y_h_mean, y_h_std = self.gpr.predict(x_h, return_std=True)
                # Give a pessimistic estimate of the ongoing trial.
                score = y_h_mean[0] + y_h_std[0]
            elif trial.status == "COMPLETED":
                score = trial.score
                # Always frame the optimization as a minimization for scipy.minimize.
                if self.objective.direction == "max":
                    score = -1 * score
            else:
                # Skip the failed and invalid trials.
                continue

            x.append(vector)
            y.append(score)

        x = np.array(x)
        y = np.array(y)
        return x, y
    
    def _random_values(self):
        # If no hyperparameters exist, exit this run
        dimensions = len(self.hyperparameters.space)
        if dimensions == 0:
            print('No hyperparameters found')
            return super()._random_values()
        
        # get the current bracket
        trial_number = len(self.trials) + 1
        c_bracket = self._current_bracket
        c_round = 0
        c_epochs = self._get_epochs(c_bracket, c_round)
        
        # find which trials we can use for bayes optimization
        used_for_bayes = [
            t for t in self.trials.values() if t.status == "COMPLETED" and t.hyperparameters.values["tuner/epochs"] == c_epochs
        ]
        
        # make sure we have enough points to use for bayes optimization
        num_initial_points = self.num_initial_points or 3 * dimensions
        num_bayes = len(used_for_bayes)
        if num_bayes < num_initial_points or self._random_state.uniform() <= self.random_chance:
            print('Trial #%s start type: Random' % trial_number)
            if num_bayes < num_initial_points:
                print('Completed %s / %s warm-ups for Bayesian' % (num_bayes, num_initial_points))
            return super()._random_values()

        # Fit a GPR to the completed trials and return the predicted optimum values.
        x, y = self._vectorize_trials(used_for_bayes)
        
        # Ensure no nan, inf in x, y. GPR cannot process nan or inf.
        keep_idx = np.isfinite(x).all(axis=1) & np.isfinite(y)
        num_before_filter = num_bayes
        x = x[keep_idx]
        y = y[keep_idx]
        num_bayes = len(y)
        
        # if after filtering nans we no longer have enough for bayes, return random
        if num_bayes < num_initial_points:
            print('Insufficient trials for Bayesian after %s NaN removed ...' % (num_before_filter - num_bayes))
            print('Trial #%s start type: Random' % trial_number)
            return super()._random_values()
        
        # Print process
        print('Trial #%s start type: Bayesian' % trial_number)
        print('Total samples for Bayesian: %s' % num_bayes)
        
        self.gpr.fit(x, y)

        def _upper_confidence_bound(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            return mu - self.beta * sigma

        optimal_val = float("inf")
        optimal_x = None
        num_restarts = 50
        bounds = bayesian_module.BayesianOptimizationOracle._get_hp_bounds(self)
        x_seeds = self._random_state.uniform(
            bounds[:, 0], bounds[:, 1], size=(num_restarts, bounds.shape[0])
        )
        for x_try in x_seeds:
            # Sign of score is flipped when maximizing.
            result = scipy.optimize.minimize(
                _upper_confidence_bound, x0=x_try, bounds=bounds, method="L-BFGS-B"
            )
            result_fun = result.fun if np.isscalar(result.fun) else result.fun[0]
            if result_fun < optimal_val:
                optimal_val = result_fun
                optimal_x = result.x
        
        # process the result to correct bounds
        optimal_x = self._process_candidate_x(optimal_x)
        
        return bayesian_module.BayesianOptimizationOracle._vector_to_values(self, optimal_x)

class BOHB(tuner_module.Tuner):

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        num_initial_points=None,
        random_chance=0.2,
        alpha=1e-3,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=8,
        **kwargs
    ):
        oracle = BOHBOracle(
            objective,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            num_initial_points=num_initial_points,
            random_chance=random_chance,
            alpha=alpha,
            beta=beta,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
        if "tuner/epochs" in hp.values:
            fit_kwargs["epochs"] = hp.values["tuner/epochs"]
            fit_kwargs["initial_epoch"] = hp.values["tuner/initial_epoch"]
        return super().run_trial(trial, *fit_args, **fit_kwargs)

    def _build_hypermodel(self, hp):
        model = super()._build_hypermodel(hp)
        if "tuner/trial_id" in hp.values:
            trial_id = hp.values["tuner/trial_id"]
            # Load best checkpoint from this trial.
            model.load_weights(self._get_checkpoint_fname(trial_id))
        return model