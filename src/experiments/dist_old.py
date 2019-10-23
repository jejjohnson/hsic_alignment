class DistributionExp:
    def __init__(
        self,
        seed=123,
        n_gamma=100,
        factor=1,
        sigma_est="median",
        save_path=None,
        save_name="scale_test",
    ):

        # fixed experimental params
        self.seed = seed
        self.rng_x = check_random_state(seed)
        self.rng_y = check_random_state(seed + 1)
        self.n_gamma = n_gamma
        self.factor = factor
        self.sigma_est = sigma_est
        self.save_path = save_path
        self.save_name = save_name

        self.d_dimensions = [2, 3, 10, 50, 100]
        # free experimental params
        self.scorers = ["hsic", "tka", "ctka"]
        self.datasets = ["gauss", "tstudent"]
        self.gamma_initializations = [
            "max",
            "silverman",
            "median",
            "belkin20",
            "belkin40",
            "belkin60",
            "belkin80",
            "scott",
        ]
        self.nus = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 10,
            "7": 15,
            "8": 20,
            "9": 30,
        }
        self.stds = {
            "1": 0.0,
            "2": 0.1,
            "3": 0.2,
            "4": 0.3,
            "5": 0.4,
            "6": 0.5,
            "7": 0.6,
            "8": 0.7,
            "9": 0.8,
            "10": 0.9,
            "11": 1,
        }
        self.trials = [1, 2, 3, 4, 5]
        self.n_samples = [
            50,
            100,
            500,
            1_000,
            5_000,
            # 10_000,
            # 30_000,
            # 50_000
        ]
        # saved dataframe

        pass

    def run_experiment(self):

        # initialize results dataframe
        self.results_df = self.generate_results_df()

        # Loop through datasets
        for idataset in self.datasets:
            print(f"Function: {idataset}")

            # Loop through samples
            for isample in self.n_samples:
                for idimension in self.d_dimensions:

                    # Loop through random seeds
                    for itrial in self.trials:

                        for igamma_method in self.gamma_initializations:
                            if idataset == "tstudent":

                                # GET MI DATA
                                mi_loader = MIData(idataset)

                                for inu in self.nus.items():

                                    X, Y, mi_value = mi_loader.get_data(
                                        samples=isample,
                                        dimensions=idimension,
                                        std=None,
                                        trial=itrial,
                                        nu=inu[0],
                                    )

                                    # estimate initial gamma
                                    sigma_init = self._get_init_sigmas(
                                        X, Y, method=igamma_method
                                    )

                                    # convert sigma to gamma
                                    gamma_init = sigma_to_gamma(sigma_init)

                                    # Loop through HSIC scoring methods
                                    for hsic_method in self.scorers:

                                        # =======================
                                        # HSIC MEASURES
                                        # =======================

                                        # Calculate HSIC
                                        hsic_score, gamma = self._get_hsic(
                                            X, Y, hsic_method, gamma_init
                                        )

                                        # append results to results dataframe
                                        self.results_df = self.append_results(
                                            results_df=self.results_df,
                                            dataset=idataset,
                                            trial=itrial,
                                            n_samples=isample,
                                            d_dimensions=idimension,
                                            gamma_init=igamma_method,
                                            gamma=gamma,
                                            nu=inu[1],
                                            std=np.nan,
                                            hsic_method=hsic_method,
                                            hsic_score=hsic_score,
                                            mi_score=mi_value,
                                        )

                                        # save results to csv
                                        self.save_data(self.results_df)

                            elif idataset == "gauss":

                                # Load the MI dataset
                                mi_loader = MIData(idataset)

                                for istd in self.stds.items():

                                    X, Y, mi_score = mi_loader.get_data(
                                        samples=isample,
                                        dimensions=idimension,
                                        std=istd[0],
                                        trial=itrial,
                                        nu=None,
                                    )

                                    # estimate initial sigmas and save
                                    gamma_init = self._get_init_sigmas(
                                        X, Y, method=igamma_method
                                    )

                                    # =======================
                                    # HSIC MEASURES
                                    # =======================

                                    # Loop through HSIC scoring methods
                                    for hsic_method in self.scorers:

                                        # =======================
                                        # HSIC MEASURES
                                        # =======================

                                        # Calculate HSIC
                                        hsic_score, gamma = self._get_hsic(
                                            X, Y, hsic_method, gamma_init
                                        )

                                        # append results to results dataframe
                                        self.results_df = self.append_results(
                                            results_df=self.results_df,
                                            dataset=idataset,
                                            trial=itrial,
                                            n_samples=isample,
                                            d_dimensions=idimension,
                                            gamma_init=igamma_method,
                                            nu=np.nan,
                                            std=istd[1],
                                            gamma=gamma,
                                            hsic_method=hsic_method,
                                            hsic_score=hsic_score,
                                            mi_score=mi_score,
                                        )

                                        # save results to csv
                                        self.save_data(self.results_df)

                            else:
                                raise ValueError(f"Unrecognized dataset: {idataset}")

        return self

    def _get_data(self, dataset, info_meas, dimensions, samples, trials, nu):

        # initialize dataset Generator
        clf_rbigdata = RBIGData(dataset=dataset, info_meas=info_meas)

        data = clf_rbigdata.get_data(
            d_dimensions=dimensions, n_samples=samples, t_trials=trials, nu=nu
        )
        return data["X"], data["Y"]

    def _get_init_sigmas(self, X, Y, method=None):

        # check override for sigma estimator
        if method in ["belkin20", "belkin40", "belkin60", "belkin80"]:
            percent = float(method[-2:]) / 100
            method = "belkin"

            # estimate initialize sigma
            sigma_x = estimate_sigma(X, method=method, percent=percent)
            sigma_y = estimate_sigma(Y, method=method, percent=percent)

        elif method is None or method == "max":

            method = self.sigma_est
            # estimate initialize sigma
            sigma_x = estimate_sigma(X, method=method)
            sigma_y = estimate_sigma(Y, method=method)

        elif method in ["silverman", "scott", "median", "mean"]:

            sigma_x = estimate_sigma(X, method=method)
            sigma_y = estimate_sigma(Y, method=method)

        else:
            raise ValueError(f"Unrecognized method: {method}")

        # init overall sigma is mean between two
        init_sigma = np.mean([sigma_x, sigma_y])

        return init_sigma

    def _get_hsic(self, X, Y, scorer, init_gamma=None):

        if scorer == "max":
            # cross validated HSIC method for maximization
            clf_hsic = train_rbf_hsic(
                X,
                Y,
                scorer,
                n_gamma=self.n_gamma,
                factor=self.factor,
                sigma_est=self.sigma_est,
                verbose=0,
                n_jobs=-1,
                cv=2,
            )
        else:
            # standard HSIC method without maximization
            clf_hsic = HSIC(
                gamma=init_gamma, kernel="rbf", scorer=scorer, subsample=None, bias=True
            )

            clf_hsic.fit(X, Y)

        # hsic value and kernel alignment score
        return clf_hsic.hsic_value, clf_hsic.gamma

    def generate_results_df(self):
        return pd.DataFrame(
            columns=[
                "dataset",
                "trial",
                "n_samples",
                "d_dimensions",
                "nu",
                "std",
                "gamma_init",
                "gamma",
                "scorer",
                "value",
                "mi_score",
            ]
        )

    def append_results(
        self,
        results_df,
        dataset,
        trial,
        n_samples,
        d_dimensions,
        std,
        nu,
        gamma,
        gamma_init,
        hsic_method,
        hsic_score,
        mi_score,
    ):
        # append data
        return results_df.append(
            {
                "dataset": dataset,
                "trial": trial,
                "n_samples": n_samples,
                "d_dimensions": d_dimensions,
                "nu": nu,
                "std": std,
                "gamma_init": gamma_init,
                "gamma": gamma,
                "scorer": hsic_method,
                "value": hsic_score,
                "mi_score": mi_score,
            },
            ignore_index=True,
        )

    def load_data(self):
        pass
