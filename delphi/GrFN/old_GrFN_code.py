class GroundedFunctionNetwork():
    def sobol_analysis(
        self, num_samples, prob_def, var_types=None
    ):
        def create_input_vector(name, vector):
            if var_types is None:
                return vector

            type_info = var_types[name]
            if type_info[0] != str:
                return vector

            if type_info[0] == str:
                (str1, str2) = type_info[1]
                return np.where(vector >= 0.5, str1, str2)
            else:
                raise ValueError(f"Unrecognized value type: {type_info[0]}")

        # Create an array of samples from the bounds supplied in prob_def
        start = time.clock()
        samples = saltelli.sample(
            prob_def, num_samples, calc_second_order=True
        )
        end = time.clock()
        sample_time = end - start

        # Create vectors of sample inputs to run through the model
        vectorized_sample_list = np.split(samples, samples.shape[1], axis=1)
        vectorized_input_samples = {
            name: create_input_vector(name, vector)
            for name, vector in zip(prob_def["names"], vectorized_sample_list)
        }

        # Produce model output and reshape for analysis
        outputs = self.run(vectorized_input_samples)
        Y = outputs[0]
        Y = Y.reshape((Y.shape[0],))

        # Recover the sensitivity indices from the sampled outputs
        start = time.clock()
        S = sobol.analyze(prob_def, Y)
        end = time.clock()
        analyze_time = end - start

        return S, sample_time, analyze_time

    def S2_surface(
        self, sizes, bounds, presets, num_samples=10
    ):
        """Calculates the sensitivity surface of a GrFN for the two variables with
        the highest S2 index.

        Args:
            num_samples: Number of samples for sensitivity analysis.
            sizes: Tuple of (number of x inputs, number of y inputs).
            bounds: Set of bounds for GrFN inputs.
            presets: Set of standard values for GrFN inputs.

        Returns:
            Tuple:
                Tuple: The names of the two variables that were selected
                Tuple: The X, Y vectors of eval values
                Z: The numpy matrix of output evaluations

        """
        args = self.inputs
        Si = self.sobol_analysis(
            num_samples,
            {
                "num_vars": len(args),
                "names": args,
                "bounds": [bounds[arg] for arg in args],
            },
        )
        S2 = Si["S2"]
        (s2_max, v1, v2) = get_max_s2_sensitivity(S2)

        x_var, y_var = args[v1], args[v2]
        x_bounds, y_bounds = bounds[x_var], bounds[y_var]
        (x_sz, y_sz) = sizes
        X = np.linspace(*x_bounds, x_sz)
        Y = np.linspace(*y_bounds, y_sz)
        Xv, Yv = np.meshgrid(X, Y)

        input_vectors = {
            arg: np.full_like(Xv, presets[arg])
            for i, arg in enumerate(args)
            if i != v1 and i != v2
        }
        input_vectors.update({x_var: Xv, y_var: Yv})

        outputs = self.run(input_vectors)
        Z = outputs[0]

        return X, Y, Z, x_var, y_var


class ForwardInfluenceBlanket():
    def sobol_analysis(
        self, num_samples, prob_def, covers, var_types=None
    ):
        samples = saltelli.sample(
            prob_def, num_samples, calc_second_order=True
        )

        Y = np.zeros(samples.shape[0])
        for i, sample in enumerate(samples):
            values = {n: val for n, val in zip(prob_def["names"], sample)}
            Y[i] = self.run(values, covers)

        return sobol.analyze(prob_def, Y)

    def S2_surface(self, sizes, bounds, presets, covers,
            num_samples = 10):
        """Calculates the sensitivity surface of a GrFN for the two variables with
        the highest S2 index.

        Args:
            num_samples: Number of samples for sensitivity analysis.
            sizes: Tuple of (number of x inputs, number of y inputs).
            bounds: Set of bounds for GrFN inputs.
            presets: Set of standard values for GrFN inputs.

        Returns:
            Tuple:
                Tuple: The names of the two variables that were selected
                Tuple: The X, Y vectors of eval values
                Z: The numpy matrix of output evaluations

        """
        args = self.inputs
        Si = self.sobol_analysis(
            num_samples,
            {
                "num_vars": len(args),
                "names": args,
                "bounds": [bounds[arg] for arg in args],
            },
            covers
        )
        S2 = Si["S2"]
        (s2_max, v1, v2) = get_max_s2_sensitivity(S2)

        x_var = args[v1]
        y_var = args[v2]
        search_space = [(x_var, bounds[x_var]), (y_var, bounds[y_var])]
        preset_vals = {
            arg: presets[arg]
            for i, arg in enumerate(args)
            if i != v1 and i != v2
        }

        X = np.linspace(*search_space[0][1], sizes[0])
        Y = np.linspace(*search_space[1][1], sizes[1])

        Xm, Ym = np.meshgrid(X, Y)
        Z = np.zeros((len(X), len(Y)))
        for x, y in itertools.product(range(len(X)), range(len(Y))):
            inputs = {n: v for n, v in presets.items()}
            inputs.update({search_space[0][0]: x, search_space[1][0]: y})
            Z[x][y] = self.run(inputs, covers)

        return X, Y, Z, x_var, y_var
