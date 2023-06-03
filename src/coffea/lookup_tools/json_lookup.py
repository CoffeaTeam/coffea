class json_lookup:
    def __init__(self, wrapped_values):
        self.values = wrapped_values

    def __call__(self, run, lumi):
        mu = []
        for i in range(len(run)):
            run_ = str(run[i])
            lumi_ = lumi[i]
            if run_ not in self.values.keys() or lumi_ not in self.values[run_].keys():
                print(f"    \033[91m run:lumi {run_}:{lumi_} not found \033[00m")
            else:
                mu.append(self.values[run_][lumi_])
        return mu
