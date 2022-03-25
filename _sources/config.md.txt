# The Delphi config file
Runtime settable parameters for Model training

### File location:
delphi/data/config.json

### Fields:
```json
{
  "config_version": "1.0.0",
  "training_stopping_min_delta": 0.1,
  "training_stopping_n_samples": 200,
  "training_burn": 1000,
}
```

**config_version**
The version of this config file.  

**training_stopping_min_delta**
The difference in log_likelihood values used to determine whether the sample burning phase of Model training should stop early

**training_stopping_n_samples**
The number of samples to test for early stopping of the sample burn phase of Model trainng.  As each log_likelihood value is generated, the early stopping algorithm finds the min and max log_likelihood for this number of previous log_likelihoods including the current sample.  If the magnitude of the difference between min and max is less than or equal to the training_stopping_min_delta then the log_likelihood is said to have converged, and sample burning is stopped.

**training_burn**
The maximum number of samples to burn out during Model Training.  This value is used if the 'DELPHI_N_SAMPLES' environment variable is not set.  
