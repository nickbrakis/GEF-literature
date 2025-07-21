# DeepTSF: Codeless machine learning operations for time series forecasting

https://doi.org/10.1016/j.softx.2024.101758

Το DeepTSF αποτελεί μία καινοτόμα πλατφόρμα MLOps για πρόβλεψη χρονοσειρών, καθώς προσφέρει ένα φιλικό προς το χρήστη, χωρίς κώδικα, περιβάλλον. Αποδεικνύεται ήδη πολύτιμο στην βραχυπρόθεσμη πρόβλεψη ενέργειας με χρήση μεθόδων DL (Βαθιας Μάθησης) και χαράζει μια σαφή πορεία για επέκταση στην ενορχήστρωση, την εξυπηρέτηση και την ευρύτερη ενσωμάτωση ροών εργασίων ML.

### MLOps

Η διαδικασία MLOps ως ένα αναδυόμενο πεδίο στην τεχνολογία ΑΙ υπερβαίνει τις συμβατικές και χειροκίνητες διαδικασίες ανάπτυξης και χρήσης ML μοντέλων. Πιο συγκεκριμένα, εστιάζει στην αυτοματοποίηση όλου του κύκλου ζωής μίας ML εργασίας, επιβλέποντας τις διεργασίες συλλογής και προετοιμασίας δεδομένων, εκπαίδευσης, αξιολόγησης και εφαρμογής μοντέλων Μηχανικής Μάθησης. Ως αποτέλεσμα, αντικαθιστώντας τις ήδη χειροκίνητες διαδικασίες του συμβατικού κύκλου ζωής επιτυγχάνεται αποδοτικότερη συνεργασία μεταξύ των ομάδων και μειώνεται ο χρόνος που απαιτείται για την ανάπτυξη μιας ολοκληρωμένης ML εφαρμογής. 

### Αρχιτεκτονική DeepTSF

Βάση Δεδομένων:

- ΜοngoDB

Backend Μηχανή Προβλέψεων:

Είναι ο πυρήνας της εφαρμογής και περιέχει τις διαδικασίες

1. Εξαγωγής δεδομένων
2. Προ-επεξεργασία δεδομένων και προσαρμογή στις απιατήσεις εισόδου του μοντέλου
3. Εκπαίδευση και προσαρμογή υπερπαραμέτρων του μοντέλου
4. Αξιολόγηση μοντέλου
5. Αποθήκευση, Έκδοση και Εξυπηρέτηση 

Οι τεχνολογίες που εκτελούν τις παραπάνω εργασίες είναι:

- Python MLflow
- Python Darts
- Python Optuna
- Python SHAP
- Python FastAPI

Γραφική Διεπαφή Χρήστη

- React

Διαχείριση Ροής Εργασιών

- Βασική μέθοδος με χρήση CLI
- Μέθοδος για προχωρημένους μέσω Dagster

Αυθεντικοποίηση Χρήστη

- Keycloak

![image.png](attachment:dc8a2485-f02e-4517-b122-caa16a110ebf:image.png)

Dagster Config sample:  

```yaml
execution:
  config:
    backend: rpc://
resources:
  config:
    config:
      a: 0.3
      analyze_with_shap: false
      convert_to_local_tz: false
      country: PT
      cut_date_test: '20230315'
      cut_date_val: '20230101'
      darts_model: LightGBM
      database_name: rdn_load_data
      device: gpu
      eval_method: ts_ID
      eval_series: eval_series
      evaluate_all_ts: true
      experiment_name: GlobalForecastingEstonia
      forecast_horizon: 24
      format: short
      from_database: false
      future_covs_csv: None
      future_covs_uri: None
      grid_search: false
      hyperparams_entrypoint:
        lags: 168
      ignore_previous_runs: true
      imputation_method: none
      loss_function: mape
      m_mase: 24
      max_thr: -1
      min_non_nan_interval: 24
      multiple: true
      n_trials: 100
      num_samples: 1
      num_workers: 4
      opt_test: false
      order: 1
      parent_run_name: LightGBM_only_target
      past_covs_csv: None
      past_covs_uri: None
      pv_ensemble: false
      resampling_agg_method: averaging
      resolution: 1h
      retrain: false
      rmv_outliers: false
      scale: true
      scale_covs: true
      series_csv: dataset-storage/raw_series_preprocessed.csv
      series_uri: None
      shap_data_size: 100
      shap_input_length: -1
      std_dev: 4.5
      stride: -1
      test_end_date: None
      time_covs: false
      trial_name: Default
      ts_used_id: None
      wncutoff: 0.000694
      ycutoff: 3
      ydcutoff: 30
      year_range: None
```