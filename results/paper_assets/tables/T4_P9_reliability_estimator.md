# Table 4 — P9 cross-cell reliability estimator

| cell            | model                   |   n_heldout_seeds |   mean_mae |      sd_mae |   mean_rmse |    sd_rmse |   mean_spearman_reliability_c10 |   sd_spearman_reliability_c10 |
|:----------------|:------------------------|------------------:|-----------:|------------:|------------:|-----------:|--------------------------------:|------------------------------:|
| CQL_HalfCheetah | action_only_sensitivity |                 5 | 0.00225612 | 0.000630704 |  0.00452962 | 0.00237355 |                       -0.88545  |                     0.0113812 |
| CQL_HalfCheetah | primary_3feature        |                 5 | 0.0034538  | 0.00308242  |  0.00551313 | 0.00321889 |                       -0.809947 |                     0.175857  |
| CQL_Hopper      | action_only_sensitivity |                 5 | 0.104312   | 0.0869925   |  0.236634   | 0.206794   |                       -0.857103 |                     0.0451026 |
| CQL_Hopper      | primary_3feature        |                 5 | 0.119608   | 0.0820959   |  0.249203   | 0.199791   |                       -0.721347 |                     0.147307  |
| CQL_Walker2d    | action_only_sensitivity |                 5 | 0.105713   | 0.0462102   |  0.232789   | 0.130682   |                       -0.811691 |                     0.0217305 |
| CQL_Walker2d    | primary_3feature        |                 5 | 0.11095    | 0.0436619   |  0.237423   | 0.125823   |                       -0.671256 |                     0.0926001 |
| IQL_HalfCheetah | action_only_sensitivity |                 5 | 0.0132461  | 0.00349358  |  0.0260527  | 0.0100853  |                       -0.929628 |                     0.0227145 |
| IQL_HalfCheetah | primary_3feature        |                 5 | 0.0156026  | 0.00510957  |  0.0272806  | 0.0105949  |                       -0.92096  |                     0.0235314 |
| IQL_Hopper      | action_only_sensitivity |                 5 | 0.0291898  | 0.0106655   |  0.0707331  | 0.0409977  |                       -0.890558 |                     0.0182727 |
| IQL_Hopper      | primary_3feature        |                 5 | 0.0307619  | 0.0107203   |  0.0712024  | 0.0408557  |                       -0.863824 |                     0.038812  |
| IQL_Walker2d    | action_only_sensitivity |                 5 | 0.0322479  | 0.0145335   |  0.0863347  | 0.0436401  |                       -0.79718  |                     0.0655641 |
| IQL_Walker2d    | primary_3feature        |                 5 | 0.0347799  | 0.0128013   |  0.0863941  | 0.0416709  |                       -0.712282 |                     0.0807214 |
