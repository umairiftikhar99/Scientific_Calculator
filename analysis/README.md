# Synthetic dataset tooling

This folder now contains two complementary scripts:

* `generate_dataset.py` builds an item-level dataset that matches the published
  scale means, standard deviations, and correlation matrix exactly while also
  reproducing the requested Cronbach's α values.  Running the script writes a
  `generated_dataset.csv` file in the same directory and prints a diagnostic
  summary of the achieved metrics.
* `constraints_check.py` documents the inconsistency between the requested
  correlation table and the regression coefficients from the thesis appendix.
  Because the correlation matrix fully determines the ordinary least squares
  solution, any dataset that adheres to the table will necessarily yield the
  coefficients shown by this checker rather than the values printed in the
  appendix.

To regenerate the dataset:

```bash
python analysis/generate_dataset.py
```

The code seeds NumPy's random number generator to keep the output stable across
runs.  If you would like to explore alternative realisations, pass a custom
`--seed` value on the command line.
