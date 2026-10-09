# Model migration

Version 0.3 reads format-2 artifacts. Convert a trusted artifact saved by version
0.2 from a source checkout with the current package installed:

```bash
python tools/convert_model.py old-model converted-model
```

The converter transfers the fitted coefficients, reference features, scalers,
validation correlations and training metadata. It validates the result before
writing a new destination directory. Keep the source model until the converted
model has been checked on your inputs. The converter is a standalone utility;
`DLUnmix.load` reads the current format.

Converted weights preserve the learned function within floating-point rounding.
Matrix dimensions can change kernel accumulation order. The numerical contract
uses rtol=1e-5 and atol=2e-5 for processed expression. Conversion does not retrain
the model. A fresh training run uses the current architecture's initialization,
so matching a previous seed is not a substitute for model conversion.
