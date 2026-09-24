
python -m chrombpnet.helpers.postprocessing.reformat_chrombpnet_h5 -cnb path/to/chrombpnet_nobias.h5 -bm /path/to/bias_model_scaled.h5 -o /path/to/outputdir

make sure outputdir exists

Writes outputdir/chrombpnet_recompiled.h5 (Keras 3; the count head is the registered LogSumExp layer instead of a
Lambda). chrombpnet 1.x also wrote a TensorFlow SavedModel directory (outputdir/chrombpnet_recompiled); that export
was dropped together with the TensorFlow dependency.
