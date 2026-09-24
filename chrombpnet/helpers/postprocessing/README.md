## Export a model for TF-Keras 2.x readers

chrombpnet 2.x writes its model files with Keras 3. TF-Keras 2.x (chrombpnet 1.x, the kundajelab variant-scorer) cannot
load them, and h5py readers such as bpnet-lite (`BPNet.from_chrombpnet`, `ChromBPNet.from_chrombpnet`) do not find the
weights, because Keras 3 names the datasets `layer/kernel` instead of `layer/kernel:0`. `chrombpnet export` writes a
copy in the layout chrombpnet 1.x wrote (TF-Keras 2.12):

```
chrombpnet export -m path/to/chrombpnet_nobias.h5 -o path/to/chrombpnet_nobias.legacy.h5 [--legacy-h5]
```

`-m` takes a bias, chrombpnet or chrombpnet_nobias model (.h5 or .keras, written by chrombpnet 2.x or 1.x).
`--legacy-h5` is the default and, for now, the only format. In Python:
`chrombpnet.helpers.postprocessing.export_legacy_h5.export_legacy_h5(model_or_path, out_path)`.

The file has the TF-Keras 2.x full-model layout:

* root attributes `keras_version` (`2.12.0`), `backend` (`tensorflow`) and `model_config`, a Keras 2 functional
  config (list-form inbound nodes, a nested model called once referenced as node 1, InputLayer
  `batch_input_shape`, plain dtype strings);
* `model_weights/<layer>/<layer>/kernel:0` (and `bias:0`) datasets, with the `layer_names` / `weight_names`
  attributes TF-Keras loads by; in a full chrombpnet model the nested models hold all their weights:
  `model_weights/model/bpnet_1conv/kernel:0` (frozen bias model) and
  `model_weights/model_wo_bias/wo_bias_bpnet_1conv/kernel:0`;
* no `training_config` and no `optimizer_weights`: load it with `compile=False`.

Keras 3 auto-generated names are written as TF-Keras named them in 1.x: a functional model named `functional` is
written as `model`, and weightless layers such as `add_4` ... `add_7` are renumbered `add` ... `add_3` in each model.
Layers with weights keep their names. bpnet-lite needs the renumbering: it takes the number of dilated layers from
the largest number at the end of any layer name, so `add_7` in a 4-layer model would make it look for `bpnet_7conv`. Layers are written in float32 (a model trained with `--precision bf16` has
float32 weights already).

Reading the file:

* **TF-Keras 2.x** (tested with TF 2.8 and 2.12): bias and no-bias models load with
  `tf.keras.models.load_model(path, compile=False)`. The count head of a full chrombpnet model is a `Lambda` that
  names its function (`chrombpnet_logsumexp`) instead of storing Python bytecode, which loads only on the Python
  version that wrote it. Pass that function:

  ```python
  import tensorflow as tf

  def chrombpnet_logsumexp(x):
      return tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)

  model = tf.keras.models.load_model("chrombpnet.legacy.h5", compile=False,
                                     custom_objects={"chrombpnet_logsumexp": chrombpnet_logsumexp})
  ```

  (or `tf.keras.utils.get_custom_objects()["chrombpnet_logsumexp"] = chrombpnet_logsumexp` before a
  `load_model` call you cannot change, e.g. in the variant-scorer). Without it, TF-Keras fails with
  `AttributeError: 'NoneType' object has no attribute 'get'`.
* **bpnet-lite** reads the weight datasets only, of bias and no-bias files:
  `ChromBPNet.from_chrombpnet("bias.legacy.h5", "chrombpnet_nobias.legacy.h5")`.
* **chrombpnet 2.x** (`chrombpnet.training.utils.model_io.load_model`) reads all of them.

## Rebuild chrombpnet.h5 from a no-bias and a bias model

```
python -m chrombpnet.helpers.postprocessing.reformat_chrombpnet_h5 -cnb path/to/chrombpnet_nobias.h5 -bm /path/to/bias_model_scaled.h5 -o /path/to/outputdir
```

make sure outputdir exists

Writes outputdir/chrombpnet_recompiled.h5 (Keras 3; the count head is the registered LogSumExp layer instead of a
Lambda). chrombpnet 1.x also wrote a TensorFlow SavedModel directory (outputdir/chrombpnet_recompiled); that export
was dropped together with the TensorFlow dependency.
