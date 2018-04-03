# MilleFeuille

Pythonic data structure to hold inhomogeneous data of various sources. The data lives in `layers`, where in a single `layer` the data is homogeneous and of the same type. Several, different, `layers` live in a `stack`. Translation methods allow to translate data between `layers` of a `stack`. Layers define their available translations methods. A `Binlayer` for instance will offer simple histogramming function, or more sophisticated translation methods to translate from `PointLayers`.

Convenience `pyplot` plotting functions are also attached to the layers, in order to produce standard plots without the hassle of figuring out the correct input to the standard matplotlib functions.

## Installation

`pip install --editable .`
