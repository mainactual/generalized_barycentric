# generalized_barycentric
Generalized barycentric coordinates

A one-to-one OpenCV C++ implementation of generalized barycentric coordinates [1] with SSE4 optimized versions for typical use cases. Depending on the size of the polygons, the speed up is up to 3x. Typical use cases are (i) color interpolation (Figure 1) and (ii) landmark-based warping (Figure 2).

![polygonfill](https://user-images.githubusercontent.com/17045868/224127320-2c0a67ad-68fa-405f-bdec-21b8c3eade09.jpg)
Figure 1. Left: simple barycentric color interpolation inside a triangle, middle: generalized barycentric color interpolation across the whole image plane; right: difference of the two.

![imagewarp](https://user-images.githubusercontent.com/17045868/224129231-2645fed0-fe90-4d37-9995-c271ebc03192.jpg)
Figure 2. Landmark warping. Left-most: target image; second: source image; third: generalized barycentric coordinates warping; fourth: eye-eye-mouth three point affine transform for comparison. The landmarks (68-point face landmarks) are shown superimposed.

[1] Hormann, "Barycentric coordinates for arbitrary polygons in the plane", 2014
