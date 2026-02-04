# GeoZarr examples

## Overview

[GeoZarr examples](https://github.com/developmentseed/geozarr-examples) provides a Python library and examples for working with GeoZarr-compliant Zarr stores. The library implements the new modular Zarr conventions:

- **[spatial:](https://github.com/zarr-conventions/spatial)** - Spatial coordinate and transformation information
- **[proj:](https://github.com/zarr-experimental/geo-proj)** - Coordinate Reference System (CRS) information
- **[multiscales](https://github.com/zarr-conventions/multiscales)** - Multiscale pyramid layout

## Python Library

Install the library to create and validate GeoZarr-compliant metadata:

```bash
pip install "geozarr-examples @ git+https://github.com/developmentseed/geozarr-examples.git"
```

Quick example:

```python exec="on" source="above" result="json"
from geozarr_examples import create_geozarr_attrs
import json

attrs = create_geozarr_attrs(
    dimensions=["Y", "X"],
    crs="EPSG:32633",
    transform=[10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
)

print(json.dumps(attrs, indent=2))
```

See the [Getting Started guide](getting-started.md) for more details, or browse the [API Reference](api/index.md).

## Goals

- Demonstrate how to write GeoZarr compliant data.
    - Provide a demonstration of writing CRS information.
    - Provide a demonstration of writing bounding box information.
    - Provide a demonstration of writing multiscale data.
    - Provide a demonstration of writing multi-scale data conforming to a specific well-known tile matrix set (TMS)
    - Provide a demonstration of storing raw data in NetCDF and overviews in native Zarr, with a virtual GeoZarr compliant entrypoint.
- Demonstrate how to read GeoZarr compliant data.
    - Provide a demonstration of reading in GeoZarr data with raw data in "native" zarr.
    - Provide a demonstration of reading in GeoZarr data with raw data and overviews in "native" zarr.
    - Provide a demonstration of reading in GeoZarr data with raw data in archival formats and overviews in "native" zarr via a single virtual GeoZarr compliant entrypoint.
- Demonstrate how to work with GeoZarr data in Xarray using flexible coordinates and the xproj extension.
- Demonstrate how to work with GeoZarr data in OpenLayers.
- Demonstrate how to work with GeoZarr data in GDAL.

## Feedback cadence

We will provide progress and solicit community feedback during the following events:

- February 04, 2026 - GeoZarr SWG Meeting.

See the [FAQ](faq.md) for common questions about GeoZarr.

## References/Acknowledgements

### Specifications and Standards

- [GeoZarr spec](https://github.com/zarr-developers/geozarr-spec) [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- [CF Conventions](https://cfconventions.org/) - Climate and Forecast metadata conventions
- [STAC Extensions Template](https://github.com/stac-extensions/template) - Convention documentation structure
- [EOPF-Explorer Data Model](https://github.com/EOPF-Explorer/data-model) - Base implementation for geo-proj and spatial conventions
- [Zarr Extensions PR #21](https://github.com/zarr-developers/zarr-extensions/pull/21) - Original draft of conventions

### Code Attribution

The Python convention models (`Spatial`, `Proj`, `Multiscales`, etc.) in this library follow patterns established in:

- [eopf-geozarr](https://github.com/EOPF-Explorer/data-model) - Pydantic models for GeoZarr conventions, part of the EOPF (Earth Observation Processing Framework) ecosystem

### Software Libraries

Examples in this repository use the following open-source libraries:

- [Zarr](https://zarr.readthedocs.io/) - Chunked array storage format (MIT License)
- [xarray](https://xarray.pydata.org/) - N-dimensional labeled arrays (Apache 2.0)
- [rioxarray](https://corteva.github.io/rioxarray/) - Rasterio xarray extension (Apache 2.0)
- [cf-xarray](https://cf-xarray.readthedocs.io/) - CF conventions for xarray (Apache 2.0)
- [Rasterio](https://rasterio.readthedocs.io/) - Geospatial raster I/O (BSD 3-Clause)
- [pyproj](https://pyproj4.github.io/) - Cartographic projections (MIT License)
- [Affine](https://github.com/rasterio/affine) - Affine transformation library (BSD 3-Clause)
- [morecantile](https://developmentseed.org/morecantile/) - Tile Matrix Set utilities (MIT License)
- [rio-tiler](https://cogeotiff.github.io/rio-tiler/) - Rasterio plugin for COG tiles (MIT License)
- [earthaccess](https://earthaccess.readthedocs.io/) - NASA Earthdata access (MIT License)

### Tools and Validators

- [GeoZarr validator](https://github.com/briannapagan/geozarr-validator) by [@briannapagan](https://github.com/briannapagan) [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- [AJV](https://ajv.js.org/) - JSON Schema validator used in convention tests (MIT License)

### Data

Example notebooks use data from:

- [NASA JPL MUR-SST](https://podaac.jpl.nasa.gov/MEaSUREs-MUR) - Multi-scale Ultra-high Resolution Sea Surface Temperature
- [Copernicus Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) via [Earth Search](https://earth-search.aws.element84.com/)

## License

Content in this repository is licensed under the [MIT License](https://github.com/developmentseed/geozarr-examples/blob/main/LICENSE.txt).
