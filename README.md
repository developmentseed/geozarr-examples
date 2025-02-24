# GeoZarr examples

## About

This repository contains in-progress work towards GeoZarr examples. If useful, the contents will eventually be migrated to a different repository, such as the
[Cloud Optimized Geospatial Formats Guide](https://github.com/cloudnativegeo/cloud-optimized-geospatial-formats-guide) or the [GeoZarr spec](https://github.com/zarr-developers/geozarr-spec)
repository.

## Goals

- Demonstrate how to write GeoZarr compliant data.
    - Provide a demonstration of writing only the MUST include metadata (most importantly grid_mapping).
    - Provide a demonstration of writing data with a WebMercatorQuad TMS.
    - Provide a demonstration of writing data with a Custom TMS that maps to simple downsampled version of the raw data, without any change in extent or CRS.
    - Provide a demonstration of storing raw data in NetCDF and overviews in native Zarr, with a virtual GeoZarr compliant entrypoint.
- Demonstrate how to read GeoZarr compliant data.
    - Provide a demonstration of reading in GeoZarr data with only MUST include metadata.
    - Provide a demonstration of reading in GeoZarr data with raw data and overviews in "native" zarr.
    - Provide a demonstration of reading in GeoZarr data with raw data in archival formats and overviews in "native" zarr via a single virtual GeoZarr compliant entrypoint.
- Demonstrate how to work with GeoZarr data in Xarray using the prototypes for flexible coordinates and the xproj extension.
    - Demonstrate whether the GeoZarr v0.4 and flexible coordinates solve the limitations highlighted in https://discourse.pangeo.io/t/example-which-highlights-the-limitations-of-netcdf-style-coordinates-for-large-geospatial-rasters/4140.
- Demonstrate how GeoZarr would need to be adapted for Zarr specification v3.

## Feedback cadence

We will provide progress and solicit community feedback during the following events:

- March 05, 2025 GeoZarr Monthly Community Meeting
- March 05, 2025 Pangeo Community Meeting
- April 02, 2025 GeoZarr Monthly Community Meeting
- April 02, 2025 Pangeo Community Meeting
- April 02, 2025 posts on Pangeo and CNG forums
- EGU 2025 Conference
- CNG 2025 Conference

## FAQ

### What's the status of GeoZarr?

GeoZarr is currently being developed as a [Open Geospatial Consortium Standard](https://www.ogc.org/announcement/ogc-forms-new-geozarr-standards-working-group-to-establish-a-zarr-encoding-for-geospatial-data/).
There is a [GeoZarr Standards Working Group](https://portal.ogc.org/index.php?m=public&orderby=default&tab=7) that meets once a month. We now have
experimental prototypes of [all](https://github.com/pydata/xarray/pull/9543) [of](https://github.com/zarr-developers/VirtualiZarr/pull/271) [the](https://xproj.readthedocs.io/en/latest/usage.html)
[pieces](https://zarr.dev/blog/zarr-python-3-release/) to move GeoZarr out of a discussion phase and into a demonstration phase. This repository stems from the hope that building demonstrations will lead to
adoption, iteration, and formalization of a stable GeoZarr specification.

### How does Zarr Specification V3 spec influence GeoZarr?

The GeoZarr spec is designed for Zarr specification version 2. This repository will demonstrate how the differences between Zarr format 2 and Zarr format 3 would
influence GeoZarr. My expectation is that there is not much difference in the metadata requirements between the two formats. However, best practices will likely be impacted by new features available
in Zarr specification version 3 (e.g., sharding).

### How does the release of Zarr-Python 3 influence GeoZarr?

The Zarr-Python 3 release will help GeoZarr users through its increased performance, modernized codebase, and support for extensions. But, it only really interacts with the GeoZarr spec in-so-far
as Zarr-Python 3 supports Zarr specification V3 (see prior question on "How does Zarr Specification V3 spec influence GeoZarr").

### What is the relationship between GeoZarr and Web-Optimized-Zarr?

Web-Optimized Zarr right now is just an idea. I hope that the term "web-optimized zarr" will promote the use of overviews in GeoZarr (which are optional in the core spec) and familiarize practitioners with the concept of storing full-resolution "archival" versions in other file formats and reduced resolution versions in "native" zarr. GeoZarr would allow a single "web-optimized" entrypoint to both via virtualization. A prototype will part of this repository.

## References

- [GeoZarr validator](https://github.com/briannapagan/geozarr-validator) by [@briannapagan](https://github.com/briannapagan) [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- [GeoZarr spec](https://github.com/zarr-developers/geozarr-spec) [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)
- Quarto configuration based on [Cloud Native Geospatial Formats Guide](https://github.com/cloudnativegeo/cloud-optimized-geospatial-formats-guide) and [Tile Benchmarking](https://developmentseed.org/tile-benchmarking/).

## License

Content in this repository is licensed under the [MIT License](LICENSE.txt).
