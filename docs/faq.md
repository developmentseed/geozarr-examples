# FAQ

### What's the status of GeoZarr?

GeoZarr is currently being developed as an [Open Geospatial Consortium Standard](https://www.ogc.org/announcement/ogc-forms-new-geozarr-standards-working-group-to-establish-a-zarr-encoding-for-geospatial-data/).
There is a [GeoZarr Standards Working Group](https://portal.ogc.org/index.php?m=public&orderby=default&tab=7) that meets once a month. The GeoZarr SWG has decided that Zarr Conventions, including the [spatial convention](https://github.com/zarr-conventions/spatial), [geo-proj convention](https://github.com/zarr-conventions/geo-proj), and [multiscales convention](https://github.com/zarr-conventions/multiscales) will be the foundation for GeoZarr.

### What's the new approach for GeoZarr conventions?

GeoZarr has shifted from a complex, abstract model (CDM) to a collection of simple, composable Zarr Conventions that directly address developer needs. This incremental, feature-driven strategy delivers immediate value and builds momentum. The [Zarr Conventions Framework](https://github.com/zarr-conventions/zarr-conventions-spec) provides the mechanism for communities (e.g., geospatial, bioimaging) to register and share metadata conventions within Zarr using a `zarr_conventions` attribute that points to a convention's spec URL, schema, or UUID.

### What tools support GeoZarr conventions?

GeoZarr conventions are gaining adoption across the ecosystem:

- **OpenLayers**: Initial PR submitted for Multiscales support
- **TiTiler**: Already compliant with Multiscales
- **GDAL**: Read-only support for Multiscales and proj: conventions funded and in development
- **EOPF Explorer V1 (ESA)**: Uses Zarr conventions as a foundational component
- **ZarrLayer (CarbonPlan)**: Experimenting with Multiscales and proj:
- **QGIS**: Plugin in development for direct GeoZarr data analysis

### What is the relationship between GeoZarr and CF conventions?

Zarr has a 10-year history of successful, informal use with CF (Climate and Forecast) data. The primary challenge is formalizing these existing patterns, while also providing optimal solutions for Earth data outside the scope of CF. The GeoZarr community is working on a proposal to formalize CF-on-Zarr patterns to ensure that GeoZarr serves both the GIS world and critical climate/weather data use cases. The [CF convention](examples/cf/) for Zarr uses unprefixed attribute names (`standard_name`, `units`, etc.) for backwards compatibility with existing CF-compliant datasets and libraries like cf-python and xarray.

### How does Zarr Specification V3 spec influence GeoZarr?

The GeoZarr conventions are designed to work with both Zarr format 2 and Zarr format 3. The convention metadata content (`spatial:`, `proj:`, `multiscales`) is identical regardless of Zarr version - only the storage location differs (`.zattrs` files in V2 vs `attributes` in `zarr.json` for V3). Best practices may be impacted by new features available in Zarr specification version 3 (e.g., sharding).

### How does the release of Zarr-Python 3 influence GeoZarr?

The Zarr-Python 3 release will help GeoZarr users through its increased performance, modernized codebase, and support for extensions. But, it only really interacts with the GeoZarr spec in-so-far
as Zarr-Python 3 supports Zarr specification V3 (see prior question on "How does Zarr Specification V3 spec influence GeoZarr").

### Where can I discuss GeoZarr?

OGC has dissolved its Google Groups. The primary communication channels are now:

- **CNG Slack**: [#geozarr channel](https://cloudnativegeo.slack.com/)
- **OGC Agora**: The official OGC discussion platform
