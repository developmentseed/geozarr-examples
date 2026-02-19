# API Reference

This page documents the public API of the `geozarr-examples` library.

## Convention Models

Pydantic models for GeoZarr conventions.

### Spatial Convention

::: geozarr_examples.Spatial
    options:
      show_source: false

::: geozarr_examples.SpatialConventionMetadata
    options:
      show_source: false

### Proj Convention

::: geozarr_examples.Proj
    options:
      show_source: false

::: geozarr_examples.ProjConventionMetadata
    options:
      show_source: false

### Multiscales Convention

::: geozarr_examples.Multiscales
    options:
      show_source: false

::: geozarr_examples.MultiscalesConventionMetadata
    options:
      show_source: false

::: geozarr_examples.ScaleLevel
    options:
      show_source: false

::: geozarr_examples.Transform
    options:
      show_source: false

### Base Convention

::: geozarr_examples.ZarrConventionMetadata
    options:
      show_source: false

## Metadata Helpers

Functions for creating convention-compliant metadata.

::: geozarr_examples.create_zarr_conventions
    options:
      show_source: false

::: geozarr_examples.create_spatial_attrs
    options:
      show_source: false

::: geozarr_examples.create_proj_attrs
    options:
      show_source: false

::: geozarr_examples.create_multiscales_layout
    options:
      show_source: false

::: geozarr_examples.create_geozarr_attrs
    options:
      show_source: false

::: geozarr_examples.from_geotransform
    options:
      show_source: false

::: geozarr_examples.from_rioxarray
    options:
      show_source: false

## Validation Helpers

Functions for validating convention compliance.

::: geozarr_examples.validate_spatial
    options:
      show_source: false

::: geozarr_examples.validate_proj
    options:
      show_source: false

::: geozarr_examples.validate_multiscales
    options:
      show_source: false

::: geozarr_examples.validate_group
    options:
      show_source: false

::: geozarr_examples.validate_attrs
    options:
      show_source: false

::: geozarr_examples.detect_conventions
    options:
      show_source: false
