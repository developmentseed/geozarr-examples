import zarr
from async_geotiff import GeoTIFF
from geozarr_toolkit import (
    MultiscalesConventionMetadata,
    ProjConventionMetadata,
    SpatialConventionMetadata,
    create_multiscales_layout,
    create_proj_attrs,
    create_spatial_attrs,
    create_zarr_conventions,
)
from obstore.store import S3Store
from zarr.storage import LocalStore

DIMENSIONS = ("band", "Y", "X")


store = S3Store("sentinel-cogs", region="us-west-2", skip_signature=True)
# path = "sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/TCI.tif"
path = "sentinel-s2-l2a-cogs/18/T/WL/2026/1/S2B_18TWL_20260101_0_L2A/TCI.tif"

geotiff = await GeoTIFF.open(path, store=store)

proj_attrs = create_proj_attrs(code=f"EPSG:{geotiff.crs.to_epsg()}")
spatial_attrs = create_spatial_attrs(
    dimensions=["Y", "X"],
    bbox=geotiff.bounds,
)

# Build multiscales layout from the COG's overviews
# The base (full-resolution) image is level 0; each overview is a coarser level.
levels = [
    {"asset": "0", "transform": {"scale": [1.0, 1.0], "translation": [0.0, 0.0]}},
]
for i, overview in enumerate(geotiff.overviews):
    ov_res = overview.transform.a
    scale_factor_x = overview.transform.a / geotiff.transform.a
    scale_factor_y = overview.transform.e / geotiff.transform.e
    levels.append(
        {
            "asset": str(i + 1),
            "derived_from": "0",
            "transform": {
                "scale": [scale_factor_x, scale_factor_y],
                "translation": [0.0, 0.0],
            },
        }
    )

multiscales_attrs = create_multiscales_layout(levels)
multiscales_attrs["multiscales"]["layout"][0]["spatial:transform"] = geotiff.transform[:6]
multiscales_attrs["multiscales"]["layout"][0]["spatial:shape"] = geotiff.shape
for item, overview in zip(
    multiscales_attrs["multiscales"]["layout"][1:], geotiff.overviews, strict=True
):
    item["spatial:transform"] = overview.transform[:6]
    item["spatial:shape"] = overview.height, overview.width


zarr_conventions = create_zarr_conventions(
    MultiscalesConventionMetadata(),
    ProjConventionMetadata(),
    SpatialConventionMetadata(),
)

geozarr_attrs = {
    **proj_attrs,
    **spatial_attrs,
    **multiscales_attrs,
    "zarr_conventions": zarr_conventions,
}

local_path = "data/TCI.zarr"
zarr_store = LocalStore(local_path)

root: zarr.Group = zarr.open_group(zarr_store, mode="w", zarr_format=3)

# Set convention attributes on the group
root.attrs.update(geozarr_attrs)

# Write the full-resolution image as level "0"
base_array = await geotiff.read()
root.create_array(
    "0",
    data=base_array.data,
    chunks=(3, 512, 512),
    dimension_names=DIMENSIONS,
)
print(f"Level 0 (base): shape={base_array.data.shape}, dtype={base_array.data.dtype}")

# Write each overview as a separate level
for i, overview in enumerate(geotiff.overviews):
    ov_array = await overview.read()
    root.create_array(
        str(i + 1),
        data=ov_array.data,
        chunks=(3, 512, 512),
        dimension_names=DIMENSIONS,
    )
    print(f"Level {i + 1} (overview): shape={ov_array.data.shape}")
