"""
COG-style multiscale utilities for GeoZarr.

This module provides functions to create Cloud Optimized GeoTIFF (COG) style
overview levels that maintain the native projection and use /2 downsampling logic.
"""

import numpy as np
import xarray as xr
import dask.array as da
import rasterio
import zarr


def calculate_overview_levels(native_width, native_height, min_dimension=256, tileWidth=256):
    """
    Calculate overview levels following COG /2 downsampling logic.
    
    Parameters
    ----------
    native_width : int
        Width of the native resolution data
    native_height : int
        Height of the native resolution data
    min_dimension : int, default 256
        Stop creating overviews when dimension is smaller than this
    tileWidth : int, default 256
        Tile width for TMS compatibility calculations
        
    Returns
    -------
    list
        List of overview level dictionaries with level, zoom, width, height, scale_factor
    """
    overview_levels = []
    level = 0
    current_width = native_width
    current_height = native_height

    while min(current_width, current_height) >= min_dimension:
        # For WebMercatorQuad TMS compatibility, calculate zoom level that can accommodate this resolution
        # This is for serving purposes - the data stays in native CRS
        zoom_for_width = max(0, int(np.ceil(np.log2(current_width / tileWidth))))
        zoom_for_height = max(0, int(np.ceil(np.log2(current_height / tileWidth))))
        zoom = max(zoom_for_width, zoom_for_height)
        
        overview_levels.append({
            'level': level,
            'zoom': zoom,  # For TMS serving compatibility
            'width': current_width,
            'height': current_height,
            'scale_factor': 2**level
        })
        
        level += 1
        # COG-style /2 downsampling
        current_width = native_width // (2**level)
        current_height = native_height // (2**level)

    return overview_levels


def create_overview_template(var, standard_name, *, level, width, height, native_crs, native_bounds, native_transform):
    """
    Create an overview template maintaining native CRS (COG-style).
    
    Parameters
    ----------
    var : str
        Variable name
    standard_name : str
        CF standard name for the variable
    level : int
        Overview level number
    width : int
        Width of this overview level
    height : int
        Height of this overview level
    native_crs : rasterio.crs.CRS
        Native CRS of the data
    native_bounds : tuple
        Native bounds (left, bottom, right, top)
    native_transform : rasterio.transform.Affine
        Native transform
        
    Returns
    -------
    xarray.Dataset
        Template dataset for this overview level
    """
    print(f"Creating template for level {level}: {width}x{height} pixels in native CRS {native_crs}")
    
    # Calculate the transform for this overview level
    overview_transform = rasterio.transform.from_bounds(*native_bounds, width, height)
    
    # Create coordinate arrays in native CRS
    left, bottom, right, top = native_bounds
    
    # Create x and y coordinate arrays for this resolution
    x_coords = np.linspace(left, right, width, endpoint=False)
    y_coords = np.linspace(top, bottom, height, endpoint=False)  # Note: top to bottom for y
    
    # Create the data array with coordinates in native CRS
    overview_da = xr.DataArray(
        da.empty(
            shape=(height, width),
            dtype=np.float32,
            chunks=(min(256, height), min(256, width)),  # Adjust chunk size for smaller overviews
        ),
        dims=["y", "x"],
        coords={
            "x": (["x"], x_coords, {"units": "m", "long_name": "x coordinate of projection"}),
            "y": (["y"], y_coords, {"units": "m", "long_name": "y coordinate of projection"}),
        },
    )
    
    template = overview_da.to_dataset(name=var)
    # Keep the native CRS (not Web Mercator!)
    template = template.rio.write_crs(native_crs)
    
    # Convert transform to GDAL's format
    transform_gdal = overview_transform.to_gdal()
    # Convert transform to space separated string
    transform_str = " ".join([str(i) for i in transform_gdal])
    # Save as an attribute in the `spatial_ref` variable
    template["spatial_ref"].attrs["GeoTransform"] = transform_str
    
    # Ensure CRS information is properly stored in spatial_ref for rioxarray compatibility
    template["spatial_ref"].attrs["crs_wkt"] = native_crs.to_wkt()
    template["spatial_ref"].attrs["spatial_ref"] = native_crs.to_wkt()
    # Add EPSG code if available
    if native_crs.to_epsg():
        template["spatial_ref"].attrs["epsg"] = native_crs.to_epsg()
    # Add proj4 string for additional compatibility
    template["spatial_ref"].attrs["proj4"] = native_crs.to_proj4()
    
    # Set grid_mapping and standard_name
    template[var].attrs["grid_mapping"] = "spatial_ref"
    template[var].attrs["standard_name"] = standard_name
    
    return template


def populate_overview_data(source_data, za, target_width, target_height):
    """
    Populate overview array with downsampled data using numpy-based methods.
    
    Parameters
    ----------
    source_data : numpy.ndarray
        Source data to downsample
    za : zarr.Array
        Target zarr array to populate
    target_width : int
        Target width for downsampling
    target_height : int
        Target height for downsampling
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get source dimensions
        if source_data.ndim == 3:
            source_height, source_width = source_data.shape[-2:]
            # For 3D data, downsample each slice
            downsampled_slices = []
            for i in range(source_data.shape[0]):
                slice_2d = source_data[i]
                # Simple block averaging for downsampling
                block_size_y = source_height // target_height
                block_size_x = source_width // target_width
                
                if block_size_y > 1 and block_size_x > 1:
                    # Reshape and average blocks
                    reshaped = slice_2d[:target_height*block_size_y, :target_width*block_size_x]
                    reshaped = reshaped.reshape(target_height, block_size_y, target_width, block_size_x)
                    downsampled_slice = reshaped.mean(axis=(1, 3))
                else:
                    # Simple subsampling if block averaging doesn't work
                    y_indices = np.linspace(0, source_height-1, target_height, dtype=int)
                    x_indices = np.linspace(0, source_width-1, target_width, dtype=int)
                    downsampled_slice = slice_2d[np.ix_(y_indices, x_indices)]
                
                downsampled_slices.append(downsampled_slice)
            
            downsampled = np.stack(downsampled_slices, axis=0)
        else:
            # 2D data
            source_height, source_width = source_data.shape
            block_size_y = source_height // target_height
            block_size_x = source_width // target_width
            
            if block_size_y > 1 and block_size_x > 1:
                # Reshape and average blocks
                reshaped = source_data[:target_height*block_size_y, :target_width*block_size_x]
                reshaped = reshaped.reshape(target_height, block_size_y, target_width, block_size_x)
                downsampled = reshaped.mean(axis=(1, 3))
            else:
                # Simple subsampling
                y_indices = np.linspace(0, source_height-1, target_height, dtype=int)
                x_indices = np.linspace(0, source_width-1, target_width, dtype=int)
                downsampled = source_data[np.ix_(y_indices, x_indices)]
        
        # Write to zarr array
        if len(za.shape) == 2:
            za[:, :] = downsampled.squeeze()
        else:
            za[:, :, :] = downsampled
        
        return True
    except Exception as e:
        print(f"Failed to populate overview data: {e}")
        return False


def create_cog_style_overviews(ds, var, v3_output, min_dimension=256, tileWidth=256):
    """
    Create COG-style overview levels for a variable in a dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset
    var : str
        Variable name to create overviews for
    v3_output : str
        Output zarr store path
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tileWidth : int, default 256
        Tile width for TMS compatibility
        
    Returns
    -------
    list
        List of created overview levels
    """
    # Get native resolution dimensions and spatial information
    native_height, native_width = ds[var].shape[-2:]
    native_crs = ds.rio.crs
    native_bounds = ds.rio.bounds()
    native_transform = ds.rio.transform()

    print(f"Native resolution: {native_width} x {native_height}")
    print(f"Native CRS: {native_crs}")
    print(f"Native bounds: {native_bounds}")

    # Calculate overview levels
    overview_levels = calculate_overview_levels(
        native_width, native_height, min_dimension, tileWidth
    )
    
    print(f"\nTotal overview levels: {len(overview_levels)}")
    for ol in overview_levels:
        print(f"Overview level {ol['level']}: {ol['width']} x {ol['height']} (TMS zoom: {ol['zoom']}, scale factor: {ol['scale_factor']})")

    # Create and write overview templates
    for overview in overview_levels:
        level = overview['level']
        width = overview['width']
        height = overview['height']
        
        template = create_overview_template(
            var,
            ds[var].attrs["standard_name"],
            level=level,
            width=width,
            height=height,
            native_crs=native_crs,
            native_bounds=native_bounds,
            native_transform=native_transform
        )
        
        # Remove grid_mapping from variable attributes to avoid conflicts
        # We'll add it back through encoding
        if "grid_mapping" in template[var].attrs:
            del template[var].attrs["grid_mapping"]
        
        # Use encoding to specify grid_mapping and avoid compression conflicts
        encoding = {
            var: {"grid_mapping": "spatial_ref"},
            "x": {"compressors": None},
            "y": {"compressors": None}
        }
        
        template.to_zarr(
            v3_output,
            group=str(level),
            compute=False,
            consolidated=False,
            mode="w",
            zarr_format=3,
            encoding=encoding,
        )
        print(f"Created template for overview level {level}")

    print(f"\nCreated {len(overview_levels)} overview levels in native CRS")
    
    # Populate overview arrays with downsampled data
    native_data = ds[var].values
    print(f"Native data shape: {native_data.shape}")

    for overview in overview_levels:
        level = overview['level']
        width = overview['width']
        height = overview['height']
        scale_factor = overview['scale_factor']
        
        print(f"\nProcessing overview level {level} (1:{scale_factor} scale)...")
        print(f"Target dimensions: {width} x {height}")
        
        # Open the zarr array for this level
        za = zarr.open_array(v3_output, path=f"{level}/{var}", zarr_version=3, mode='r+')
        
        # Populate with downsampled data
        if populate_overview_data(native_data, za, width, height):
            print(f"Level {level}: Successfully populated with downsampled data")
        else:
            print(f"Level {level}: Failed to populate")

    print("\nAll overview levels populated with COG-style downsampled data!")
    
    return overview_levels


def verify_overview_coordinates(v3_output, overview_levels, native_crs, max_levels=3):
    """
    Verify that coordinates and CRS are properly maintained in overview levels.
    
    Parameters
    ----------
    v3_output : str
        Path to zarr store
    overview_levels : list
        List of overview level dictionaries
    native_crs : rasterio.crs.CRS
        Expected native CRS
    max_levels : int, default 3
        Maximum number of levels to check
    """
    import xarray as xr
    
    print("Checking coordinates and CRS in overview levels:")
    for overview in overview_levels[:max_levels]:
        level = overview['level']
        
        # Open the overview level as an xarray dataset
        overview_ds = xr.open_zarr(v3_output, group=str(level), zarr_format=3)
        
        # Try to manually set CRS if it's not recognized
        if overview_ds.rio.crs is None and 'spatial_ref' in overview_ds:
            try:
                # Try to set CRS from spatial_ref attributes
                if 'epsg' in overview_ds.spatial_ref.attrs:
                    epsg_code = overview_ds.spatial_ref.attrs['epsg']
                    overview_ds = overview_ds.rio.write_crs(f"EPSG:{epsg_code}")
                elif 'crs_wkt' in overview_ds.spatial_ref.attrs:
                    overview_ds = overview_ds.rio.write_crs(overview_ds.spatial_ref.attrs['crs_wkt'])
                elif 'proj4' in overview_ds.spatial_ref.attrs:
                    overview_ds = overview_ds.rio.write_crs(overview_ds.spatial_ref.attrs['proj4'])
            except Exception as e:
                print(f"  Warning: Could not set CRS from spatial_ref: {e}")
        
        print(f"\nLevel {level}:")
        print(f"  Variables: {list(overview_ds.data_vars)}")
        print(f"  Coordinates: {list(overview_ds.coords)}")
        print(f"  Dimensions: {dict(overview_ds.sizes)}")
        print(f"  CRS: {overview_ds.rio.crs}")
        
        # Check spatial_ref attributes
        if 'spatial_ref' in overview_ds:
            print(f"  Spatial_ref attrs: {list(overview_ds.spatial_ref.attrs.keys())}")
            if 'epsg' in overview_ds.spatial_ref.attrs:
                print(f"  EPSG in spatial_ref: {overview_ds.spatial_ref.attrs['epsg']}")
        
        if 'x' in overview_ds.coords and 'y' in overview_ds.coords:
            print(f"  X range: {overview_ds.x.min().values:.2f} to {overview_ds.x.max().values:.2f}")
            print(f"  Y range: {overview_ds.y.min().values:.2f} to {overview_ds.y.max().values:.2f}")
            print(f"  ✓ Coordinates present in native CRS")
        else:
            print(f"  ✗ Missing coordinates!")
        
        # Check if CRS matches native CRS
        if overview_ds.rio.crs == native_crs:
            print(f"  ✓ Native CRS maintained: {native_crs}")
        else:
            print(f"  ✗ CRS changed! Expected: {native_crs}, Got: {overview_ds.rio.crs}")


def plot_overview_levels(v3_output, overview_levels, var, max_plots=3):
    """
    Plot overview levels using xarray's native plot() method.
    
    Parameters
    ----------
    v3_output : str
        Path to zarr store
    overview_levels : list
        List of overview level dictionaries
    var : str
        Variable name to plot
    max_plots : int, default 3
        Maximum number of plots to create
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    
    num_plots = min(max_plots, len(overview_levels))
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    for i, overview in enumerate(overview_levels[:num_plots]):
        level = overview['level']
        width = overview['width']
        height = overview['height']
        scale_factor = overview['scale_factor']
        
        # Open the overview level as an xarray dataset
        overview_ds = xr.open_zarr(v3_output, group=str(level), zarr_format=3)
        
        # Use xarray's native plot method - this automatically uses coordinates!
        overview_ds[var].plot(
            ax=axes[i],
            cmap='viridis',
            add_colorbar=True,
            robust=True  # Use robust color scaling
        )
        
        axes[i].set_title(f"Level {level} (1:{scale_factor} scale)\n{width}x{height} pixels\nNative CRS: {overview_ds.rio.crs}")
        axes[i].set_xlabel(f"X coordinate ({overview_ds.rio.crs})")
        axes[i].set_ylabel(f"Y coordinate ({overview_ds.rio.crs})")
        
        # Format coordinate labels to be more readable
        axes[i].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

    plt.tight_layout()
    plt.suptitle(f"COG-style Overview Levels in Native CRS", y=1.02, fontsize=14)
    plt.show()
    
    return fig
