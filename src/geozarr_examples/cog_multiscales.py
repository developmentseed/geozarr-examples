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
import matplotlib.pyplot as plt
import time
from pathlib import Path


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
    tuple
        (bool, float) - (Success status, processing time in seconds)
    """
    import time
    start_time = time.time()
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
        
        processing_time = time.time() - start_time
        return True, processing_time
    except Exception as e:
        print(f"Failed to populate overview data: {e}")
        return False, 0.0


def create_cog_style_overviews(ds, var, v3_output, min_dimension=256, tileWidth=256, group_prefix=None, collect_timing=True):
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
    group_prefix : str, optional
        Group prefix for hierarchical zarr stores
        
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
    group_prefix : str, optional
        Group prefix for hierarchical zarr stores
    collect_timing : bool, default True
        Whether to collect timing data for each overview level
        
    Returns
    -------
    tuple
        (overview_levels, timing_data)
        - overview_levels: List of created overview level dictionaries
        - timing_data: List of timing measurements for each level
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
    
    timing_data = []
    if collect_timing:
        # Time and pixel count for each overview level
        start_time = time.time()
        for overview in overview_levels:
            level = overview['level']
            width = overview['width']
            height = overview['height']
            
            # Open the zarr array for this level
            za = zarr.open_array(v3_output, path=f"{level}/{var}", zarr_version=3, mode='r')
            # Record the level data
            success, proc_time = populate_overview_data(native_data, za, width, height)
            if success:
                timing_data.append({
                    'level': level,
                    'time': proc_time,
                    'pixels': width * height,
                    'width': width,
                    'height': height,
                    'scale_factor': 2**level
                })
    
    return overview_levels, timing_data


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


def setup_eopf_metadata(reflectance_ds):
    """
    Set up CF standard names and CRS information for EOPF reflectance measurements.
    
    Parameters
    ----------
    reflectance_ds : xarray.DataTree
        The reflectance measurements DataTree from EOPF
        
    Returns
    -------
    dict
        Dictionary mapping group names to processed datasets
    """
    processed_groups = {}
    
    # Loop over the reflectance groups
    for group in reflectance_ds.groups:
        if not reflectance_ds[group].data_vars:
            # Skip groups without data variables
            continue
            
        print(f"Processing group: {group}")
        ds = reflectance_ds[group].ds.copy()
        
        # Loop over the bands in the group
        for band in ds.data_vars:
            print(f"  Processing band: {band}")
            
            # Set CF standard name
            ds[band].attrs["standard_name"] = "toa_bidirectional_reflectance"
            
            # Check if the band has the proj:epsg attribute
            if "proj:epsg" in ds[band].attrs:
                epsg = ds[band].attrs["proj:epsg"]
                print(f"    Setting CRS for {band} to EPSG:{epsg}")
                ds = ds.rio.write_crs(f"epsg:{epsg}")
                ds[band].attrs["grid_mapping"] = "spatial_ref"
        
        processed_groups[group] = ds
        
    return processed_groups


def create_full_eopf_zarr_store(dt, output_path, spatial_chunk=4096, min_dimension=256, tileWidth=256, 
                               load_data=True, max_retries=3, skip_existing=True, force_overwrite=False):
    """
    Create a full EOPF Zarr store with all resolutions, groups, and variables.
    
    Parameters
    ----------
    dt : xarray.DataTree
        Input EOPF DataTree
    output_path : str
        Output path for the Zarr store
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tileWidth : int, default 256
        Tile width for TMS compatibility
    load_data : bool, default True
        Whether to load data into memory before processing (helps with timeouts)
    max_retries : int, default 3
        Maximum number of retries for network operations
    skip_existing : bool, default True
        Skip processing if output already exists (speeds up testing)
    force_overwrite : bool, default False
        Force overwrite existing data (overrides skip_existing)
        
    Returns
    -------
    dict
        Dictionary containing processed groups and overview information
    """
    from zarr.codecs import BloscCodec
    import time
    import os
    
    # Set up compression
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle='shuffle', blocksize=0)
    
    # Process reflectance measurements
    reflectance_ds = dt["measurements/reflectance"]
    processed_groups = setup_eopf_metadata(reflectance_ds)
    
    # Create the main zarr store structure
    result = {
        'processed_groups': processed_groups,
        'overview_levels': {},
        'output_path': output_path
    }
    
    # Check if we should skip existing data
    if skip_existing and not force_overwrite and os.path.exists(output_path):
        print(f"⏭️  Output path {output_path} already exists.")
        print("🔍 Checking for existing data...")
        
        # Try to load existing results
        existing_groups = {}
        existing_overviews = {}
        all_exist = True
        
        for group_name in processed_groups.keys():
            group_path = f"{output_path}/{group_name}"
            if os.path.exists(group_path):
                try:
                    # Try to load the existing group
                    existing_ds = xr.open_zarr(group_path, zarr_format=3)
                    existing_groups[group_name] = existing_ds
                    print(f"✅ Found existing data for {group_name}")
                    
                    # Check for existing overviews
                    group_overviews = {}
                    for var in existing_ds.data_vars:
                        if var in ['spatial_ref', 'time']:
                            continue
                        overview_path = f"{group_path}/{var}_overviews"
                        if os.path.exists(overview_path):
                            try:
                                # Try to determine overview levels
                                overview_dirs = [d for d in os.listdir(overview_path) 
                                               if os.path.isdir(os.path.join(overview_path, d)) and d.isdigit()]
                                if overview_dirs:
                                    # Create mock overview levels info
                                    overview_levels = []
                                    for level_str in sorted(overview_dirs, key=int):
                                        level = int(level_str)
                                        try:
                                            level_ds = xr.open_zarr(overview_path, group=level_str, zarr_format=3)
                                            if var in level_ds.data_vars:
                                                height, width = level_ds[var].shape[-2:]
                                                overview_levels.append({
                                                    'level': level,
                                                    'width': width,
                                                    'height': height,
                                                    'scale_factor': 2**level
                                                })
                                        except Exception:
                                            continue
                                    
                                    if overview_levels:
                                        group_overviews[var] = {
                                            'levels': overview_levels,
                                            'path': overview_path
                                        }
                                        print(f"✅ Found existing overviews for {group_name}/{var}")
                                    else:
                                        print(f"⚠️  No valid overviews found for {group_name}/{var}")
                                        all_exist = False
                                else:
                                    print(f"⚠️  No overview directories found for {group_name}/{var}")
                                    all_exist = False
                            except Exception as e:
                                print(f"⚠️  Could not load overviews for {group_name}/{var}: {e}")
                                all_exist = False
                        else:
                            print(f"⚠️  Overview path not found for {group_name}/{var}")
                            all_exist = False
                    
                    existing_overviews[group_name] = group_overviews
                    
                except Exception as e:
                    print(f"⚠️  Could not load existing data for {group_name}: {e}")
                    all_exist = False
            else:
                print(f"⚠️  Group path not found: {group_path}")
                all_exist = False
        
        if all_exist and existing_groups:
            print("🎉 All data already exists! Returning existing results.")
            return {
                'processed_groups': existing_groups,
                'overview_levels': existing_overviews,
                'output_path': output_path
            }
        else:
            print("⚠️  Some data missing or incomplete. Proceeding with processing...")
    
    # Create the root zarr store first
    print(f"Creating root Zarr store at: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Create a simple approach: process one group at a time and create separate stores
    # This is more reliable than trying to create complex hierarchical structures
    for group_name, ds in processed_groups.items():
        print(f"\n=== Processing {group_name} ===")
        
        # Load data into memory if requested to avoid timeout issues
        if load_data:
            print(f"Loading {group_name} data into memory to avoid timeouts...")
            try:
                # Load with retries
                for attempt in range(max_retries):
                    try:
                        ds = ds.load()
                        print(f"Successfully loaded {group_name} data")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed, retrying in 5 seconds...")
                            time.sleep(5)
                        else:
                            print(f"Failed to load {group_name} after {max_retries} attempts: {e}")
                            print("Continuing with lazy loading (may be slower)...")
                            break
            except Exception as e:
                print(f"Warning: Could not load {group_name} data: {e}")
                print("Continuing with lazy loading...")
        
        # Create encoding for all variables in this group
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {
                "chunks": (1, spatial_chunk, spatial_chunk),
                "compressors": compressor,
            }
        
        # Add coordinate encoding
        for coord in ds.coords:
            encoding[coord] = {"compressors": None}
        
        # Write the base resolution data as a subgroup
        group_path = f"{output_path}/{group_name}"
        for attempt in range(max_retries):
            try:
                print(f"Writing base resolution for {group_name} to {group_path}")
                ds.to_zarr(
                    group_path,
                    mode="w",
                    consolidated=True,
                    zarr_format=3, 
                    encoding=encoding
                )
                print(f"Written base resolution for {group_name}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Write attempt {attempt + 1} failed, retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Failed to write {group_name} after {max_retries} attempts: {e}")
                    raise
        
        # Create overviews for each variable (simplified approach)
        group_overviews = {}
        for var in ds.data_vars:
            print(f"\nChecking {group_name}/{var}")
            
            # Skip special variables
            if var in ['spatial_ref', 'time']:
                continue
            
            # Check if this variable already exists
            var_overview_path = f"{group_path}/{var}_overviews"
            if skip_existing and not force_overwrite and os.path.exists(var_overview_path):
                try:
                    # Try to verify existing overviews
                    overview_dirs = [d for d in os.listdir(var_overview_path) 
                                   if os.path.isdir(os.path.join(var_overview_path, d)) and d.isdigit()]
                    if overview_dirs:
                        print(f"✅ Found existing overviews for {group_name}/{var}, skipping...")
                        # Add to results
                        var_overviews = []
                        for level_str in sorted(overview_dirs, key=int):
                            level = int(level_str)
                            try:
                                level_ds = xr.open_zarr(var_overview_path, group=level_str, zarr_format=3)
                                if var in level_ds.data_vars:
                                    height, width = level_ds[var].shape[-2:]
                                    var_overviews.append({
                                        'level': level,
                                        'width': width,
                                        'height': height,
                                        'scale_factor': 2**level
                                    })
                            except Exception as e:
                                print(f"⚠️  Could not verify level {level}: {e}")
                                continue
                        
                        if var_overviews:
                            group_overviews[var] = {
                                'levels': var_overviews,
                                'path': var_overview_path
                            }
                            continue
                        else:
                            print("⚠️  No valid overviews found, will recreate")
                except Exception as e:
                    print(f"⚠️  Could not verify existing overviews: {e}")
            
            print(f"Creating overviews for {group_name}/{var}")
            try:
                # Add multiscales metadata
                native_height, native_width = ds[var].shape[-2:]
                overview_levels = calculate_overview_levels(native_width, native_height, min_dimension, tileWidth)
                
                tile_matrix_limits = {str(ol['level']): {} for ol in overview_levels}
                ds[var].attrs["multiscales"] = {
                    "tile_matrix_set": "WebMercatorQuad",
                    "resampling_method": "nearest",
                    "tile_matrix_limits": tile_matrix_limits,
                }
                
                # Create a single-variable dataset for overview creation
                var_ds = ds[[var]].copy()  # Create a dataset with just this variable
                
                # Create overview levels in a separate store for now
                var_overview_path = f"{group_path}/{var}_overviews"
                var_overviews, timing = create_cog_style_overviews(
                    ds=var_ds,
                    var=var,
                    v3_output=var_overview_path,
                    min_dimension=min_dimension,
                    tileWidth=tileWidth
                )
                
                group_overviews[var] = {
                    'levels': var_overviews,
                    'path': var_overview_path,
                    'timing': timing
                }
                
            except Exception as e:
                print(f"Warning: Failed to create overviews for {group_name}/{var}: {e}")
                print("Continuing with next variable...")
                continue
        
        result['overview_levels'][group_name] = group_overviews
    
    # Create a simple root zarr.json to make it a valid zarr store
    try:
        import json
        root_metadata = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "description": "EOPF Zarr store with multiscale support",
                "groups": list(processed_groups.keys())
            }
        }
        
        with open(f"{output_path}/zarr.json", "w") as f:
            json.dump(root_metadata, f, indent=2)
        
        print(f"Created root zarr.json")
        
    except Exception as e:
        print(f"Warning: Could not create root zarr.json: {e}")
    
    return result


def plot_rgb_overview(zarr_store_path, group_name, red_band, green_band, blue_band, overview_level=0, 
                     stretch_percentiles=(2, 98), figsize=(12, 8)):
    """
    Create an RGB plot using overview data from a Zarr store.
    
    Parameters
    ----------
    zarr_store_path : str
        Path to the Zarr store
    group_name : str
        Name of the resolution group (e.g., 'r10m', 'r20m', 'r60m')
    red_band : str
        Name of the red band variable
    green_band : str
        Name of the green band variable  
    blue_band : str
        Name of the blue band variable
    overview_level : int, default 0
        Overview level to use (0 = native resolution)
    stretch_percentiles : tuple, default (2, 98)
        Percentiles for contrast stretching
    figsize : tuple, default (12, 8)
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    
    # Load the overview data for each band
    bands_data = {}
    coords = None
    crs = None
    
    for band_name, band_var in [('red', red_band), ('green', green_band), ('blue', blue_band)]:
        if overview_level == 0:
            # Use native resolution
            band_path = f"{zarr_store_path}/{group_name}"
            ds = xr.open_zarr(band_path, zarr_format=3)
            band_data = ds[band_var]
        else:
            # Use overview level
            overview_path = f"{zarr_store_path}/{group_name}/{band_var}_overviews"
            ds = xr.open_zarr(overview_path, group=str(overview_level), zarr_format=3)
            band_data = ds[band_var]
        
        # Store the data and get coordinates from first band
        bands_data[band_name] = band_data.values.squeeze()
        if coords is None:
            coords = {'x': band_data.x.values, 'y': band_data.y.values}
            if hasattr(ds, 'rio') and ds.rio.crs:
                crs = ds.rio.crs
    
    # Create RGB array
    rgb_array = np.stack([bands_data['red'], bands_data['green'], bands_data['blue']], axis=-1)
    
    # Apply contrast stretching
    rgb_stretched = np.zeros_like(rgb_array)
    for i in range(3):
        band = rgb_array[:, :, i]
        # Remove NaN values for percentile calculation
        valid_data = band[~np.isnan(band)]
        if len(valid_data) > 0:
            p_low, p_high = np.percentile(valid_data, stretch_percentiles)
            band_stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
            rgb_stretched[:, :, i] = band_stretched
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use imshow with proper extent
    extent = [coords['x'].min(), coords['x'].max(), coords['y'].min(), coords['y'].max()]
    im = ax.imshow(rgb_stretched, extent=extent, origin='upper', aspect='equal')
    
    # Set labels and title
    scale_factor = 2**overview_level if overview_level > 0 else 1
    title = f"RGB Composite ({red_band}, {green_band}, {blue_band})\n"
    title += f"Group: {group_name}, Overview Level: {overview_level} (1:{scale_factor} scale)"
    if crs:
        title += f"\nCRS: {crs}"
    
    ax.set_title(title)
    ax.set_xlabel(f"X coordinate ({crs if crs else 'unknown CRS'})")
    ax.set_ylabel(f"Y coordinate ({crs if crs else 'unknown CRS'})")
    
    # Format coordinate labels
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
    return fig


def get_sentinel2_rgb_bands(group_name):
    """
    Get appropriate RGB band combinations for Sentinel-2 data based on resolution group.
    
    Parameters
    ----------
    group_name : str
        Resolution group name (e.g., 'r10m', 'r20m', 'r60m')
        
    Returns
    -------
    tuple
        (red_band, green_band, blue_band) names
    """
    # Sentinel-2 band mapping by resolution
    rgb_mapping = {
        'r10m': ('b04', 'b03', 'b02'),  # Red, Green, Blue at 10m
        'r20m': ('b04', 'b03', 'b02'),  # Red, Green, Blue at 20m (if available)
        'r60m': ('b04', 'b03', 'b02'),  # Red, Green, Blue at 60m (if available)
    }
    
    return rgb_mapping.get(group_name, ('b04', 'b03', 'b02'))
