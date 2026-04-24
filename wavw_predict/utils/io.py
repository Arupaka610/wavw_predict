from pathlib import Path
import numpy as np
import pandas as pd


def read_netcdf(path: str | Path, variables: list[str] | None = None) -> dict:
    """Read NetCDF file and return dict of numpy arrays."""
    import netCDF4 as nc  # noqa: PLC0415
    with nc.Dataset(path, "r") as ds:
        if variables is None:
            variables = list(ds.variables.keys())
        return {v: np.array(ds.variables[v][:]) for v in variables if v in ds.variables}


def write_netcdf(path: str | Path, data: dict, dims: dict, attrs: dict | None = None) -> None:
    """Write dict of numpy arrays to NetCDF file."""
    import netCDF4 as nc  # noqa: PLC0415
    with nc.Dataset(path, "w") as ds:
        for dim_name, dim_size in dims.items():
            ds.createDimension(dim_name, dim_size)
        for var_name, arr in data.items():
            shape = arr.shape
            dim_names = list(dims.keys())[:len(shape)]
            var = ds.createVariable(var_name, arr.dtype, dim_names)
            var[:] = arr
        if attrs:
            ds.setncatts(attrs)


def read_hdf5(path: str | Path, key: str = "/") -> dict:
    """Read HDF5 file and return dict of numpy arrays."""
    import h5py  # noqa: PLC0415
    result = {}
    with h5py.File(path, "r") as f:
        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = obj[()]
        f.visititems(_visitor)
    return result


def read_csv_timeseries(path: str | Path, time_col: str = "time") -> pd.DataFrame:
    """Read CSV time series and parse datetime index."""
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.set_index(time_col).sort_index()
    return df


def read_bathymetry(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read bathymetry grid. Returns (depth, lon, lat) arrays."""
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".nc4"}:
        data = read_netcdf(path)
        depth_key = next((k for k in ("depth", "elevation", "z", "topo") if k in data), None)
        if depth_key is None:
            raise KeyError(f"No depth variable found in {path}. Keys: {list(data.keys())}")
        depth = data[depth_key]
        lon = data.get("lon", data.get("longitude", np.arange(depth.shape[1])))
        lat = data.get("lat", data.get("latitude", np.arange(depth.shape[0])))
        return depth, lon, lat
    elif suffix in {".npy"}:
        return np.load(path), np.array([]), np.array([])
    raise ValueError(f"Unsupported bathymetry format: {suffix}")
