from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


TEMPERATURE_VARIABLE = "t"
REFERENCE_PRESSURE_HPA = 1000.0
POTENTIAL_TEMPERATURE_KAPPA = 287.05 / 1004.0
DEFAULT_DATASET_PATH = Path(
    "data/global-pressure-level-temperature-stack-for-nov-8-12z-dry-theta-climatology_1990-to-2020_p1-to-1000.nc"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/era5_dry-potential-temperature-climatology_1990-2020_11-08_12.nc"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dry-potential-temperature climatology from an ERA5 pressure-level "
            "temperature stack that contains one matched month-day-hour sample per year."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the ERA5 pressure-level temperature stack.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output dry-potential-temperature climatology NetCDF file.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def compute_dry_potential_temperature(temperature: xr.DataArray) -> xr.DataArray:
    pressure_levels = temperature.coords["pressure_level"].astype(np.float32)
    pressure_factor = (
        REFERENCE_PRESSURE_HPA / pressure_levels
    ) ** POTENTIAL_TEMPERATURE_KAPPA
    theta = temperature.astype(np.float32) * pressure_factor
    theta.name = "theta"
    theta.attrs = {
        "long_name": "Dry potential temperature",
        "units": "K",
        "formula": "theta = T * (1000 hPa / p)^kappa",
        "kappa": float(POTENTIAL_TEMPERATURE_KAPPA),
        "reference_pressure_hpa": float(REFERENCE_PRESSURE_HPA),
        "source_variable": TEMPERATURE_VARIABLE,
    }
    return theta


def build_climatology(theta: xr.DataArray) -> xr.Dataset:
    climatology_mean = theta.mean(dim="valid_time", skipna=True).astype(np.float32)
    climatology_std = theta.std(dim="valid_time", skipna=True, ddof=1).astype(np.float32)
    sample_count = theta.count(dim="valid_time").astype(np.int16)

    climatology_mean.name = "theta_climatology_mean"
    climatology_mean.attrs = {
        "long_name": "Dry potential temperature climatological mean",
        "units": "K",
    }

    climatology_std.name = "theta_climatology_std"
    climatology_std.attrs = {
        "long_name": "Dry potential temperature climatological sample standard deviation",
        "units": "K",
        "ddof": 1,
    }

    sample_count.name = "theta_sample_count"
    sample_count.attrs = {
        "long_name": "Number of valid dry potential temperature samples used in climatology",
        "units": "1",
    }

    time_values = theta.coords["valid_time"].values
    time_strings = [str(value) for value in time_values]

    return xr.Dataset(
        data_vars={
            climatology_mean.name: climatology_mean,
            climatology_std.name: climatology_std,
            sample_count.name: sample_count,
        },
        coords={
            "pressure_level": theta.coords["pressure_level"],
            "latitude": theta.coords["latitude"],
            "longitude": theta.coords["longitude"],
        },
        attrs={
            "title": "ERA5 dry potential temperature climatology",
            "summary": (
                "Dry potential temperature climatology built from a pressure-level ERA5 "
                "temperature stack containing one matched month-day-hour sample per year."
            ),
            "source_dataset": "reanalysis-era5-pressure-levels",
            "source_variable": TEMPERATURE_VARIABLE,
            "sample_count": int(theta.sizes["valid_time"]),
            "valid_time_samples": ", ".join(time_strings),
            "climatology_method": "mean_and_sample_std_over_valid_time",
            "formula": "theta = T * (1000 hPa / p)^kappa",
            "reference_pressure_hpa": float(REFERENCE_PRESSURE_HPA),
            "kappa": float(POTENTIAL_TEMPERATURE_KAPPA),
        },
    )


def main() -> None:
    args = parse_args()
    dataset_path = resolve_path(args.dataset)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = xr.open_dataset(dataset_path, chunks={"valid_time": 1})
    try:
        if TEMPERATURE_VARIABLE not in dataset:
            raise KeyError(
                f"Expected variable '{TEMPERATURE_VARIABLE}' in {dataset_path.name}."
            )
        temperature = dataset[TEMPERATURE_VARIABLE]
        if "valid_time" not in temperature.dims:
            raise ValueError("Expected a 'valid_time' dimension in the temperature stack.")
        theta = compute_dry_potential_temperature(temperature)
        climatology = build_climatology(theta)
        encoding = {
            "theta_climatology_mean": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "theta_climatology_std": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "theta_sample_count": {"zlib": True, "complevel": 4, "dtype": "int16"},
        }
        climatology.to_netcdf(output_path, encoding=encoding)
    finally:
        dataset.close()

    print(f"input_dataset={dataset_path}")
    print(f"output_dataset={output_path}")
    print(f"sample_count={climatology.attrs['sample_count']}")
    sizes = climatology.sizes
    print(
        "grid="
        f"{sizes['pressure_level']} pressure levels x "
        f"{sizes['latitude']} latitudes x "
        f"{sizes['longitude']} longitudes"
    )


if __name__ == "__main__":
    main()
