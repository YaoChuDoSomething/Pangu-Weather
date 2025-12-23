import argparse
import yaml
import os
import logging
import cdsapi
from datetime import datetime, timedelta
from typing import Dict, Any, List
from cdo import Cdo
from src.pangu.data.pangu_era5 import PanguERA5Downloader

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ERA5Downloader:
    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.client = cdsapi.Client() if not dry_run else None

    def download(self, dataset_name: str, params: Dict[str, Any], output_path: str):
        # Construct CDS request
        # Filter out metadata keys that shouldn't go to CDS
        cds_params = {
            k: v
            for k, v in params.items()
            if k
            not in [
                "type",
                "dataset_name",
                "surface",
                "upper",
                "components",
                "grid",
                "param_template",
                "level_template",
                "variables",
                "levels",
            ]
        }

        request = cds_params.copy()

        # Ensure area is list if strictly required, but usually config has it
        if "area" not in request:
            # Default global if not specified
            request["area"] = [90, 0, -90, 360]

        logger.info(f"Preparing to download {dataset_name} to {output_path}")

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would call cdsapi.retrieve('{dataset_name}', {request}, '{output_path}')"
            )
        else:
            try:
                self.client.retrieve(dataset_name, request, output_path)
                logger.info("Download successful.")
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")


class GFSDownloader:
    def __init__(self, config: Dict[str, Any], dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        try:
            from src.pangu.data.gfs_gdex import GFSGDEXDownloader

            self.downloader = GFSGDEXDownloader()
        except ImportError:
            logger.warning("Could not import GFSGDEXDownloader. GFS download may fail.")
            self.downloader = None

    def download(
        self, dataset_name: str, params: Dict[str, Any], output_path: str, date_str: str
    ):
        if not self.downloader and not self.dry_run:
            logger.error("GFS downloader not available. Skipping.")
            return

        # Transform YAML params to GDEX control params
        variables = params.get("variables", [])
        levels = params.get("levels", [])

        param_str = "/".join(variables)
        level_str = "/".join(levels) if levels else ""

        # Apply templates if present
        param_template = params.get("param_template", "{vars}")
        level_template = params.get("level_template", "{levels}")

        if "{vars}" in param_template:
            param_str = param_template.replace("{vars}", param_str)

        if "{levels}" in level_template:
            level_str = level_template.replace("{levels}", level_str)
        elif level_template != "{levels}":
            # Static level string from config
            level_str = level_template

        control_data = {
            "dataset": dataset_name,
            "date": date_str,
            "param": param_str,
            "level": level_str,
            "targetdir": os.path.dirname(output_path),
            "oformat": "netCDF",
        }

        # Add other pass-through params
        for k, v in params.items():
            if k not in [
                "variables",
                "levels",
                "param_template",
                "level_template",
                "type",
                "dataset_name",
                "surface",
                "upper",
            ]:
                control_data[k] = v

        logger.info(f"Preparing to download GFS {dataset_name} for {date_str}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would submit GDEX request: {control_data}")
        else:
            try:
                # Use the new API 'run' instead of 'submit_and_download'
                self.downloader.run(
                    control_data, output_dir=os.path.dirname(output_path)
                )
            except Exception as e:
                logger.error(f"Failed to download GFS: {e}")


def parse_datetime(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="Pangu-Weather Data Preparation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_prepare.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate downloads without executing"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    common = config.get("common", {})
    output_dir = common.get("output_dir", "data/input_data")
    os.makedirs(output_dir, exist_ok=True)

    start_str = common.get("start_date")
    end_str = common.get("end_date")
    step_hours = int(common.get("time_step_hours", 1))

    if not start_str or not end_str:
        logger.error("Both start_date and end_date must be defined in common.")
        return

    start_dt = parse_datetime(start_str)
    end_dt = parse_datetime(end_str)

    logger.info(f"Processing range: {start_dt} to {end_dt} (Step: {step_hours}h)")

    # Group timestamps by date (YYYY-MM-DD) -> list of HH:MM
    dates_to_times: Dict[str, List[str]] = {}
    current = start_dt
    while current <= end_dt:
        d_str = current.strftime("%Y-%m-%d")
        t_str = current.strftime("%H:%M")
        if d_str not in dates_to_times:
            dates_to_times[d_str] = []
        dates_to_times[d_str].append(t_str)
        current += timedelta(hours=step_hours)

    datasets = config.get("datasets", {})

    # Setup dependencies
    cds_client = cdsapi.Client() if not args.dry_run else None
    cdo_instance = Cdo()
    if not args.dry_run:
        try:
            cdo_instance.check()
        except Exception as e:
            logger.warning(f"CDO check failed: {e}")

    era5_downloader = ERA5Downloader(config, args.dry_run)
    gfs_downloader = GFSDownloader(config, args.dry_run)

    # Inject dependencies
    pangu_downloader = PanguERA5Downloader(
        client=cds_client, cdo=cdo_instance, dry_run=args.dry_run
    )

    for day_str, times in dates_to_times.items():
        logger.info(f"Processing Date: {day_str} with {len(times)} time steps.")

        for name, ds_config in datasets.items():
            ds_type = ds_config.get("type")
            dataset_name = ds_config.get(
                "dataset_name"
            )  # Can be None for pangu combined

            # In flat YAML, ds_config IS the params dict
            params = ds_config.copy()

            # Skip if it's a component meant for inclusion only
            if ds_type.startswith("cds_pangu_") and ds_type != "cds_pangu_combined":
                continue

            if ds_type == "cds_pangu_combined":
                # For Pangu, we want per-timestamp files: era5_YYYYMMDD_HHMM.nc
                dt_day = datetime.strptime(day_str, "%Y-%m-%d")

                for t in times:
                    hour_str = t.replace(":", "")
                    output_filename = f"era5_{dt_day.strftime('%Y%m%d')}_{hour_str}.nc"
                    output_file = os.path.join(output_dir, output_filename)

                    # Create single-time usage params
                    single_time_params = params.copy()
                    single_time_params["time"] = [t]

                    # Prepare configs for surface and upper
                    # Logic needs to find surface and upper keys
                    surface_key = params.get("surface")
                    upper_key = params.get("upper")

                    sfc_config = datasets.get(surface_key)
                    pl_config = datasets.get(upper_key)

                    if not sfc_config or not pl_config:
                        logger.error(
                            f"Missing config for components: {surface_key}, {upper_key}"
                        )
                        continue

                    date_params = {
                        "year": dt_day.strftime("%Y"),
                        "month": dt_day.strftime("%m"),
                        "day": dt_day.strftime("%d"),
                        "time": [t],
                    }

                    pangu_downloader.download(
                        sfc_config, pl_config, date_params, output_file
                    )

            elif ds_type == "cds":
                # Download daily file with all times
                dt_day = datetime.strptime(day_str, "%Y-%m-%d")
                update_params = {
                    "year": dt_day.strftime("%Y"),
                    "month": dt_day.strftime("%m"),
                    "day": dt_day.strftime("%d"),
                    "time": times,
                }
                params.update(update_params)

                output_file = os.path.join(output_dir, f"{name}_{day_str}.grib")
                era5_downloader.download(dataset_name, params, output_file)

            elif ds_type == "gdex":
                # GFS usually handled differently
                start_t = times[0].replace(":", "")
                end_t = times[-1].replace(":", "")
                day_clean = day_str.replace("-", "")

                date_param = f"{day_clean}{start_t}/to/{day_clean}{end_t}"

                output_file = os.path.join(output_dir, f"{name}_{day_str}.nc")
                gfs_downloader.download(dataset_name, params, output_file, date_param)

    logger.info("Data preparation process finished.")


if __name__ == "__main__":
    main()
