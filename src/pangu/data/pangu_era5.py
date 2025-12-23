import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PanguERA5Downloader:
    """
    Handles downloading and merging of Pangu-Weather specific ERA5 data.
    Requires Surface and Upper-air components to be merged into one file.
    """

    def __init__(self, client: Any, cdo: Any, dry_run: bool = False):
        """
        Args:
            client: cdsapi.Client instance or similar protocol.
            cdo: cdo.Cdo instance.
            dry_run: If True, skip actual IO.
        """
        self.client = client
        self.cdo = cdo
        self.dry_run = dry_run

    def download(
        self,
        sfc_config: Dict[str, Any],
        pl_config: Dict[str, Any],
        date_params: Dict[str, str],
        output_path: str,
    ) -> None:
        """
        Download surface and upper-air data and merge them.

        Args:
            sfc_config: Configuration for surface data request.
            pl_config: Configuration for upper-air data request.
            date_params: Date/Time parameters (year, month, day, time).
            output_path: Final destination for the merged NetCDF.
        """
        temp_dir = os.path.dirname(output_path)
        timestamp = f"{date_params.get('year')}{date_params.get('month')}{date_params.get('day')}"
        sfc_temp = os.path.join(temp_dir, f"temp_pangu_sfc_{timestamp}.nc")
        pl_temp = os.path.join(temp_dir, f"temp_pangu_pl_{timestamp}.nc")

        try:
            self._download_component(sfc_config, date_params, sfc_temp, "Surface")
            self._download_component(pl_config, date_params, pl_temp, "Upper")

            self._merge_files(sfc_temp, pl_temp, output_path)

        except Exception as e:
            logger.error(f"Pangu download flow failed: {e}")
            raise
        finally:
            self._cleanup([sfc_temp, pl_temp])

    def _download_component(
        self, config: Dict[str, Any], date_params: Dict[str, str], path: str, label: str
    ) -> None:
        """Process request parameters and execute download."""
        req = self._build_request(config, date_params)
        dataset_name = config.get("dataset_name")

        logger.info(f"Downloading Pangu {label} to {path}...")

        if self.dry_run:
            logger.info(f"[DRY RUN] retrieve('{dataset_name}', {req}, '{path}')")
        else:
            self.client.retrieve(dataset_name, req, path)

    def _build_request(
        self, config: Dict[str, Any], date_params: Dict[str, str]
    ) -> Dict[str, Any]:
        """Filter config and merge with date parameters."""
        # excluded keys that are invalid for CDS API
        excluded = {"type", "dataset_name", "surface", "upper", "grid"}

        req = {k: v for k, v in config.items() if k not in excluded}
        req.update(date_params)
        return req

    def _merge_files(self, sfc_path: str, pl_path: str, output_path: str) -> None:
        logger.info(f"Merging components into {output_path}...")

        if self.dry_run:
            logger.info(f"[DRY RUN] cdo.merge {sfc_path} {pl_path} -> {output_path}")
        else:
            self.cdo.merge(input=f"{sfc_path} {pl_path}", output=output_path)
            logger.info("Merge successful.")

    def _cleanup(self, paths: List[str]) -> None:
        if self.dry_run:
            return

        for path in paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"Failed to remove temp file {path}: {e}")
