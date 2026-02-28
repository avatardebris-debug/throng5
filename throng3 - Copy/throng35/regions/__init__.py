"""Regions package for Throng3.5."""

from throng35.regions.region_base import RegionBase
from throng35.regions.striatum import StriatumRegion
from throng35.regions.cortex import CortexRegion
from throng35.regions.hippocampus import HippocampusRegion

__all__ = ['RegionBase', 'StriatumRegion', 'CortexRegion', 'HippocampusRegion']
