"""Retired configurations must fail rather than silently select another route."""
import pytest

from spherepackgen.config.loader import config_from_mapping
from spherepackgen.domain.enums import SpatialOrderClass
from spherepackgen.generators.registry import get_generator


@pytest.mark.parametrize("value", ["stealthy_hyperuniform", "shu", "shu3d", "stealthy", "stealthy_hu", "stealthy_hyperuniform_3d"])
def test_retired_structure_is_rejected(value):
    with pytest.raises(ValueError, match="Unknown SpatialOrderClass"):
        SpatialOrderClass.from_value(value)


@pytest.mark.parametrize("name", ["rcpgenerator", "rcp_generator", "random_close_packing", "stealthy_hyperuniform", "shu", "shu3d", "collective_coordinate"])
def test_retired_algorithm_is_rejected(name, tmp_path):
    config = config_from_mapping({"physical": {"mean_diameter": "200 nm", "packing_fraction": .1},
                                  "algorithm": {"name": name}}, tmp_path)
    with pytest.raises(NotImplementedError, match="not available in this release"):
        get_generator(config)
