from metadata import MetaData
import pytest


@pytest.fixture
def md_h5log():
    """Scan with h5 log but without image log (typical P05 type)."""
    return MetaData('/asap3/petra3/gpfs/p05/2020/data/11008476/raw/hzb_108_F6-32900wh/')


@pytest.fixture()
def md_imlog():
    """Scan with empty h5 log and image log (temporary P07 type)."""
    return MetaData('/asap3/petra3/gpfs/p07/2020/data/11010172/raw/swerim_21_12_oh_a/')


@pytest.fixture()
def md_dpc():
    """DPC scan"""
    return 0


class TestMetaData:
    def test_metadata_exeptions(self):
        with pytest.raises(ValueError):
            MetaData('')

