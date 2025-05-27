from off_moo_bench.datasets.discrete_dataset import DiscreteDataset
from off_moo_bench.disk_resource import DiskResource

RFP_FILES = ["rfp/rfp-x-0.npy"]
RFP_TEST_FILES = ["rfp/rfp-test-x-0.npy"]
rfp_FRONTS_FILE = "rfp/rfp_fronts.json"


class RFPDataset(DiscreteDataset):
    name = "rfp"
    x_name = "red_fluorescent_protein_sequence"
    y_name = "stability&SASA"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in RFP_FILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in RFP_FILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in RFP_TEST_FILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in RFP_TEST_FILES
        ]

    @classmethod
    def register_fronts_shards(cls):
        return DiskResource(
            rfp_FRONTS_FILE,
            is_absolute=False,
        )

    def __init__(self, **kwargs):
        super(RFPDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            self.register_fronts_shards(),
            **kwargs
        )
