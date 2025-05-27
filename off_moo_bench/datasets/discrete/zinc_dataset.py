from off_moo_bench.datasets.discrete_dataset import DiscreteDataset
from off_moo_bench.disk_resource import DiskResource

ZINC_FILES = ["zinc/zinc-x-0.npy"]
ZINC_TEST_FILES = ["zinc/zinc-test-x-0.npy"]
zinc_FRONTS_FILE = "zinc/zinc_fronts.json"


class ZINCDataset(DiscreteDataset):
    name = "zinc"
    x_name = "in_silico_sequence"
    y_name = "LogP&QED"

    @staticmethod
    def register_x_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in ZINC_FILES
        ]

    @staticmethod
    def register_y_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in ZINC_FILES
        ]

    @staticmethod
    def register_x_test_shards():
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in ZINC_TEST_FILES
        ]

    @staticmethod
    def register_y_test_shards():
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in ZINC_TEST_FILES
        ]

    @classmethod
    def register_fronts_shards(cls):
        return DiskResource(
            zinc_FRONTS_FILE,
            is_absolute=False,
        )

    def __init__(self, **kwargs):
        super(ZINCDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            self.register_fronts_shards(),
            **kwargs
        )
