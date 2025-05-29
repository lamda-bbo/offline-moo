from off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from off_moo_bench.disk_resource import DiskResource

_support_benchmarks = [
    "adaptec1",
    "adaptec2",
    "adaptec3",
    "adaptec4",
    "bigblue1",
    "bigblue3",
]


def _get_x_files_from_name(env_name):
    return [f"{env_name}/{env_name}-x-0.npy"]


def _get_x_test_files_from_name(env_name):
    return [f"{env_name}/{env_name}-test-x-0.npy"]


def _get_fronts_files_from_name(env_name):
    return f"{env_name}/{env_name}_fronts.json"


class PlacementDataset(ContinuousDataset):
    name = "eda_placement"
    x_name = "macro_coordinates"
    y_name = "hpwl&congestion&regularity"

    @classmethod
    def register_x_shards(cls):
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in _get_x_files_from_name(cls.name)
        ]

    @classmethod
    def register_y_shards(cls):
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in _get_x_files_from_name(cls.name)
        ]

    @classmethod
    def register_x_test_shards(cls):
        return [
            DiskResource(
                file,
                is_absolute=False,
            )
            for file in _get_x_test_files_from_name(cls.name)
        ]

    @classmethod
    def register_y_test_shards(cls):
        return [
            DiskResource(
                file.replace("-x-", "-y-"),
                is_absolute=False,
            )
            for file in _get_x_test_files_from_name(cls.name)
        ]

    @classmethod
    def register_fronts_shards(cls):
        return DiskResource(
            _get_fronts_files_from_name(cls.name),
            is_absolute=False,
        )

    def __init__(self, **kwargs):
        self.name = self.name.lower()
        assert self.name in _support_benchmarks
        super(PlacementDataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            self.register_fronts_shards(),
            **kwargs,
        )


class Adaptec1Dataset(PlacementDataset):
    name = "adaptec1"

class Adaptec2Dataset(PlacementDataset):
    name = "adaptec2"

class Adaptec3Dataset(PlacementDataset):
    name = "adaptec3"

class Adaptec4Dataset(PlacementDataset):
    name = "adaptec4"

class Bigblue1Dataset(PlacementDataset):
    name = "bigblue1"

class Bigblue3Dataset(PlacementDataset):
    name = "bigblue3"