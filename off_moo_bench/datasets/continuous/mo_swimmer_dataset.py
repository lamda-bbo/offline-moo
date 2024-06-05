from off_moo_bench.datasets.continuous_dataset import ContinuousDataset
from off_moo_bench.disk_resource import DiskResource 
import pickle 

MO_SWIMMER_V2_FILES = ["mo_swimmer_v2/mo_swimmer_v2-x-0.npy"]
MO_SWIMMER_V2_TEST_FILES = ["mo_swimmer_v2/mo_swimmer_v2-test-x-0.npy"]
PARAMS_SHAPES_FILE = "mo_swimmer_v2/params_shapes.pkl"

class MOSwimmerV2Dataset(ContinuousDataset):
    
    name = "mo_swimmer_v2"
    x_name = "policy_weights"
    y_name = "sum_return"
    
    @staticmethod
    def register_x_shards():
        return [DiskResource(file, is_absolute=False,)
               for file in MO_SWIMMER_V2_FILES]
    
    @staticmethod
    def register_y_shards():
        return [DiskResource(file.replace('-x-', '-y-'), is_absolute=False,)
               for file in MO_SWIMMER_V2_FILES]
        
    @staticmethod
    def register_x_test_shards():
        return [DiskResource(file, is_absolute=False,)
               for file in MO_SWIMMER_V2_TEST_FILES]
    
    @staticmethod
    def register_y_test_shards():
        return [DiskResource(file.replace('-x-', '-y-'), is_absolute=False,)
               for file in MO_SWIMMER_V2_TEST_FILES]
    
    def __init__(self, **kwargs):
        self.params_shapes = DiskResource(
            PARAMS_SHAPES_FILE,
            is_absolute=False
        )
        assert self.params_shapes.is_downloaded, \
            f"{self.params_shapes.disk_target} not found"
            
        super(MOSwimmerV2Dataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            self.register_x_test_shards(),
            self.register_y_test_shards(),
            **kwargs
        )
        with open(self.params_shapes.disk_target, "rb+") as f:
            self.params_shapes = pickle.load(f)