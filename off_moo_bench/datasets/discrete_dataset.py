from typing import List

import numpy as np

from off_moo_bench.datasets.dataset_builder import DatasetBuilder


def one_hot(a, num_classes):
    """A helper function that converts integers into a floating
    point one-hot representation using pure numpy:
    https://stackoverflow.com/questions/36960320/
    convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    """

    out = np.zeros((a.size, num_classes), dtype=np.float32)
    out[np.arange(a.size), a.ravel()] = 1.0
    out.shape = a.shape + (num_classes,)
    return out


class DiscreteDataset(DatasetBuilder):
    name = "DiscreteDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        return dict(
            is_logits=self.is_logits, soft_interpolation=self.soft_interpolation
        )

    @property
    def subclass(self):
        return DiscreteDataset

    def __init__(
        self,
        *args,
        is_logits=False,
        num_classes_on_each_position: List[int] = [2, 2],
        soft_interpolation=0.6,
        **kwargs
    ):
        self.soft_interpolation = soft_interpolation
        self.num_classes_on_each_position = num_classes_on_each_position
        self.is_logits = is_logits

        # initialize the dataset using the method in the base class
        super(DiscreteDataset, self).__init__(*args, **kwargs)

    def batch_transform(self, x_batch, y_batch, return_x=True, return_y=True):
        # convert the design values from integers to logits
        if self.is_logits and return_x and np.issubdtype(x_batch.dtype, np.integer):
            x_batch = self.to_logits(x_batch)

        # convert the design values from logits to integers
        elif (
            not self.is_logits
            and return_x
            and np.issubdtype(x_batch.dtype, np.floating)
        ):
            x_batch = self.to_integers(x_batch)

        # return processed batches of designs an predictions
        return super(DiscreteDataset, self).batch_transform(
            x_batch, y_batch, return_x=return_x, return_y=return_y
        )

    def update_x_statistics(self):
        # handle corner case when we need statistics but they were
        # not computed yet and the dataset is currently mapped to integers
        original_is_logits = self.is_logits
        self.is_logits = True
        super(DiscreteDataset, self).update_x_statistics()
        self.is_logits = original_is_logits

    def map_normalize_x(self):
        # check that the dataset is in a form that supports normalization
        if not self.is_logits:
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_normalize_x()

    def normalize_x(self, x):
        # check that the dataset is in a form that supports normalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        return super(DiscreteDataset, self).normalize_x(x)

    def map_denormalize_x(self):
        # check that the dataset is in a form that supports denormalization
        if not self.is_logits:
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_denormalize_x()

    def denormalize_x(self, x):
        # check that the dataset is in a form that supports denormalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        return super(DiscreteDataset, self).denormalize_x(x)

    def help_to_logits(
        self, x: np.ndarray, num_classes: int, soft_interpolation: float = 0.6
    ):
        """
        Convert the integers to logits.
        :param x: np.ndarray: the input data, with shape (n_samples, 1)
        :param num_classes: int: the number of classes
        :param soft_interpolation: float: the soft interpolation parameter
        :return: np.ndarray: the logits, with shape (n_samples, 1, n_classes)
        """
        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("cannot convert non-integers to logits")

        # convert the integers to one hot vectors
        one_hot_x = one_hot(x, num_classes)

        # build a uniform distribution to interpolate between
        uniform_prior = np.full_like(one_hot_x, 1 / float(num_classes))

        # interpolate between a dirac distribution and a uniform prior
        soft_x = (
            soft_interpolation * one_hot_x + (1.0 - soft_interpolation) * uniform_prior
        )

        # convert to log probabilities
        x = np.log(soft_x)

        # remove one degree of freedom caused by \sum_i p_i = 1.0
        return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)

    def to_logits(
        self,
        x: np.ndarray,
        soft_interpolation: float = 0.6,
    ):
        """
        Since this is a sequence of categorical variables, we need to
        convert the integers at each position to logits. Then we will
        concatenate the logits for each position to get the final logits
        :param x: np.ndarray: the input data, with shape (n_samples, seq_len)
        :param num_classes_on_each_position: list[int]: the number of classes on each position
        :param soft_interpolation: float: the soft interpolation parameter
        :return: np.ndarray: the logits, with shape (n_samples, seq_len, n_classes)
        """
        num_classes = self.num_classes_on_each_position
        num_classes = [i + 1 for i in num_classes]
        # if count[i] = 1, then we will have a single class, so we add 1
        # to the such cases, this is only happening for RFP-Exact-v0
        # treat this as adding a dummy class
        num_classes = [i + 1 if i == 1 else i for i in num_classes]
        logits = []
        sequence_length = len(num_classes)
        for i in range(sequence_length):
            temp_x = x[:, i].reshape(-1, 1)
            logits.append(
                self.help_to_logits(temp_x, num_classes[i], soft_interpolation)
            )
        return np.concatenate(logits, axis=2).squeeze()

    def help_to_integers(self, x: np.ndarray, true_num_of_classes: int):
        """
        Convert the logits to integers.
        :param x: np.ndarray: the input data, with shape (n_samples, 1, n_classes)
        :param true_num_of_classes: int: the true number of classes
        :return: np.ndarray: the integers, with shape (n_samples, 1)
        """
        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot convert non-floats to integers")

        # Since we might add a dummy class, we need to make sure that the last class is not selected
        # For RFP-Exact-v0
        if true_num_of_classes == 1:
            return np.zeros(x.shape[:-1], dtype=np.int32)

        # add an additional component of zero and find the class
        # with maximum probability
        return np.argmax(
            np.pad(x, [[0, 0]] * (len(x.shape) - 1) + [[1, 0]]), axis=-1
        ).astype(np.int32)

    def to_integers(self, x: np.ndarray):
        """
        Since this is a sequence of categorical variables, we need to
        convert the logits at each position to integers. Then we will
        concatenate the integers for each position to get the final integers
        :param x: np.ndarray: the input data, with shape (n_samples, seq_len, n_classes)
        :param num_classes_on_each_position: list[int]: the number of classes on each position
        :return: np.ndarray: the integers, with shape (n_samples, seq_len)
        """
        true_num_classes = self.num_classes_on_each_position
        true_num_classes = [i + 1 for i in true_num_classes]
        # if count[i] = 1, then we will have a single class, so we
        # add 1 to the such cases, this is only happening for RFP-Exact-v0
        # treat this as adding a dummy class
        num_classes = [i + 1 if i == 1 else i for i in true_num_classes]
        integers = []
        start = 0
        sequence_length = len(num_classes)
        for i in range(sequence_length):
            temp_x = x[:, start : (start + num_classes[i] - 1)].reshape(
                -1, 1, num_classes[i] - 1
            )
            integers.append(self.help_to_integers(temp_x, num_classes[i]))
            start += num_classes[i] - 1
        return np.concatenate(integers, axis=1)

    def map_to_logits(self):
        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_logits:
            # set the appropriate state variable
            self.is_logits = True

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

            for x0 in self.iterate_test_samples(return_y=False):
                assert self.input_shape == x0.shape
                assert self.input_size == int(np.prod(x0.shape))
                assert self.input_dtype == x0.dtype
                break

    def map_to_integers(self):
        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if self.is_logits:
            # if design values are normalized then denorm
            if self.is_normalized_x:
                self.map_denormalize_x()

            # set the appropriate state variable
            self.is_logits = False

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

            for x0 in self.iterate_test_samples(return_y=False):
                assert self.input_shape == x0.shape
                assert self.input_size == int(np.prod(x0.shape))
                assert self.input_dtype == x0.dtype
                break
