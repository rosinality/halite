import math
from collections import deque
from statistics import mean, stdev
from typing import TYPE_CHECKING, Iterable

from halite.transformers.tokainfer.engine.monitoring import track_time_decorator

if TYPE_CHECKING:
    from halite.transformers.tokainfer.engine.types import Sequence


def calc_cumulative_mean_stds(
    vals: list[float],
) -> tuple[list[float], list[float]]:
    """
    Given a list vals of length n, calculates for each i in [0, n-1]
    the mean and std of the subsequence vals[i:].
    """
    n = len(vals)
    means = []
    std_devs = []

    # Initialize running sum and sum of squares from the end
    running_sum = sum(vals)
    running_sum_sq = sum(x * x for x in vals)

    for i in range(n):
        # Calculate current mean and std for subsequence[i:]
        count = n - i
        current_mean = running_sum / count
        means.append(current_mean)

        if count == 1:
            std_devs.append(0.0)  # Standard deviation of a single number is 0
        else:
            # Var(X) = E[(X - E[X])^2] = E[X^2] - E[X]^2
            variance = (running_sum_sq / count) - (current_mean * current_mean)
            std_dev = math.sqrt(max(0, variance))
            std_devs.append(std_dev)

        # Remove the contribution of position i for next iteration
        cur_val = vals[i]
        running_sum -= cur_val
        running_sum_sq -= cur_val * cur_val

    return means, std_devs


class EarlyStoppingBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

        self.initialize()

    def initialize(self):
        self.deque = deque[float](maxlen=self.buffer_size)

    def cleanup(self):
        self.deque = None

    def __len__(self):
        return len(self.deque)

    def empty(self):
        return len(self.deque) == 0

    def add(self, value: float):
        assert 0 <= value <= 1.0
        self.deque.append(value)

    def calc_conditional_mean_stds(self, sorted_vals: list[float]):
        """
        For each v in vals, calculate the mean and std of the buffer values
        that are greater than or equal to v.
        """

        sorted_buffer = list(sorted(self.deque))

        cumulative_means, cumulative_stds = calc_cumulative_mean_stds(sorted_buffer)

        conditional_means = []
        conditional_stds = []

        buffer_pos = 0
        for val in sorted_vals:
            assert 0 <= val <= 1.0

            while buffer_pos < len(sorted_buffer) and sorted_buffer[buffer_pos] < val:
                buffer_pos += 1

            if buffer_pos == len(sorted_buffer):
                conditional_mean = 1.0
                conditional_std = 0.0
            else:
                conditional_mean = cumulative_means[buffer_pos]
                conditional_std = cumulative_stds[buffer_pos]

            conditional_means.append(conditional_mean)
            conditional_stds.append(conditional_std)

        return conditional_means, conditional_stds

    # def calc_conditional_mean_stds(self, vals: list[float]):
    #     """
    #     For each v in vals, calculate the mean and std of the buffer values
    #     that are greater than or equal to v.
    #     """

    #     sorted_buffer = list(sorted(self.deque))

    #     # we need the indices too so we can unsort
    #     sorted_val_pairs = list(
    #         sorted([(v, i) for v, i in zip(vals, range(len(vals)), strict=True)])
    #     )
    #     sorted_vals = [v for v, _ in sorted_val_pairs]
    #     sorted_idxs = [i for _, i in sorted_val_pairs]

    #     inverse_sorted_idxs = [0] * len(sorted_idxs)
    #     for i, idx in enumerate(sorted_idxs):
    #         inverse_sorted_idxs[idx] = i

    #     cumulative_means, cumulative_stds = calc_cumulative_mean_stds(sorted_buffer)

    #     conditional_means = []
    #     conditional_stds = []

    #     buffer_pos = 0
    #     for val in sorted_vals:
    #         assert 0 <= val <= 1.0

    #         while buffer_pos < len(sorted_buffer) and sorted_buffer[buffer_pos] < val:
    #             buffer_pos += 1

    #         if buffer_pos == len(sorted_buffer):
    #             conditional_mean = 1.0
    #             conditional_std = 0.0
    #         else:
    #             conditional_mean = cumulative_means[buffer_pos]
    #             conditional_std = cumulative_stds[buffer_pos]

    #         conditional_means.append(conditional_mean)
    #         conditional_stds.append(conditional_std)

    #     isorted_conditional_means = [conditional_means[i] for i in inverse_sorted_idxs]
    #     isorted_conditional_stds = [conditional_stds[i] for i in inverse_sorted_idxs]

    #     return isorted_conditional_means, isorted_conditional_stds

    def mean(self):
        assert len(self.deque) > 0
        return mean(self.deque)

    def std(self):
        assert len(self.deque) > 0
        if len(self.deque) == 1:
            return 0.0
        return stdev(self.deque)


class PredictionMap:
    def __init__(self, means: list[float], stds: list[float], std_buffer_scale: float):
        assert len(means) == len(stds)

        self.means = means
        self.stds = stds
        self.num_buckets = len(means)
        self.std_buffer_scale = std_buffer_scale

    def predict(self, frac: float):
        assert 0 <= frac <= 1.0
        if frac == 1.0:
            return 1.0, 0.0

        continuous_pos = frac * (self.num_buckets - 1)
        lower_bucket = math.floor(continuous_pos)
        upper_bucket = lower_bucket + 1

        lower_bucket_mean = self.means[lower_bucket]
        lower_bucket_std = self.stds[lower_bucket]

        upper_bucket_mean = self.means[upper_bucket]
        upper_bucket_std = self.stds[upper_bucket]

        interp_scale = continuous_pos - lower_bucket

        mean = lower_bucket_mean * (1 - interp_scale) + upper_bucket_mean * interp_scale
        std = lower_bucket_std * (1 - interp_scale) + upper_bucket_std * interp_scale

        return mean, std

        # # quantize to the nearest bucket
        # bucket_idx = round(frac * (self.num_buckets - 1))
        # mean = self.means[bucket_idx]
        # std = self.stds[bucket_idx]

        # return mean, std

    def update_seq_predictions(self, seq: "Sequence"):
        assert seq.completion_scheduled < seq.completion_total
        # since we haven't stopped the sequence yet, we know it will
        # generate at least one more token before stopping
        min_completion_frac = (seq.completion_scheduled + 1) / seq.completion_total

        predicted_completion_frac, predicted_completion_std = self.predict(
            min_completion_frac
        )

        buffered_completion_frac = min(
            predicted_completion_frac
            + predicted_completion_std * self.std_buffer_scale,
            1.0,
        )
        seq._expected_completion_length = round(
            predicted_completion_frac * seq.completion_total
        )
        seq._expected_completion_length_with_buffer = round(
            buffered_completion_frac * seq.completion_total
        )

        assert (
            seq.completion_scheduled
            < seq._expected_completion_length
            <= seq.completion_total
        ), f"seq.completion_scheduled: {seq.completion_scheduled}, seq._expected_completion_length: {seq._expected_completion_length}, seq.completion_total: {seq.completion_total}"
        assert (
            seq.completion_scheduled
            < seq._expected_completion_length_with_buffer
            <= seq.completion_total
        ), f"seq.completion_scheduled: {seq.completion_scheduled}, seq._expected_completion_length_with_buffer: {seq._expected_completion_length_with_buffer}, seq.completion_total: {seq.completion_total}"


class EarlyStoppingTracker:
    def __init__(
        self,
        buffer_size: int,
        initial_wait: int,
        init_mean: float | None = None,
        init_std: float | None = None,
    ):
        if init_mean is None:
            init_mean = 1.0

        if init_std is None:
            init_std = 0.0

        assert buffer_size > 0
        # assert 0 <= initial_wait <= buffer_size

        self.buffer_size = buffer_size
        self.initial_wait = initial_wait
        self.buffer = EarlyStoppingBuffer(buffer_size)

        self.init_mean = init_mean
        self.init_std = init_std

        self.warmup_count = 0

    def initialize(self):
        self.buffer.initialize()
        self.warmup_count = 0

    def cleanup(self):
        self.buffer.cleanup()

    # def add_sequence(self, num_generated: int, max_len: int):
    #     processed_through = num_generated / max_len
    #     self.buffer.add(processed_through)

    def add_finished_sequences(self, seqs: Iterable["Sequence"]):
        check_for_warmup_steps = not self.is_warmed_up() and len(self.buffer) > 0
        if check_for_warmup_steps:
            cur_mean = self.buffer_mean()

        for seq in seqs:
            # NOTE: I think that completion_scheduled is what we want here,
            # since it's the scheduler's perspective we care about wrt using
            # our early stopping predictions.
            frac = seq.completion_scheduled / seq.completion_total
            self.buffer.add(frac)

            if check_for_warmup_steps and frac <= cur_mean:
                self.warmup_count += 1

    def buffer_empty(self):
        return self.buffer.empty()

    def buffer_mean(self):
        return self.buffer.mean()

    def buffer_std(self):
        return self.buffer.std()

    def buffer_len(self):
        return len(self.buffer)

    def is_warmed_up(self):
        return self.warmup_count >= self.initial_wait and self.buffer_len() > 0

    @track_time_decorator()
    def make_prediction_map(
        self, num_buckets: int, std_buffer_scale: float
    ) -> PredictionMap:
        buckets = [i / num_buckets for i in range(num_buckets + 1)]

        if not self.is_warmed_up():
            means = []
            stds = []

            for bucket in buckets:
                if bucket < self.init_mean:
                    means.append(self.init_mean)
                    stds.append(self.init_std)
                else:
                    means.append(1.0)
                    stds.append(0.0)
        else:
            means, stds = self.buffer.calc_conditional_mean_stds(buckets)

        return PredictionMap(means=means, stds=stds, std_buffer_scale=std_buffer_scale)

    def predict_completion_lengths(
        self, seqs: list["Sequence"], std_buffer_scale: float
    ) -> tuple[list[int], list[int]]:
        min_completion_fracs = []

        for seq in seqs:
            assert seq.completion_scheduled < seq.completion_total
            min_completion_fracs.append(
                (seq.completion_scheduled + 1) / seq.completion_total
            )

        if not self.is_warmed_up():
            # heuristic while the server is warming up / buffer is being populated.

            predicted_completion_fracs = []
            predicted_completion_stds = []

            for frac in min_completion_fracs:
                if frac < self.init_mean:
                    predicted_completion_fracs.append(self.init_mean)
                    predicted_completion_stds.append(self.init_std)
                else:
                    predicted_completion_fracs.append(1.0)
                    predicted_completion_stds.append(0.0)

        else:
            predicted_completion_fracs, predicted_completion_stds = (
                self.buffer.calc_conditional_mean_stds(min_completion_fracs)
            )

        predicted_completion_lengths = [
            round(predicted_completion_fracs[i] * seqs[i].completion_total)
            for i in range(len(seqs))
        ]

        buffered_completion_fracs = [
            min(
                predicted_completion_fracs[i]
                + predicted_completion_stds[i] * std_buffer_scale,
                1.0,
            )
            for i in range(len(seqs))
        ]

        buffered_predicted_completion_lengths = [
            round(buffered_completion_fracs[i] * seqs[i].completion_total)
            for i in range(len(seqs))
        ]

        return predicted_completion_lengths, buffered_predicted_completion_lengths

    def predict_completion_lengths_for_unstarted_seqs(
        self, seqs: list["Sequence"], std_buffer_scale: float
    ):
        if not self.is_warmed_up():
            completion_frac = self.init_mean
            completion_std = self.init_std
        else:
            completion_frac = self.buffer.mean()
            completion_std = self.buffer.std()

        buffered_frac = min(
            completion_frac + completion_std * std_buffer_scale,
            1.0,
        )

        completion_lengths = []
        buffered_completion_lengths = []

        for seq in seqs:
            assert seq.completion_scheduled == 0
            predicted_completion_length = round(completion_frac * seq.completion_total)
            buffered_predicted_completion_length = round(
                buffered_frac * seq.completion_total
            )

            completion_lengths.append(predicted_completion_length)
            buffered_completion_lengths.append(buffered_predicted_completion_length)

        return completion_lengths, buffered_completion_lengths
