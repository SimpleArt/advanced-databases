from collections.abc import Iterator
from typing import Final, Generic, TypeVar

import advanced.databases._src as src

T_co = TypeVar("T_co", covariant=True)

Self = TypeVar("Self", bound="SequenceIsliceProxy")


class SequenceIsliceProxy(Generic[T_co]):
    _sequence: Final["src.viewable_sequence.ViewableSequence[T_co]"]

    __slots__ = {
        "_sequence":
            "The viewable sequence.",
    }

    def __init__(self: Self, sequence: "src.viewable_sequence.ViewableSequence[T_co]", /) -> None:
        assert isinstance(sequence, src.viewable_sequence.ViewableSequence)
        self._sequence = sequence

    def __getitem__(self: Self, index: slice, /) -> "src.sequence_islice.SequenceIslice[T_co]":
        if not isinstance(index, slice):
            raise TypeError(f"expected sequence.islice[start:stop:step], got sequence.islice[{index!r}]")
        # Cast integer-like indices to integers.
        range_ = range(len(self._sequence))[index]
        # Collect None or integers.
        start = None if index.start is None else range_.start
        stop = None if index.stop is None else range_.stop
        step = None if index.step is None else range_.step
        return src.sequence_islice.SequenceIslice(self._sequence, start, stop, step)
