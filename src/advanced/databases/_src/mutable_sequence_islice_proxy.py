from typing import Generic, TypeVar, Union, overload

import advanced.databases._src as src
from .sequence_islice_proxy import SequenceIsliceProxy

T = TypeVar("T")

Self = TypeVar("Self", bound="MutableSequenceIsliceProxy")


class MutableSequenceIsliceProxy(SequenceIsliceProxy[T], Generic[T]):
    _sequence: "src.viewable_mutable_sequence.ViewableMutableSequence[T]"

    __slots__ = ()

    def __init__(self: Self, sequence: "src.viewable_mutable_sequence.ViewableMutableSequence[T]", /) -> None:
        assert isinstance(sequence, src.viewable_mutable_sequence.ViewableMutableSequence)
        self._sequence = sequence

    def __getitem__(self: Self, index: slice, /) -> "src.mutable_sequence_islice.MutableSequenceIslice[T]":
        if not isinstance(index, slice):
            raise TypeError(f"expected sequence.islice[start:stop:step], got sequence.islice[{index!r}]")
        # Cast integer-like indices to integers.
        range_ = range(len(self._sequence))[index]
        # Collect None or integers.
        start = None if index.start is None else range_.start
        stop = None if index.stop is None else range_.stop
        step = None if index.step is None else range_.step
        return src.mutable_sequence_islice.MutableSequenceIslice(self._sequence, start, stop, step)
