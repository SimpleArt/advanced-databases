from __future__ import annotations
from typing import Final, Generic, Optional, TypeVar, overload

from .viewable_sequence import ViewableSequence

T_co = TypeVar("T_co", covariant=True)

Self = TypeVar("Self", bound="SequenceIslice")


class SequenceIslice(ViewableSequence[T_co], Generic[T_co]):
    _sequence: Final[ViewableSequence[T_co]]
    _start: Optional[int]
    _stop: Optional[int]
    _step: Optional[int]

    __slots__ = {
        "_sequence":
            "The viewable sequence.",
        "_start":
            "The starting index.",
        "_stop":
            "The stopping index.",
        "_step":
            "The step size.",
    }

    def __init__(
        self: Self,
        sequence: ViewableSequence[T_co],
        start: Optional[int],
        stop: Optional[int],
        step: Optional[int],
        /,
    ) -> None:
        assert isinstance(sequence, ViewableSequence)
        self._sequence = sequence
        self._start = start
        self._stop = stop
        self._step = step

    @overload
    def __getitem__(self: Self, index: int, /) -> T_co: ...

    @overload
    def __getitem__(self: Self, index: slice, /) -> SequenceIslice[T_co]: ...

    def __getitem__(self, index, /):
        if isinstance(index, slice):
            range_ = range(len(self._sequence))[index]
            start = None if index.start is None else range_.start
            stop = None if index.stop is None else range_.stop
            step = None if index.step is None else range_.step
            return type(self)(self, start, stop, step)
        else:
            index = range(len(self._sequence))[self._start:self._stop:self._step][index]
            return self._sequence[index]

    def __islice__(self: Self, start: Optional[int], stop: Optional[int], step: Optional[int], /) -> Iterator[T_co]:
        range_ = range(len(self._sequence))[self._start:self._stop:self._step][start:stop:step]
        return self._sequence.__islice__(range_.start, range_.stop, range_.step)

    def __iter__(self: Self, /) -> Iterator[T_co]:
        return self._sequence.__islice__(self._start, self._stop, self._step)

    def __len__(self: Self, /) -> int:
        return len(range(len(self._sequence))[self._start:self._stop:self._step])

    def __repr__(self: Self, /) -> str:
        start = "" if self._start is None else repr(self._start)
        stop = "" if self._stop is None else repr(self._stop)
        step = "" if self._step is None else f":{self._step!r}"
        return f"{self._sequence!r}[{start}:{stop}{step}]"

    def __reversed__(self: Self, /) -> Iterator[T_co]:
        return self.__islice__(None, None, -1)
