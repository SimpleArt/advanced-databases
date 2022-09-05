from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from typing import Generic, TypeVar, overload

from .mutable_sequence_islice_proxy import MutableSequenceIsliceProxy
from .viewable_sequence import ViewableSequence

T = TypeVar("T")

Self = TypeVar("Self", bound="ViewableMutableSequence")


class ViewableMutableSequence(ViewableSequence[T], MutableSequence[T], ABC, Generic[T]):

    __slots__ = ()

    @overload
    def __getitem__(self: Self, index: int, /) -> T: ...

    @overload
    def __getitem__(self: Self, index: slice, /) -> "ViewableMutableSequence[T]": ...

    @abstractmethod
    def __getitem__(self, index, /):
        raise NotImplementedError(f"__getitem__ is a required method for viewable mutable sequences")

    @property
    def islice(self: Self, /) -> MutableSequenceIsliceProxy[T]:
        return MutableSequenceIsliceProxy(self)
