import operator
from collections.abc import Iterable, Iterator, MutableSequence, Sequence
from itertools import accumulate, groupby
from typing import Optional, Union, overload


class FenwickIndex(MutableSequence[int]):
    """
    A fenwick tree used to track nested indexing. Useful
    for segmented arrays or implicitly segmented arrays.

    Examples
    --------
    Tracking elements inside of nested data:
        >>> data = [[0, 1, 2, 3], [4, 5, 6]]
        >>> indexes = FenwickIndex(map(len, data))
        >>> 
        >>> indexes.find(3)
        (0, 3)
        >>> data[0][3]
        3
        >>> 
        >>> indexes.find(5)
        (1, 1)
        >>> data[1][1]
        5
        >>> 
        >>> data[0].append(-1)
        >>> indexes.update(0, +1)
        >>> 
        >>> indexes.find(5)
        (1, 0)
        >>> data[1][0]
        4
        >>> 
        >>> del data[0][-1]
        >>> indexes.update(0, -1)
        >>> 
        >>> indexes.find(5)
        (1, 1)
        >>> data[1][1]
        5

    Implicitly nested data inside of flat data:
        >>> data = [0, 1, 2, 3, 4, 5, 6]
        >>> # Implicitly split into 4 and 3 items.
        >>> indexes = FenwickIndex([4, 3])
        >>> 
        >>> indexes.upto(0), indexes[0]
        (0, 4)
        >>> data[0 : 0 + 4]
        [0, 1, 2, 3]
        >>> 
        >>> indexes.upto(1), indexes[1]
        (4, 3)
        >>> data[4 : 4 + 3]
        [4, 5, 6]
    """
    _data: list[int]
    _fenwick: Optional[list[int]]
    _len: Optional[int]

    __slots__ = ("_data", "_fenwick", "_len")

    def __init__(self, data: Iterable[int] = ()) -> None:
        self._data = [*map(operator.index, data)]
        self._fenwick = None
        self._len = None

    def __delitem__(self, index: Union[int, slice]) -> None:
        """
        Delete a segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> del data[0]
            >>> del indexes[0]
            >>> 
            >>> indexes
            FenwickIndex([3])
        """
        data = self._data
        if isinstance(index, slice):
            deleted = len(range(len(data))[index])
            if deleted == len(data):
                self.clear()
            elif deleted > 0:
                del data[index]
                self._fenwick = None
            if self._len is not None:
                self._len -= deleted
        else:
            index = range(len(data))[index]
            fenwick = self._fenwick
            if fenwick is not None:
                if index + 1 == len(data):
                    del fenwick[-1]
                else:
                    self._fenwick = None
            if self._len is None:
                del self._data[index]
            else:
                self._len -= self._data.pop(index)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index):
        """
        Get the length of a segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> indexes[0]
            4
        """
        return self._data[index]

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the length of each segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> print(*indexes)
            4 3
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Returns the amount of segments.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> print(len(indexes))
            2
        """
        return len(self._data)

    def __repr__(self) -> str:
        """
        Returns a string representation of the fenwick array.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> indexes
            FenwickIndex([4, 3])
        """
        return f"{type(self).__name__}({self._data})"

    @overload
    def __setitem__(self, index: int, value: int) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[int]) -> None: ...

    def __setitem__(self, index, value):
        """
        Set the length of a segment.

        Equivalent to self.update(index, value - self[index]),
        but performs less efficiently.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> data[0].append(-1)
            >>> indexes[0] = len(data[0])
            >>> indexes[0]
            5
        """
        if isinstance(index, slice):
            self._fenwick = None
            self._data[index] = value
            self._len = None
        else:
            self.update(index, value - self[index])

    def append(self, value: int) -> None:
        """
        Append the length of a new segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> data.append([7, 8, 9])
            >>> indexes.append(3)
            >>> print(*indexes)
            4 3 3
        """
        self._data.append(value)
        if self._len is not None:
            self._len += value
        fenwick = self._fenwick
        if fenwick is not None:
            i = len(fenwick)
            j = i & -i
            fenwick.append(value)
            while j > 1:
                j //= 2
                fenwick[i] += fenwick[i - j]

    def clear(self) -> None:
        """
        Clear the segments.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> data.clear()
            >>> indexes.clear()
            >>> indexes
            FenwickIndex([])
        """
        self._data.clear()
        self._fenwick = None
        self._len = None

    def find(self, index: int) -> tuple[int, int]:
        """
        Find the segment and position in the segment for an index.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> indexes.find(3)
            (0, 3)
            >>> data[0][3]
            3
            >>> 
            >>> indexes.find(5)
            (1, 1)
            >>> data[1][1]
            5
        """
        if self._len is None:
            self.total_length()
        index = range(self._len)[index]
        fenwick = self._fenwick
        if fenwick is None:
            fenwick = [0, *self._data]
            fenwick_len = len(fenwick)
            for i in range(1, fenwick_len):
                j = i + (i & -i)
                if j < fenwick_len:
                    fenwick[j] += fenwick[i]
            self._fenwick = fenwick
        else:
            fenwick_len = len(self._fenwick)
        i = 0
        j = 1 << fenwick_len.bit_length()
        while j > 0:
            i += j
            if i < fenwick_len and fenwick[i] <= index:
                index -= fenwick[i]
            else:
                i -= j
            j //= 2
        return (i, index)

    def insert(self, index: int, value: int) -> None:
        """
        Insert the length of a segment before an index.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> data.insert(0, [-1] * 5)
            >>> indexes.insert(0, 5)
            >>> 
            >>> indexes
            FenwickIndex([5, 4, 3])
        """
        data = self._data
        index = operator.index(index)
        if index < 0:
            index += len(data)
        if index >= len(data):
            self.append(value)
        else:
            data.insert(index, value)
            self._fenwick = None
        if self._len is not None:
            self._len += value

    def reverse(self) -> None:
        self._data.reverse()
        self._fenwick = None

    def total_length(self) -> int:
        """
        Returns the combined length of all segments.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> indexes.total_length()
            7
        """
        if self._len is None:
            if len(self._data) == 0:
                self._len = 0
            else:
                self._len = self.upto(-1) + self._data[-1]
        return self._len

    def update(self, index: int, value: int) -> None:
        """
        Update the length of a segment.

        Equivalent to self[index] += value,
        but performs more efficiently.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = FenwickIndex(map(len, data))
            >>> 
            >>> data[0].append(-1)
            >>> indexes.update(0, 1)
            >>> indexes[0]
            5
        """
        index = range(len(self._data))[index]
        self._data[index] += value
        if self._len is not None:
            self._len += value
        fenwick = self._fenwick
        if fenwick is not None:
            fenwick_len = len(fenwick)
            index = range(1, fenwick_len)[index]
            if index & (index - 1) == 0:
                while index < fenwick_len:
                    fenwick[index] += value
                    index *= 2
            else:
                while index < fenwick_len:
                    fenwick[index] += value
                    index += index & -index

    def upto(self, index: int) -> int:
        """
        Find the length of all segments upto (not including) a segment.

        Example
        -------
            >>> data = [0, 1, 2, 3, 4, 5, 6]
            >>> # Implicitly split into 4 and 3 items.
            >>> indexes = FenwickIndex([4, 3])
            >>> 
            >>> indexes.upto(0), indexes[0]
            (0, 4)
            >>> data[0 : 0 + 4]
            [0, 1, 2, 3]
            >>> 
            >>> indexes.upto(1), indexes[1]
            (4, 3)
            >>> data[4 : 4 + 3]
            [4, 5, 6]
        """
        index = range(len(self._data))[index]
        fenwick = self._fenwick
        if fenwick is None:
            fenwick = [0, *self._data]
            fenwick_len = len(fenwick)
            for i in range(1, fenwick_len):
                j = i + (i & -i)
                if j < fenwick_len:
                    fenwick[j] += fenwick[i]
            self._fenwick = fenwick
        else:
            fenwick_len = len(self._fenwick)
        result = 0
        while index > 0:
            result += fenwick[index]
            index -= index & -index
        return result


class NestedIndex(MutableSequence[int]):
    """
    Tracks nested indexes where groups of consecutive segments are
    expected to have the same sizes, similar to numpy arrays but with
    more flexibility.

    Examples
    --------
    Tracking elements inside of nested data:
        >>> data = [[0, 1, 2, 3], [4, 5, 6]]
        >>> indexes = NestedIndex(map(len, data))
        >>> 
        >>> indexes.find(3)
        (0, 3)
        >>> data[0][3]
        3
        >>> 
        >>> indexes.find(5)
        (1, 1)
        >>> data[1][1]
        5
        >>> 
        >>> data[0].append(-1)
        >>> indexes.update(0, +1)
        >>> 
        >>> indexes.find(5)
        (1, 0)
        >>> data[1][0]
        4
        >>> 
        >>> del data[0][-1]
        >>> indexes.update(0, -1)
        >>> 
        >>> indexes.find(5)
        (1, 1)
        >>> data[1][1]
        5

    Implicitly nested data inside of flat data:
        >>> data = [0, 1, 2, 3, 4, 5, 6]
        >>> # Implicitly split into 4 and 3 items.
        >>> indexes = NestedIndex([4, 3])
        >>> 
        >>> indexes.upto(0), indexes[0]
        (0, 4)
        >>> data[0 : 0 + 4]
        [0, 1, 2, 3]
        >>> 
        >>> indexes.upto(1), indexes[1]
        (4, 3)
        >>> data[4 : 4 + 3]
        [4, 5, 6]
    """
    _fenwick: FenwickIndex
    _segments: FenwickIndex

    __slots__ = ("_fenwick", "_segments")

    def __init__(self, data: Iterable[int] = ()) -> None:
        fenwick = []
        segments = []
        for key, group in groupby(map(operator.index, data)):
            fenwick.append(key)
            segments.append(sum(1 for _ in group))
        self._fenwick = FenwickIndex(x * y for x, y in zip(fenwick, segments))
        self._segments = FenwickIndex(segments)

    def __delitem__(self, index: Union[int, slice]) -> None:
        """
        Delete a segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> del data[0]
            >>> del indexes[0]
            >>> 
            >>> indexes
            NestedIndex([3])
        """
        fenwick = self._fenwick
        segments = self._segments
        if isinstance(index, slice):
            range_ = range(len(segments))[index]
            if len(range_) == len(data):
                self.clear()
            elif len(range_) > 0:
                if range_.step > 0:
                    range_ = range_[::-1]
                for i in range_:
                    del self[i]
        else:
            i = segments.find(index)[0]
            if segments[i] == 1:
                del fenwick[i]
                del segments[i]
            else:
                fenwick.update(i, -fenwick[i] // segments[i])
                segments.update(i, -1)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index):
        """
        Get the length of a segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> indexes[0]
            4
        """
        if isinstance(index, slice):
            range_ = range(len(self))[index]
            return [self[index] for i in range_]
        else:
            i = self._segments.find(index)[0]
            return self._fenwick[i] // self._segments[i]

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over the length of each segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> print(*indexes)
            4 3
        """
        return (
            x // y
            for x, y in zip(self._fenwick, self._segments)
            for _ in range(y)
        )

    def __len__(self) -> int:
        """
        Returns the amount of segments.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> print(len(indexes))
            2
        """
        return self._segments.total_length()

    def __repr__(self) -> str:
        """
        Returns a string representation of the fenwick array.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> indexes
            NestedIndex([4, 3])
        """
        return f"{type(self).__name__}({[*self]})"

    @overload
    def __setitem__(self, index: int, value: int) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[int]) -> None: ...

    def __setitem__(self, index, value):
        """
        Set the length of a segment.

        Equivalent to self.update(index, value - self[index]),
        but performs less efficiently.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> data[0].append(-1)
            >>> indexes[0] = len(data[0])
            >>> indexes[0]
            5
        """
        if isinstance(index, slice):
            range_ = range(len(self))[index]
            if not isinstance(value, Sequence):
                value = [*value]
            if range_.step == 1:
                iterator = iter(value)
                for i, x in zip(range_, iterator):
                    self[i] = x
                if len(range_) < len(value):
                    if range_.stop == len(self):
                        self.extend(iterator)
                    else:
                        for i, x in enumerate(iterator, range_.stop):
                            self.insert(i, x)
                elif len(range_) > len(value):
                    del self[range_.start + len(value) : range_.stop]
            elif len(range_) != len(value):
                raise ValueError(f"attempt to assign sequence of size {len(value)} to extended slice of size {len(range_)}")
            else:
                for i, x in zip(range_, value):
                    self[i] = x
        else:
            self.update(index, value - self[index])

    def append(self, value: int) -> None:
        """
        Append the length of a new segment.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> data.append([7, 8, 9])
            >>> indexes.append(3)
            >>> print(*indexes)
            4 3 3
        """
        fenwick = self._fenwick
        segments = self._segments
        if len(self) == 0:
            fenwick.append(value)
            segments.append(1)
        elif fenwick[-1] // segments[-1] == value:
            fenwick.update(-1, value)
            segments.update(-1, +1)
        else:
            fenwick.append(value)
            segments.append(1)

    def clear(self) -> None:
        """
        Clear the segments.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> data.clear()
            >>> indexes.clear()
            >>> indexes
            NestedIndex([])
        """
        self._fenwick.clear()
        self._segments.clear()

    def find(self, index: int) -> tuple[int, int]:
        """
        Find the segment and position in the segment for an index.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> indexes.find(3)
            (0, 3)
            >>> data[0][3]
            3
            >>> 
            >>> indexes.find(5)
            (1, 1)
            >>> data[1][1]
            5
        """
        i, j = self._fenwick.find(index)
        j, k = divmod(j, self._fenwick[i] // self._segments[i])
        return self._segments.upto(i) + j, k

    def insert(self, index: int, value: int) -> None:
        """
        Insert the length of a segment before an index.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> data.insert(0, [-1] * 5)
            >>> indexes.insert(0, 5)
            >>> 
            >>> indexes
            NestedIndex([5, 4, 3])
        """
        fenwick = self._fenwick
        segments = self._segments
        i, j = segments.find(index)
        fi = fenwick[i]
        si = segments[i]
        fs = fi // si
        if fs == value:
            fenwick.update(i, value)
            segments.update(i, value)
        elif j == 0:
            if i > 0 and fenwick[i - 1] // segments[i - 1] == value:
                fenwick.update(i - 1, value)
                segments.update(i - 1, 1)
            else:
                fenwick.insert(i, value)
                segments.insert(i, 1)
        elif j + 1 == si:
            if i + 1 < len(fenwick) and fenwick[i + 1] // segments[i + 1] == value:
                fenwick.update(i + 1, value)
                segments.update(i + 1, 1)
            elif i + 1 == len(fenwick):
                fenwick.append(value)
                segments.append(1)
            else:
                fenwick.insert(i + 1, value)
                segments.insert(i + 1, 1)
        else:
            fenwick.insert(i + 1, value)
            fenwick.insert(i + 2, fi - j * fs)
            fenwick.update(i, j * fs - fi)
            segments.insert(i + 1, 1)
            segments.insert(i + 2, si - j)
            segments.update(i, j - si)

    def reverse(self) -> None:
        self._fenwick.reverse()
        self._segments.reverse()

    def total_length(self) -> int:
        """
        Returns the combined length of all segments.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> indexes.total_length()
            7
        """
        return self._fenwick.total_length()

    def update(self, index: int, value: int) -> None:
        """
        Update the length of a segment.

        Equivalent to self[index] += value,
        but performs more efficiently.

        Example
        -------
            >>> data = [[0, 1, 2, 3], [4, 5, 6]]
            >>> indexes = NestedIndex(map(len, data))
            >>> 
            >>> data[0].append(-1)
            >>> indexes.update(0, 1)
            >>> indexes[0]
            5
        """
        index = range(len(self))[index]
        if value == 0:
            return
        fenwick = self._fenwick
        segments = self._segments
        i, j = segments.find(index)
        fi = fenwick[i]
        si = segments[i]
        fs = fi // si
        left = None if i == 0 else fenwick[i - 1] // segments[i - 1]
        right = None if i + 1 == len(fenwick) else fenwick[i + 1] // segments[i + 1]
        if si == 1 and left != fs + value != right:
            fenwick.update(i, value)
            segments.update(i, 1)
        elif si == 1:
            if fs + value != left:
                del fenwick[i]
                fenwick.update(i, right)
                del segments[i]
                segments.update(i, 1)
            elif fs + value != right:
                del fenwick[i]
                fenwick.update(i - 1, left)
                del segments[i]
                segments.update(i - 1, 1)
            else:
                del fenwick[i]
                fenwick.update(i - 1, left)
                del fenwick[i]
                del segments[i]
                segments.update(i - 1, segments[i] + 1)
                del segments[i]
        elif j == 0:
            if fs + value == left:
                fenwick.update(i - 1, left)
                fenwick.update(i, -fs)
                segments.update(i - 1, 1)
                segments.update(i, -1)
            else:
                fenwick.update(i, -fs)
                fenwick.insert(i, fs + value)
                segments.update(i, -1)
                segments.insert(i, 1)
        elif j + 1 == si:
            if fs + value == right:
                fenwick.update(i, -fs)
                fenwick.update(i + 1, right)
                segments.update(i, -1)
                segments.update(i + 1, 1)
            else:
                fenwick.update(i, -fs)
                fenwick.insert(i + 1, fs + value)
                segments.update(i, -1)
                segments.insert(i + 1, 1)
        else:
            fenwick.insert(i + 1, fs + value)
            fenwick.insert(i + 2, fi - j * fs)
            fenwick.update(i, j * fs - fi)
            segments.insert(i + 1, 1)
            segments.insert(i + 2, si - j)
            segments.update(i, j - si)

    def upto(self, index: int) -> int:
        """
        Find the length of all segments upto (not including) a segment.

        Example
        -------
            >>> data = [0, 1, 2, 3, 4, 5, 6]
            >>> # Implicitly split into 4 and 3 items.
            >>> indexes = NestedIndex([4, 3])
            >>> 
            >>> indexes.upto(0), indexes[0]
            (0, 4)
            >>> data[0 : 0 + 4]
            [0, 1, 2, 3]
            >>> 
            >>> indexes.upto(1), indexes[1]
            (4, 3)
            >>> data[4 : 4 + 3]
            [4, 5, 6]
        """
        fenwick = self._fenwick
        segments = self._segments
        i, j = segments.find(index)
        return fenwick.upto(i) + fenwick[i] // segments[i] * j
