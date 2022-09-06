import operator
import pickle
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from itertools import chain, islice
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Generic, Optional, SupportsIndex, Type
from typing import TypeVar, Union, overload

from .index import NestedIndex
from .mutable_sequence_islice import MutableSequenceIslice
from .viewable_mutable_sequence import ViewableMutableSequence

ET = TypeVar("ET", bound=BaseException)
T = TypeVar("T")

Self = TypeVar("Self", bound="BigList")

CHUNKSIZE = 1024
CHUNKSIZE_EXTENDED = 1536

assert CHUNKSIZE * 3 == CHUNKSIZE_EXTENDED * 2

def ensure_file(path: Path, default: T) -> T:
    if path.exists():
        with open(path, mode="rb") as file:
            return pickle.load(file)
    else:
        with open(path, mode="wb") as file:
            pickle.dump(default, file)
        return default


class BigList(ViewableMutableSequence[T], Generic[T]):
    _cache: Final[OrderedDict[int, list[T]]]
    _filenames: Final[list[int]]
    _index: Final[NestedIndex]
    _path: Final[Path]

    __slots__ = {
        "_cache":
            "A cache containing the 4 most recently accessed segments.",
        "_filenames":
            "The file names for each segment.",
        "_index":
            "A data structure for fast repeated indexing.",
        "_path":
            "The folder containing all of the files.",
    }

    def __init__(self: Self, path: Union[PathLike, bytes, str], /) -> None:
        path = Path(path).resolve()
        path.mkdir(exist_ok=True)
        (path / "list").mkdir(exist_ok=True)
        ensure_file(path / "list" / "counter.txt", 0)
        self._cache = OrderedDict()
        self._filenames = ensure_file(path / "list" / "filenames.txt", [])
        self._index = NestedIndex(ensure_file(path / "list" / "lens.txt", []))
        self._path = path

    def __delitem__(self: Self, index: Union[int, slice], /) -> None:
        if isinstance(index, slice):
            len_ = len(self)
            range_ = range(len_)[index]
            size = len(range_)
            if range_.step < 0:
                range_ = range_[::-1]
            if size == len_:
                self.clear()
                return
            elif size == 0:
                return
            elif range_.step == 1 and range_.start == 0:
                i, j = self._index.find(range_.stop)
                for k in range(i):
                    (self._path / "list" / f"{self._filenames[k]}.txt").unlink()
                del self._filenames[:i]
                del self._index[:i]
                del self._cache_chunk(0)[:j]
                self._index.update(0, -j)
                self._balance(0)
            elif range_.step == 1 and range_.stop == len(self):
                i, j = self._index.find(range_.start)
                for k in range(i + 1, len(self._filenames)):
                    (self._path / "list" / f"{self._filenames[k]}.txt").unlink()
                del self._filenames[i + 1 :]
                del self._index[i + 1 :]
                del self._cache_chunk(-1)[j:]
                self._index[-1] = j
                self._balance(-1)
            elif range_.step == 1:
                start = self._index.find(range_.start)
                stop = self._index.find(range_.stop)
                del self._cache_chunk(stop[0])[:stop[1]]
                self._index.update(stop[0], -stop[1])
                for k in range(start[0] + 1, stop[0]):
                    (self._path / "list" / f"{self._filenames[k]}.txt").unlink()
                del self._filenames[start[0] + 1 : stop[0]]
                del self._index[start[0] + 1 : stop[0]]
                del self._cache_chunk(start[0])[start[1]:]
                self._index[start[0]] = start[1]
                self._balance(start[0])
            else:
                for i in reversed(range_):
                    del self[i]
        else:
            index = range(len(self))[index]
            i, j = self._index.find(index)
            del self._cache_chunk(i)[j]
            self._index.update(i, -1)
            self._balance(i)

    def __enter__(self: Self, /) -> Self:
        return self

    def __exit__(
        self: Self,
        exc_type: Optional[Type[ET]],
        exc_val: Optional[ET],
        exc_traceback: Optional[TracebackType],
        /,
    ) -> None:
        self.commit()

    @overload
    def __getitem__(self: Self, index: int, /) -> T: ...

    @overload
    def __getitem__(self: Self, index: slice, /) -> MutableSequenceIslice[T]: ...

    def __getitem__(self, index, /):
        if isinstance(index, slice):
            range_ = range(len(self))[index]
            return self.islice[index]
        else:
            index = range(len(self))[index]
            i, j = self._index.find(index)
            return self._cache_chunk(i)[j]

    def __getstate__(self: Self, /) -> Path:
        return self._path

    def __islice(self: Self, start: int, stop: int, step: int, /) -> Iterator[T]:
        f_start = self._index.find(start)
        f_stop = self._index.find(stop)
        if step > 0:
            n = f_start[1]
            for i in range(f_start[0], f_stop[0]):
                chunk = self._cache_chunk(i)
                while n < len(chunk):
                    yield chunk[n]
                    n += step
                n -= len(chunk)
            chunk = self._cache_chunk(f_stop[0])
            for i in range(n, f_stop[1] + 1, step):
                yield chunk[i]
        else:
            chunk = self._cache_chunk(f_start[0])
            n = f_start[1] - len(chunk)
            for i in reversed(range(f_stop[0], f_start[0])):
                while n > -len(chunk):
                    yield chunk[n]
                    n += step
                n += len(chunk)
                chunk = self._cache_chunk(i)
            for i in range(n, f_stop[1] - len(chunk) - 1, step):
                yield chunk[i]

    def __islice__(self: Self, start: Optional[int], stop: Optional[int], step: Optional[int], /) -> Iterator[T]:
        range_ = range(len(self))[start:stop:step]
        if len(range_) == len(self):
            return iter(self) if range_.step == 1 else reversed(self)
        elif len(range_) == 0:
            return (self[i] for i in range_)
        elif range_.step == 1 and range_.start == 0:
            return islice(self, range_.stop)
        elif range_.step == 1 and range_.stop == len(self):
            f_start = self._index.find(range_.start)
            if f_start[1] == 0:
                return chain.from_iterable(
                    self._cache_chunk(i)
                    for i in range(f_start[0], len(self._filenames))
                )
            else:
                return chain(
                    self._cache_chunk(f_start[0])[f_start[1]:],
                    chain.from_iterable(
                        self._cache_chunk(i)
                        for i in range(f_start[0] + 1, len(self._filenames))
                    ),
                )
        elif range_.step == 1:
            f_start = self._index.find(range_.start)
            f_stop = self._index.find(range_.stop)
            if f_start[0] == f_stop[0]:
                return iter(self._cache_chunk(f_start[0])[f_start[1]:f_stop[1]])
            elif f_start[1] == 0 == f_stop[1]:
                return chain.from_iterable(
                    self._cache_chunk(i)
                    for i in range(f_start[0], f_stop[0])
                )
            elif f_start[1] == 0:
                return chain(
                    chain.from_iterable(
                        self._cache_chunk(i)
                        for i in range(f_start[0], f_stop[0])
                    ),
                    self._cache_chunk(f_stop[0])[:f_stop[1]],
                )
            elif f_stop == 0:
                return chain(
                    self._cache_chunk(f_start[0])[f_start[1]:],
                    chain.from_iterable(
                        self._cache_chunk(i)
                        for i in range(f_start[0] + 1, f_stop[0])
                    ),
                )
            else:
                return chain(
                    self._cache_chunk(f_start[0])[f_start[1]:],
                    chain.from_iterable(
                        self._cache_chunk(i)
                        for i in range(f_start[0] + 1, f_stop[0])
                    ),
                    self._cache_chunk(f_stop[0])[:f_stop[1]],
                )
        elif range_.step == -1 and range_.start + 1 == len(self._filenames):
            return islice(reversed(self), len(range_))
        elif range_.step == -1 and range_.stop + 1 == 0:
            f_start = self._index.find(range_.start)
            if f_start[1] + 1 == self._index[f_start[0]]:
                return chain.from_iterable(reversed(self._cache_chunk(i)) for i in range(f_start[0], -1, -1))
            else:
                return chain(
                    self._cache_chunk(f_start[0])[f_start[1]::-1],
                    chain.from_iterable(
                        reversed(self._cache_chunk(i))
                        for i in reversed(range(f_start[0]))
                    ),
                )
        elif range_.step == -1:
            f_start = self._index.find(range_.start)
            f_stop = self._index.find(range_.stop)
            if f_start[0] == f_stop[0]:
                return iter(self._cache_chunk(f_start[0])[f_start[1]:f_stop[1]:-1])
            elif (
                f_start[1] + 1 == self._index[f_start[0]]
                and f_stop[1] + 1 == self._index[f_stop[0]]
            ):
                return chain.from_iterable(
                    reversed(self._cache_chunk(i))
                    for i in range(f_start[0], f_stop[0], -1)
                )
            elif f_start[1] + 1 == self._index[f_start[0]]:
                return chain(
                    chain.from_iterable(
                        reversed(self._cache_chunk(i))
                        for i in range(f_start[0], f_stop[0], -1)
                    ),
                    self._cache_chunk(f_stop[0])[:f_stop[1]:-1],
                )
            elif f_stop + 1 == self._index[f_stop[0]]:
                return chain(
                    self._cache_chunk(f_start[0])[f_start[1]::-1],
                    chain.from_iterable(
                        reversed(self._cache_chunk(i))
                        for i in range(f_start[0] - 1, f_stop[0], -1)
                    ),
                )
            else:
                return chain(
                    self._cache_chunk(f_start[0])[f_start[1]::-1],
                    chain.from_iterable(
                        reversed(self._cache_chunk(i))
                        for i in range(f_start[0] - 1, f_stop[0], -1)
                    ),
                    self._cache_chunk(f_stop[0])[:f_stop[1]:-1],
                )
        elif abs(range_.step) < CHUNKSIZE * 2:
            return self.__islice(range_.start, range_.stop, range_.step)
        else:
            return super().__islice__(range_.start, range_.stop, range_.step)

    def __iter__(self: Self, /) -> Iterator[T]:
        return chain.from_iterable(
            self._cache_chunk(i)
            for i, _ in enumerate(self._filenames)
        )

    def __len__(self: Self, /) -> int:
        return self._index.total_length()

    def __repr__(self: Self, /) -> str:
        return f"{type(self).__name__}({self._path})"

    def __reversed__(self: Self, /) -> Iterator[T]:
        return chain.from_iterable(
            reversed(self._cache_chunk(~i))
            for i, _ in enumerate(self._filenames)
        )

    @overload
    def __setitem__(self: Self, index: int, value: T, /) -> None: ...

    @overload
    def __setitem__(self: Self, index: slice, value: Iterable[T], /) -> None: ...

    def __setitem__(self, index, value, /):
        if isinstance(index, slice):
            raise NotImplementedError("big lists do not support slice assignments")
        index = range(len(self))[index]
        i, j = self._index.find(index)
        self._cache_chunk(i)[j] = value

    def __setstate__(self: Self, path: Path, /) -> None:
        type(self).__init__(self, path)

    def _balance(self: Self, index: int, /) -> None:
        indexer = self._index
        if len(self) == 0:
            self.clear()
            return
        elif len(self._filenames) != 1:
            pass
        elif indexer[0] > 2 * CHUNKSIZE:
            chunk = self._cache_chunk(0)
            self._filenames.append(self._get_filename())
            self._cache[self._filenames[-1]] = chunk[len(chunk) // 2:]
            indexer.update(0, -(len(chunk) + 1) // 2)
            del chunk[len(chunk) // 2 :]
            indexer.append(len(chunk))
            return
        else:
            return
        index = range(len(self._filenames))[index]
        if index == 0:
            if indexer[0] + indexer[1] < CHUNKSIZE:
                indexer.update(0, indexer[1])
                self._cache_chunk(0).extend(self._pop_chunk(1))
            elif indexer[0] + indexer[1] > 4 * CHUNKSIZE:
                chunk = [
                    *self._cache_chunk(0),
                    *self._cache_chunk(1),
                ]
                self._cache_chunk(0)[:] = chunk[: len(chunk) // 3]
                indexer[0] = len(self._cache_chunk(0))
                self._cache_chunk(1)[:] = chunk[len(chunk) // 3 : 2 * len(chunk) // 3]
                indexer[1] = len(self._cache_chunk(1))
                del chunk[: 2 * len(chunk) // 3]
                self._filenames.insert(2, self._get_filename())
                self._free_cache()
                self._cache[self._filenames[2]] = chunk
                indexer.insert(2, len(chunk))
            elif (
                CHUNKSIZE // 2 < indexer[0] < CHUNKSIZE * 2
                and CHUNKSIZE_EXTENDED < indexer[0] + indexer[1] < 3 * CHUNKSIZE
            ):
                pass
            elif indexer[0] > indexer[1]:
                diff = indexer[0] - indexer[1]
                self._cache_chunk(1)[:0] = self._cache_chunk(0)[-diff // 2 :]
                del self._cache_chunk(0)[-diff // 2:]
                indexer.update(0, len(self._cache_chunk(0)) - indexer[0])
                indexer.update(1, len(self._cache_chunk(1)) - indexer[1])
            else:
                diff = indexer[1] - indexer[0]
                self._cache_chunk(0).extend(self._cache_chunk(1)[: diff // 2])
                del self._cache_chunk(1)[:diff // 2]
                indexer.update(0, len(self._cache_chunk(0)) - indexer[0])
                indexer.update(1, len(self._cache_chunk(1)) - indexer[1])
        elif index + 1 == len(self._filenames):
            if indexer[-1] + indexer[-2] < CHUNKSIZE:
                indexer.update(-2, indexer[-1])
                self._cache_chunk(index - 1).extend(self._pop_chunk(index))
            elif indexer[-1] + indexer[-2] > 4 * CHUNKSIZE:
                chunk = [
                    *self._cache_chunk(-2),
                    *self._cache_chunk(-1),
                ]
                self._cache_chunk(-2)[:] = chunk[: len(chunk) // 3]
                indexer[-2] = len(self._cache_chunk(-2))
                self._cache_chunk(-1)[:] = chunk[len(chunk) // 3 : 2 * len(chunk) // 3]
                indexer[-1] = len(self._cache_chunk(-1))
                del chunk[: 2 * len(chunk) // 3]
                self._filenames.append(self._get_filename())
                self._free_cache()
                self._cache[self._filenames[-1]] = chunk
                indexer.append(len(chunk))
            elif (
                CHUNKSIZE // 2 < indexer[-1] < CHUNKSIZE * 2
                and CHUNKSIZE_EXTENDED < indexer[-1] + indexer[-2] < 3 * CHUNKSIZE
            ):
                pass
            elif indexer[-1] < indexer[-2]:
                diff = indexer[-2] - indexer[-1]
                self._cache_chunk(-1)[:0] = self._cache_chunk(-2)[-diff // 2 :]
                del self._cache_chunk(-2)[-diff // 2 :]
                indexer.update(-1, len(self._cache_chunk(-1)) - indexer[-1])
                indexer.update(-2, len(self._cache_chunk(-2)) - indexer[-2])
            else:
                diff = indexer[-1] - indexer[-2]
                self._cache_chunk(-2).extend(self._cache_chunk(-1)[: diff // 2])
                del self._cache_chunk(-1)[: diff // 2]
                indexer.update(-1, len(self._cache_chunk(-1)) - indexer[-1])
                indexer.update(-2, len(self._cache_chunk(-2)) - indexer[-2])
        else:
            if indexer[index - 1] + indexer[index] + indexer[index + 1] < CHUNKSIZE_EXTENDED:
                chunk = [
                    *self._cache_chunk(index - 1),
                    *self._cache_chunk(index),
                    *self._pop_chunk(index + 1),
                ]
                self._cache_chunk(index - 1)[:] = chunk[: len(chunk) // 2]
                self._cache_chunk(index)[:] = chunk[len(chunk) // 2 :]
                indexer[index - 1] = len(chunk) // 2
                indexer[index] = (len(chunk) + 1) // 2
            elif indexer[index - 1] + indexer[index] + indexer[index + 1] > 6 * CHUNKSIZE:
                chunk = [
                    *self._cache_chunk(index - 1),
                    *self._cache_chunk(index),
                    *self._cache_chunk(index + 1),
                ]
                self._cache_chunk(index - 1)[:] = chunk[: len(chunk) // 4]
                indexer[index - 1] = len(self._cache_chunk(index - 1))
                self._cache_chunk(index)[:] = chunk[len(chunk) // 4 : len(chunk) // 2]
                indexer[index] = len(self._cache_chunk(index))
                self._cache_chunk(index + 1)[:] = chunk[len(chunk) // 2 : 3 * len(chunk) // 4]
                indexer[index + 1] = len(self._cache_chunk(index + 1))
                del chunk[: 3 * len(chunk) // 4]
                self._filenames.insert(index + 2, self._get_filename())
                self._free_cache()
                self._cache[self._filenames[index + 2]] = chunk
                indexer.insert(index + 2, len(chunk))
            elif not all(CHUNKSIZE // 2 < 2 * L // 3 < CHUNKSIZE for L in indexer[index - 1 : index + 2]):
                chunk = [
                    *self._cache_chunk(index - 1),
                    *self._cache_chunk(index),
                    *self._cache_chunk(index + 1),
                ]
                self._cache_chunk(index - 1)[:] = chunk[: len(chunk) // 3]
                self._cache_chunk(index)[:] = chunk[len(chunk) // 3 : 2 * len(chunk) // 3]
                self._cache_chunk(index + 1)[:] = chunk[2 * len(chunk) // 3 :]
                indexer.update(index - 1, len(self._cache_chunk(index - 1)) - indexer[index - 1])
                indexer.update(index, len(self._cache_chunk(index)) - indexer[index])
                indexer.update(index + 1, len(self._cache_chunk(index + 1)) - indexer[index + 1])

    def _cache_chunk(self: Self, index: int, /) -> list[T]:
        filename = self._filenames[index]
        if filename in self._cache:
            self._cache.move_to_end(filename)
        else:
            self._free_cache()
            with open(self._path / "list" / f"{filename}.txt", mode="rb") as file:
                self._cache[filename] = pickle.load(file)
        return self._cache[filename]

    def _commit_chunk(self: Self, filename: int, segment: list[T], /) -> None:
        with open(self._path / "list" / f"{filename}.txt", mode="wb") as file:
            pickle.dump(segment, file)

    def _del_chunk(self: Self, index: int, /) -> None:
        index = range(len(self._filenames))[index]
        filename = self._filenames.pop(index)
        (self._path / "list" / f"{filename}.txt").unlink()
        del self._index[index]
        self._cache.pop(filename, None)

    def _free_cache(self: Self, /) -> None:
        while len(self._cache) >= 4:
            self._commit_chunk(*self._cache.popitem(last=False))

    def _from_iterable(cls: Type[Self], iterable: Iterable[T], /) -> Self:
        raise NotImplementedError(
            "cannot create big instances, create a destination to save"
            " into instead."
        )

    def _get_filename(self: Self, /) -> int:
        path = self._path
        with open(path / "list" / "counter.txt", mode="rb") as file:
            uid = pickle.load(file)
        with open(path / "list" / "counter.txt", mode="wb") as file:
            pickle.dump(uid + 1, file)
        with open(path / "list" / f"{uid}.txt", mode="wb") as file:
            pickle.dump([], file)
        return uid

    def _pop_chunk(self: Self, index: int, /) -> list[T]:
        filename = self._filenames[index]
        if filename in self._cache:
            segment = self._cache.pop(filename)
        else:
            with open(self._path / "list" / f"{filename}.txt", mode="rb") as file:
                segment = pickle.load(file)
        self._del_chunk(index)
        return segment

    def append(self: Self, value: T, /) -> None:
        if len(self) == 0:
            self._filenames.append(self._get_filename())
            self._cache[self._filenames[-1]] = [value]
            self._index.append(1)
            return
        else:
            self._cache_chunk(-1).append(value)
            self._index.update(-1, 1)
            self._balance(-1)

    def clear(self: Self, /) -> None:
        path = self._path
        for filename in self._filenames:
            (path / "list" / f"{filename}.txt").unlink()
        with open(path / "list" / "counter.txt", mode="wb") as file:
            pickle.dump(0, file)
        with open(path / "list" / "filenames.txt", mode="wb") as file:
            pickle.dump([], file)
        with open(path / "list" / "lens.txt", mode="wb") as file:
            pickle.dump([], file)
        self._cache.clear()
        self._filenames.clear()
        self._index.clear()

    def commit(self: Self, /) -> None:
        path = self._path
        with open(path / "list" / "filenames.txt", mode="wb") as file:
            pickle.dump(self._filenames, file)
        with open(path / "list" / "lens.txt", mode="wb") as file:
            pickle.dump(self._index._data, file)
        for filename, segment in self._cache.items():
            self._commit_chunk(filename, segment)

    def extend(self: Self, iterable: Iterable[T], /) -> None:
        filenames = self._filenames
        indexer = self._index
        if not isinstance(iterable, Iterable):
            raise TypeError(f"extend expected iterable, got {iterable!r}")
        elif isinstance(iterable, list):
            if len(iterable) == 0:
                return
            elif len(self._filenames) == 1 and indexer[0] < CHUNKSIZE_EXTENDED:
                offset = CHUNKSIZE_EXTENDED - indexer[0]
                self._cache_chunk(0).extend(iterable[:offset])
                offset -= CHUNKSIZE_EXTENDED - len(self._cache_chunk(0))
                indexer.update(0, len(self._cache_chunk(0)) - indexer[0])
            else:
                offset = 0
            for i in range(offset, len(iterable), CHUNKSIZE_EXTENDED):
                chunk = iterable[i : i + CHUNKSIZE_EXTENDED]
                filenames.append(self._get_filename())
                self._commit_chunk(filenames[-1], chunk)
                indexer.append(len(chunk))
        else:
            iterator = iter(iterable)
            if len(self._filenames) == 1 and indexer[0] < CHUNKSIZE_EXTENDED:
                self._cache_chunk(0).extend(islice(iterator, CHUNKSIZE_EXTENDED - indexer[0]))
                indexer.update(0, len(self._cache_chunk(0)) - indexer[0])
            while True:
                chunk = [*islice(iterator, CHUNKSIZE_EXTENDED)]
                if not chunk:
                    break
                filenames.append(self._get_filename())
                self._commit_chunk(filenames[-1], chunk)
                indexer.append(len(chunk))

    def insert(self: Self, index: int, value: T, /) -> None:
        index = operator.index(index)
        if len(self._filenames) == 0:
            return self.append(value)
        elif index < 0:
            index += len(self)
            if index < 0:
                index = 0
        elif index >= len(self):
            return self.append(value)
        i, j = self._index.find(index)
        self._cache_chunk(i).insert(j, value)
        self._index.update(i, 1)
        self._balance(i)

    def reverse(self: Self, /) -> None:
        for i, _ in enumerate(self._filenames):
            self._cache_chunk(i).reverse()
        self._filenames.reverse()
        self._index.reverse()
