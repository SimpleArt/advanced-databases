import pickle
from bisect import bisect
from collections import OrderedDict
from collections.abc import Set as AbstractSet, Iterable, Iterator
from collections.abc import MutableSet
from itertools import chain, groupby, islice
from operator import length_hint
from os import PathLike
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Generic, Optional, SupportsIndex, Type
from typing import TypeVar, Union, overload

ET = TypeVar("ET", bound=BaseException)
T = TypeVar("T")

Self = TypeVar("Self", bound="BigSortedSet")

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


class BigSortedSet(MutableSet[T], Generic[T]):
    _cache: Final[OrderedDict[str, list[T]]]
    _filenames: Final[list[Path]]
    _len: int
    _lens: Final[list[int]]
    _mins: Final[list[T]]
    _path: Final[Path]

    __slots__ = {
        "_cache":
            "A cache containing the 16 most recently accessed segments.",
        "_filenames":
            "The file names for each segment.",
        "_len":
            "The length of the entire set.",
        "_lens":
            "The lengths of every segment.",
        "_mins":
            "The first element of each segment.",
        "_path":
            "The folder containing all of the files.",
    }

    def __init__(self: Self, path: Union[PathLike, bytes, str], /) -> None:
        path = Path(path).resolve()
        path.mkdir(exist_ok=True)
        (path / "sorted_set").mkdir(exist_ok=True)
        ensure_file(path / "sorted_set" / "counter.txt", 0)
        self._cache = OrderedDict()
        self._filenames = ensure_file(path / "sorted_set" / "filenames.txt", [])
        self._lens = ensure_file(path / "sorted_set" / "lens.txt", [])
        self._len = sum(self._lens)
        self._mins = ensure_file(path / "sorted_set" / "mins.txt", [])
        self._path = path

    def __contains__(self: Self, element: Any, /) -> bool:
        try:
            i = bisect(self._mins, element)
        except TypeError:
            return False
        if i == 0:
            return False
        j = bisect(self._cache_chunk(i - 1), element)
        return j > 0 and self._cache_chunk(i - 1)[j - 1] == element

    def __delitem__(self: Self, element: T, /) -> T:
        i = bisect(self._mins, element)
        if i == 0:
            raise KeyError(element)
        j = bisect(self._cache_chunk(i - 1), element)
        if j == 0 or element != self._cache_chunk(i - 1)[j - 1]:
            raise KeyError(element)
        del self._cache_chunk(i - 1)[j - 1]
        self._len -= 1
        self._lens[i - 1] -= 1
        if j == 1 and self._lens[i - 1] > 0:
            self._mins[i - 1] = self._cache_chunk(i - 1)[0]
        self._balance(i - 1)

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

    def __getstate__(self: Self, /) -> Path:
        return self._path

    def __iter__(self: Self, /) -> Iterator[T]:
        return chain.from_iterable(
            self._cache_chunk(i)
            for i, _ in enumerate(self._filenames)
        )

    def __len__(self: Self, /) -> int:
        return self._len

    def __repr__(self: Self, /) -> str:
        return f"{type(self).__name__}({self._path})"

    def __setstate__(self: Self, path: Path, /) -> None:
        type(self).__init__(self, path)

    def _balance(self: Self, index: int, /) -> None:
        lens = self._lens
        mins = self._mins
        if self._len == 0:
            self.clear()
            return
        elif len(lens) != 1:
            pass
        elif lens[0] > 2 * CHUNKSIZE:
            chunk = self._cache_chunk(0)
            self._filenames.append(self._get_filename())
            self._cache[self._filenames[-1]] = chunk[len(chunk) // 2:]
            lens[0] = len(chunk) // 2
            del chunk[len(chunk) // 2 :]
            lens.append(len(chunk))
            mins.append(chunk[0])
        else:
            return
        index = range(len(self._filenames))[index]
        if index == 0:
            if lens[0] + lens[1] < CHUNKSIZE:
                lens[0] += lens[1]
                self._len += lens[1]
                self._cache_chunk(0).extend(self._pop_chunk(1))
            elif lens[0] + lens[1] > 4 * CHUNKSIZE:
                chunk = [
                    *self._cache_chunk(0),
                    *self._cache_chunk(1),
                ]
                self._cache_chunk(0)[:] = chunk[: len(chunk) // 3]
                lens[0] = len(self._cache_chunk(0))
                self._cache_chunk(1)[:] = chunk[len(chunk) // 3 : 2 * len(chunk) // 3]
                lens[1] = len(self._cache_chunk(1))
                mins[1] = self._cache_chunk(1)[0]
                del chunk[: 2 * len(chunk) // 3]
                self._filenames.insert(2, self._get_filename())
                self._free_cache()
                self._cache[self._filenames[2]] = chunk
                lens.insert(2, len(chunk))
                mins.insert(2, chunk[0])
            elif (
                CHUNKSIZE // 2 < lens[0] < CHUNKSIZE * 2
                and CHUNKSIZE_EXTENDED < lens[0] + lens[1] < 3 * CHUNKSIZE
            ):
                pass
            elif lens[0] > lens[1]:
                diff = lens[0] - lens[1]
                self._cache_chunk(1)[:0] = self._cache_chunk(0)[-diff // 2 :]
                lens[1] = len(self._cache_chunk(1))
                mins[1] = self._cache_chunk(1)[0]
                del self._cache_chunk(0)[-diff // 2 :]
                lens[0] = len(self._cache_chunk(0))
            else:
                diff = lens[1] - lens[0]
                self._cache_chunk(0).extend(self._cache_chunk(1)[: diff // 2])
                lens[0] = len(self._cache_chunk(0))
                del self._cache_chunk(1)[: diff // 2]
                lens[1] = len(self._cache_chunk(1))
                mins[1] = self._cache_chunk(1)[0]
        elif index + 1 == len(self._filenames):
            if lens[-1] + lens[-2] < CHUNKSIZE:
                lens[-2] += lens[-1]
                self._cache_chunk(index - 1).extend(self._pop_chunk(index))
            elif lens[-1] + lens[-2] > 4 * CHUNKSIZE:
                chunk = [
                    *self._cache_chunk(-2),
                    *self._cache_chunk(-1),
                ]
                self._cache_chunk(-2)[:] = chunk[: len(chunk) // 3]
                lens[-2] = len(self._cache_chunk(-2))
                self._cache_chunk(-1)[:] = chunk[len(chunk) // 3 : 2 * len(chunk) // 3]
                lens[-1] = len(self._cache_chunk(-1))
                mins[-1] = self._cache_chunk(-1)[0]
                del chunk[: 2 * len(chunk) // 3]
                self._filenames.append(self._get_filename())
                self._free_cache()
                self._cache[self._filenames[-1]] = chunk
                lens.append(len(chunk))
                mins.append(chunk[0])
            elif (
                CHUNKSIZE // 2 < lens[-1] < CHUNKSIZE * 2
                and CHUNKSIZE_EXTENDED < lens[-1] + lens[-2] < 3 * CHUNKSIZE
            ):
                pass
            elif lens[-1] < lens[-2]:
                diff = lens[-2] - lens[-1]
                self._cache_chunk(-1)[:0] = self._cache_chunk(-2)[-diff // 2 :]
                lens[-1] = len(self._cache_chunk(-1))
                mins[-1] = self._cache_chunk(-1)[0]
                del self._cache_chunk(-2)[-diff // 2 :]
                lens[-2] = len(self._cache_chunk(-2))
            else:
                diff = lens[-1] - lens[-2]
                self._cache_chunk(-2).extend(self._cache_chunk(-1)[: diff // 2])
                lens[-2] = len(self._cache_chunk(-2))
                del self._cache_chunk(-1)[: diff // 2]
                lens[-1] = len(self._cache_chunk(-1))
                mins[-1] = self._cache_chunk(-1)[0]
        else:
            if lens[index - 1] + lens[index] + lens[index + 1] < CHUNKSIZE_EXTENDED:
                chunk = [
                    *self._cache_chunk(index - 1),
                    *self._cache_chunk(index),
                    *self._pop_chunk(index + 1),
                ]
                self._cache_chunk(index - 1)[:] = chunk[: len(chunk) // 2]
                self._cache_chunk(index)[:] = chunk[len(chunk) // 2 :]
                lens[index - 1] = len(chunk) // 2
                lens[index] = (len(chunk) + 1) // 2
                mins[index] = self._cache_chunk(index)[0]
            elif lens[index - 1] + lens[index] + lens[index + 1] > 6 * CHUNKSIZE:
                chunk = [
                    *self._cache_chunk(index - 1),
                    *self._cache_chunk(index),
                    *self._cache_chunk(index + 1),
                ]
                self._cache_chunk(index - 1)[:] = chunk[: len(chunk) // 4]
                lens[index - 1] = len(self._cache_chunk(index - 1))
                self._cache_chunk(index)[:] = chunk[len(chunk) // 4 : len(chunk) // 2]
                lens[index] = len(self._cache_chunk(index))
                mins[index] = self._cache_chunk(index)[0]
                self._cache_chunk(index + 1)[:] = chunk[len(chunk) // 2 : 3 * len(chunk) // 4]
                lens[index + 1] = len(self._cache_chunk(index + 1))
                mins[index + 1] = self._cache_chunk(index + 1)[0]
                del chunk[: 3 * len(chunk) // 4]
                self._filenames.insert(index + 2, self._get_filename())
                self._free_cache()
                self._cache[self._filenames[index + 2]] = chunk
                lens.insert(index + 2, len(chunk))
                mins.insert(index + 2, chunk[0])
            elif not all(CHUNKSIZE // 2 < 2 * L // 3 < CHUNKSIZE for L in lens[index - 1 : index + 2]):
                chunk = [
                    *self._cache_chunk(index - 1),
                    *self._cache_chunk(index),
                    *self._cache_chunk(index + 1),
                ]
                self._cache_chunk(index - 1)[:] = chunk[: len(chunk) // 3]
                lens[index - 1] = len(self._cache_chunk(index - 1))
                self._cache_chunk(index)[:] = chunk[len(chunk) // 3 : 2 * len(chunk) // 3]
                lens[index] = len(self._cache_chunk(index))
                mins[index] = self._cache_chunk(index)[0]
                self._cache_chunk(index + 1)[:] = chunk[2 * len(chunk) // 3 :]
                lens[index + 1] = len(self._cache_chunk(index + 1))
                mins[index + 1] = self._cache_chunk(index + 1)[0]

    def _cache_chunk(self: Self, index: int, /) -> list[T]:
        filename = self._filenames[index]
        if filename in self._cache:
            self._cache.move_to_end(filename)
        else:
            self._free_cache()
            with open(self._path / "sorted_set" / f"{filename}.txt", mode="rb") as file:
                self._cache[filename] = pickle.load(file)
        return self._cache[filename]

    def _commit_chunk(self: Self, filename: int, segment: list[T], /) -> None:
        with open(self._path / "sorted_set" / f"{filename}.txt", mode="wb") as file:
            pickle.dump(segment, file)

    def _del_chunk(self: Self, index: int, /) -> None:
        index = range(len(self._filenames))[index]
        filename = self._filenames.pop(index)
        (self._path / "sorted_set" / f"{filename}.txt").unlink()
        self._len -= self._lens.pop(index)
        del self._mins[index]

    def _free_cache(self: Self, /) -> None:
        while len(self._cache) >= 16:
            self._commit_chunk(*self._cache.popitem(last=False))

    def _get_filename(self: Self, /) -> int:
        path = self._path
        with open(path / "sorted_set" / "counter.txt", mode="rb") as file:
            uid = pickle.load(file)
        with open(path / "sorted_set" / "counter.txt", mode="wb") as file:
            pickle.dump(uid + 1, file)
        with open(path / "sorted_set" / f"{uid}.txt", mode="wb") as file:
            pickle.dump([], file)
        return uid

    def _pop_chunk(self: Self, index: int, /) -> list[T]:
        filename = self._filenames[index]
        if filename in self._cache:
            segment = self._cache.pop(filename)
        else:
            with open(self._path / "sorted_set" / f"{filename}.txt", mode="rb") as file:
                segment = pickle.load(file)
        self._del_chunk(index)
        return segment

    def add(self: Self, element: T, /) -> None:
        i = bisect(self._mins, element)
        if i > 0:
            j = bisect(self._cache_chunk(i - 1), element)
            if j == 0 or self._cache_chunk(i - 1)[j - 1] != element:
                self._cache_chunk(i - 1).insert(j, element)
                self._len += 1
                self._lens[i - 1] += 1
                self._balance(i - 1)
        elif self._len > 0:
            self._cache_chunk(0).insert(0, element)
            self._len += 1
            self._lens[0] += 1
            self._balance(0)
        else:
            self._filenames.append(self._get_filename())
            self._cache[self._filenames[-1]] = [element]
            self._len = 1
            self._lens.append(1)
            self._mins.append(element)

    def clear(self: Self, /) -> None:
        path = self._path
        for filename in self._filenames:
            (path / "sorted_set" / f"{filename}.txt").unlink()
        with open(path / "sorted_set" / "counter.txt", mode="wb") as file:
            pickle.dump(0, file)
        with open(path / "sorted_set" / "filenames.txt", mode="wb") as file:
            pickle.dump([], file)
        with open(path / "sorted_set" / "lens.txt", mode="wb") as file:
            pickle.dump([], file)
        with open(path / "sorted_set" / "mins.txt", mode="wb") as file:
            pickle.dump([], file)
        self._cache.clear()
        self._filenames.clear()
        self._len = 0
        self._lens.clear()
        self._mins.clear()

    def commit(self: Self, /) -> None:
        path = self._path
        with open(path / "sorted_set" / "filenames.txt", mode="wb") as file:
            pickle.dump(self._filenames, file)
        with open(path / "sorted_set" / "lens.txt", mode="wb") as file:
            pickle.dump(self._lens, file)
        with open(path / "sorted_set" / "mins.txt", mode="wb") as file:
            pickle.dump(self._mins, file)
        for filename, segment in self._cache.items():
            self._commit_chunk(filename, segment)

    def difference_update(self: Self, /, *iterables: Iterable[Any]) -> None:
        for iterable in iterables:
            for element in iterable:
                self.discard(element)

    def discard(self: Self, element: Any, /) -> None:
        i = bisect(self._mins, element)
        if i > 0:
            j = bisect(self._cache_chunk(i - 1), element)
            if j > 0 and self._cache_chunk(i - 1)[j - 1] == element:
                del self._cache_chunk(i - 1)[j]
                self._len -= 1
                self._lens[i - 1] -= 1
                self._balance(i - 1)

    def intersection_update(self: Self, /, *iterables: Iterable[Any]) -> None:
        path = self._path
        temp = path / "__temp__"
        for iterable in iterables:
            with BigSortedSet(temp) as db:
                db.update(element for element in iterable if element in self)
            for filename in ("counter.txt", "filenames.txt", "lens.txt", "mins.txt"):
                (temp / "sorted_set" / filename).rename(
                    path / "sorted_set" / filename
                )
            for filename in db._filenames:
                (temp / "sorted_set" / f"{filename}.txt").rename(
                    path / "sorted_set" / f"{filename}.txt"
                )
            self._filenames[:] = db._filenames[:]
            self._len = db._len
            self._lens[:] = db._lens[:]
            self._mins[:] = db._mins[:]

    def isdisjoint(self: Self, iterable: Iterable[Any], /) -> bool:
        if length_hint(iterable) > len(self) and isinstance(iterable, AbstractSet):
            return not any(element in iterable for element in self)
        else:
            iterator = iter(iterable)
            while True:
                chunk = sorted(islice(iterator, CHUNKSIZE_EXTENDED))
                if not chunk:
                    return True
                elif any(element in self for element in chunk):
                    return False

    def issubset(self: Self, iterable: Iterable[Any], /) -> bool:
        if isinstance(iterable, AbstractSet):
            return iterable >= self
        path = self._path
        temp = path / "__temp__"
        with BigSortedSet(temp) as db:
            db.update(element for element in iterable if element in self)
            result = len(db) == len(self)
            db.clear()
        return result

    def issuperset(self: Self, iterable: Iterable[Any], /) -> bool:
        iterator = iter(iterable)
        while True:
            chunk = sorted(islice(iterable, CHUNKSIZE_EXTENDED))
            if not chunk:
                return True
            elif not all(element in self for element in chunk):
                return False

    def pop(self: Self, /) -> T:
        if self._len == 0:
            raise KeyError
        else:
            self._len -= 1
            self._lens[-1] -= 1
            element = self._cache_chunk(-1).pop()
            self._balance(-1)
            return element

    def remove(self: Self, element: Any, /) -> None:
        len_ = self._len
        self.discard(element)
        if self._len == len_:
            raise KeyError(element)

    def symmetric_difference_update(self: Self, /, *iterables: Iterable[Any]) -> None:
        path = self._path
        temp = path / "__temp__"
        for iterable in iterables:
            with BigSortedSet(temp) as db:
                db.update(iterable)
                for _ in reversed(range(len(db._filenames))):
                    for element in db._pop_chunk(-1):
                        len_ = self._len
                        self.add(element)
                        if self._len == len_:
                            self.remove(element)
                db.clear()

    def update(self: Self, /, *iterables: Iterable[Any]) -> None:
        if self._len == 0:
            iterables = (*map(iter, iterables),)
            # Optimized for not `if not in self:` checks when initially empty.
            iterator = chain.from_iterable(iterables)
            front = (
                key
                for key, _ in groupby(sorted(islice(
                    iterator,
                    CHUNKSIZE_EXTENDED ** 2,
                )))
            )
            while True:
                chunk = [*islice(front, CHUNKSIZE_EXTENDED)]
                if not chunk:
                    return
                self._filenames.append(self._get_filename())
                self._commit_chunk(filenames[-1], chunk)
                self._len += len(chunk)
                self._lens.append(len(chunk))
                self._mins.append(chunk[0])
        elif self._len < CHUNKSIZE_EXTENDED ** 2:
            iterables = (*map(iter, iterables),)
            # Optimized for not `if not in self:` checks when initially empty.
            iterator = chain.from_iterable(iterables)
            front = [*islice(iterator, CHUNKSIZE_EXTENDED ** 2)]
            # Manually perform small insertions.
            if len(front) <= self._len // 8:
                # Sort to encourage cache efficiency.
                front.sort()
                for element, _ in groupby(front):
                    self.add(element)
                return
            # Large insertions are merged with the database and then
            # bulk sorted.
            front.extend(self)
            self.clear()
            front.sort()
            while True:
                chunk = [*islice(front, CHUNKSIZE_EXTENDED)]
                if not chunk:
                    return
                self._filenames.append(self._get_filename())
                self._commit_chunk(filenames[-1], chunk)
                self._len += len(chunk)
                self._lens.append(len(chunk))
                self._mins.append(chunk[0])
        # Chain iterables and collect only new items.
        iterator = chain.from_iterable(iterables)
        while True:
            has_elements = False
            # Fast cache-efficient insertion by sorting in chunks and
            # inserting nearby elements together.
            for key, _ in groupby(sorted(islice(
                iterator,
                CHUNKSIZE_EXTENDED ** 2,
            ))):
                has_elements = True
                self.add(element)
            if not has_elements:
                return
