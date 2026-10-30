#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include <memory>
#include <vector>

#include <tc/buffer.hpp>
#include <tc/codec.hpp>
#include <tc/detail/aggregate_fields.hpp>
#include <tc/mat.hpp>
#include <tc/vec.hpp>
#include <tc/workgroup.hpp>

namespace tc {

// A table layout decides where the fields of element `index` live in the
// word stream after the four-word header. It is plain constexpr C++: the
// host calls it, tc-transpile translates fieldAddress to GLSL and evaluates
// fieldAlignment at build time. Users can write their own:
//
//   struct MyLayout {
//       // Whether an element's fields are consecutive words (whole-element
//       // loads); false for column layouts.
//       static constexpr bool contiguousRecords = ...;
//       // Words used by `count` elements of `recordWords` words.
//       static constexpr std::size_t dataWords(std::size_t count, std::uint32_t recordWords);
//       // First word of the field at `fieldOffset` (`fieldWords` long).
//       static constexpr std::size_t fieldAddress(std::size_t index, std::uint32_t recordWords,
//                                                 std::uint32_t fieldOffset, std::uint32_t fieldWords);
//       // 1, 2 or 4: fieldAddress is a multiple of it for every index, so
//       // the GPU may move the field in uvec2 / uvec4 chunks.
//       static constexpr std::uint32_t fieldAlignment(std::uint32_t recordWords, std::uint32_t fieldOffset,
//                                                     std::uint32_t fieldWords);
//   };

namespace detail {

// The widest chunk, 4, 2 or 1 words, dividing `words`.
constexpr std::uint32_t chunkWords(std::uint32_t const words) noexcept {
    return words % 4 == 0 ? 4 : words % 2 == 0 ? 2 : 1;
}

} // namespace detail

// Each element's words one after another.
struct Aos {
    static constexpr bool contiguousRecords = true;

    static constexpr std::size_t dataWords(std::size_t const count, std::uint32_t const recordWords) noexcept {
        return count * recordWords;
    }
    static constexpr std::size_t fieldAddress(std::size_t const index, std::uint32_t const recordWords,
                                              std::uint32_t const fieldOffset, std::uint32_t /*fieldWords*/) noexcept {
        return index * recordWords + fieldOffset;
    }
    static constexpr std::uint32_t fieldAlignment(std::uint32_t const recordWords, std::uint32_t const fieldOffset,
                                                  std::uint32_t /*fieldWords*/) noexcept {
        auto const record = detail::chunkWords(recordWords);
        auto const field = detail::chunkWords(fieldOffset);
        return record < field ? record : field;
    }
};

// Elements in pages of PageSize; within a page every field is a column of
// PageSize consecutive values. Reading one field of many elements then
// touches only that field's memory, and a GPU wave reads it contiguously.
// Pages keep the columns of one element close together (AoSoA).
template <std::uint32_t PageSize = 256>
struct Soa {
    static_assert(PageSize >= 4 and (PageSize & (PageSize - 1)) == 0,
                  "the page size is a power of two of at least 4, so every column starts 16-byte aligned");
    static constexpr bool contiguousRecords = false;

    static constexpr std::size_t dataWords(std::size_t const count, std::uint32_t const recordWords) noexcept {
        return (count + PageSize - 1) / PageSize * PageSize * recordWords;
    }
    static constexpr std::size_t fieldAddress(std::size_t const index, std::uint32_t const recordWords,
                                              std::uint32_t const fieldOffset,
                                              std::uint32_t const fieldWords) noexcept {
        std::size_t const page = index / PageSize;
        std::size_t const slot = index % PageSize;
        return page * PageSize * recordWords + fieldOffset * PageSize + slot * fieldWords;
    }
    static constexpr std::uint32_t fieldAlignment(std::uint32_t /*recordWords*/, std::uint32_t /*fieldOffset*/,
                                                  std::uint32_t const fieldWords) noexcept {
        return detail::chunkWords(fieldWords);
    }
};

namespace detail {

template <class T>
struct IsStdArray : std::false_type {};
template <class T, std::size_t N>
struct IsStdArray<std::array<T, N>> : std::true_type {};

template <class T>
struct IsTcVec : std::false_type {};
template <class T, int N>
struct IsTcVec<vec<T, N>> : std::true_type {};

template <class T>
struct IsTcMat : std::false_type {};
template <class T, int C, int R>
struct IsTcMat<mat<T, C, R>> : std::true_type {};

// Converts to anything; only used in unevaluated aggregate initialisation.
struct AnyField {
    template <class U>
    operator U() const noexcept;
};

template <class T, class Indices, class = void>
struct BraceInitializable : std::false_type {};
template <class T, std::size_t... I>
struct BraceInitializable<T, std::index_sequence<I...>, std::void_t<decltype(T{((void)I, AnyField{})...})>>
    : std::true_type {};

// Number of fields of an aggregate: the most initializers T{...} accepts.
// C arrays would be counted per element (brace elision); use std::array.
template <class T, std::size_t N = 1>
constexpr std::size_t countFields() noexcept {
    if constexpr (N > kMaxAggregateFields or not BraceInitializable<T, std::make_index_sequence<N>>::value) {
        return N - 1;
    } else {
        return countFields<T, N + 1>();
    }
}

// A struct packed field by field, as opposed to a scalar, vector, matrix or
// std::array.
template <class T>
inline constexpr bool isRecordSchema = std::is_class_v<T> and std::is_aggregate_v<T> and not IsStdArray<T>::value
                                   and not IsTcVec<T>::value and not IsTcMat<T>::value and countFields<T>() > 0;

template <class T>
using FieldRefs = decltype(tieFields(std::declval<T &>(), FieldCountTag<countFields<T>()>{}));

template <class T, std::size_t I>
using FieldType = std::remove_reference_t<std::tuple_element_t<I, FieldRefs<T>>>;

template <class F, std::size_t... I>
constexpr void forEachIndex(F &&f, std::index_sequence<I...>) {
    (f(std::integral_constant<std::size_t, I>{}), ...);
}

// The packed representation of a value: a run of 32-bit words with no
// padding. tc-transpile generates the matching GLSL from the same rules.
template <class T, class = void>
struct Packing {
    static constexpr bool supported = false;
    static constexpr std::uint32_t words = 0;
};

template <class T>
struct Packing<T, std::enable_if_t<std::is_same_v<T, float> or std::is_same_v<T, std::int32_t>
                                   or std::is_same_v<T, std::uint32_t>>> {
    static constexpr bool supported = true;
    static constexpr std::uint32_t words = 1;
    static void store(T const &value, std::uint32_t *out) noexcept { std::memcpy(out, &value, sizeof value); }
    static void load(T &value, std::uint32_t const *in) noexcept { std::memcpy(&value, in, sizeof value); }
};

template <>
struct Packing<bool> {
    static constexpr bool supported = true;
    static constexpr std::uint32_t words = 1;
    static void store(bool const &value, std::uint32_t *out) noexcept { *out = value ? 1u : 0u; }
    static void load(bool &value, std::uint32_t const *in) noexcept { value = *in != 0u; }
};

// Low word first, as GLSL unpackDouble2x32 / packDouble2x32 order them.
template <>
struct Packing<double> {
    static constexpr bool supported = true;
    static constexpr std::uint32_t words = 2;
    static void store(double const &value, std::uint32_t *out) noexcept {
        auto bits = std::uint64_t{0};
        std::memcpy(&bits, &value, sizeof bits);
        out[0] = static_cast<std::uint32_t>(bits);
        out[1] = static_cast<std::uint32_t>(bits >> 32);
    }
    static void load(double &value, std::uint32_t const *in) noexcept {
        auto const bits = std::uint64_t{in[0]} | (std::uint64_t{in[1]} << 32);
        std::memcpy(&value, &bits, sizeof bits);
    }
};

template <class T>
struct Packing<T, std::enable_if_t<std::is_enum_v<T> and sizeof(T) == 4>> {
    using Underlying = std::underlying_type_t<T>;
    static constexpr bool supported = true;
    static constexpr std::uint32_t words = 1;
    static void store(T const &value, std::uint32_t *out) noexcept {
        auto const raw = static_cast<Underlying>(value);
        std::memcpy(out, &raw, sizeof raw);
    }
    static void load(T &value, std::uint32_t const *in) noexcept {
        auto raw = Underlying{};
        std::memcpy(&raw, in, sizeof raw);
        value = static_cast<T>(raw);
    }
};

template <class T, int N>
struct Packing<vec<T, N>> {
    using Element = Packing<T>;
    static constexpr bool supported = Element::supported;
    static constexpr std::uint32_t words = N * Element::words;
    static void store(vec<T, N> const &value, std::uint32_t *out) noexcept {
        for (int i = 0; i < N; ++i) {
            Element::store(value[i], out + i * Element::words);
        }
    }
    static void load(vec<T, N> &value, std::uint32_t const *in) noexcept {
        for (int i = 0; i < N; ++i) {
            Element::load(value[i], in + i * Element::words);
        }
    }
};

// Column-major, without the padding std430 puts after 3-row columns.
template <class T, int C, int R>
struct Packing<mat<T, C, R>> {
    using Element = Packing<T>;
    static constexpr bool supported = Element::supported;
    static constexpr std::uint32_t words = C * R * Element::words;
    static void store(mat<T, C, R> const &value, std::uint32_t *out) noexcept {
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                Element::store(value[c][r], out + (c * R + r) * Element::words);
            }
        }
    }
    static void load(mat<T, C, R> &value, std::uint32_t const *in) noexcept {
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                Element::load(value[c][r], in + (c * R + r) * Element::words);
            }
        }
    }
};

template <class T, std::size_t N>
struct Packing<std::array<T, N>> {
    using Element = Packing<T>;
    static constexpr bool supported = Element::supported;
    static constexpr std::uint32_t words = static_cast<std::uint32_t>(N) * Element::words;
    static void store(std::array<T, N> const &value, std::uint32_t *out) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            Element::store(value[i], out + i * Element::words);
        }
    }
    static void load(std::array<T, N> &value, std::uint32_t const *in) noexcept {
        for (std::size_t i = 0; i < N; ++i) {
            Element::load(value[i], in + i * Element::words);
        }
    }
};

// The fields of a table element: those of a record, or the element itself.
template <class T, class = void>
struct TableFields {
    static constexpr std::size_t count = 1;
    template <std::size_t I>
    using Type = T;
    template <std::size_t I>
    static constexpr std::uint32_t offset = 0;
    template <class V>
    static auto refs(V &value) noexcept {
        return std::tie(value);
    }
};

template <class T>
struct TableFields<T, std::enable_if_t<isRecordSchema<T>>> {
    static constexpr std::size_t count = countFields<T>();
    template <std::size_t I>
    using Type = FieldType<T, I>;

private:
    template <std::size_t... J>
    static constexpr std::uint32_t wordsBefore(std::index_sequence<J...>) noexcept {
        return (Packing<FieldType<T, J>>::words + ... + 0u);
    }

public:
    // First word of field I within the packed record.
    template <std::size_t I>
    static constexpr std::uint32_t offset = wordsBefore(std::make_index_sequence<I>{});
    template <class V>
    static auto refs(V &value) noexcept {
        return tieFields(value, FieldCountTag<count>{});
    }
};

template <class T>
struct Packing<T, std::enable_if_t<isRecordSchema<T>>> {
    using Fields = TableFields<T>;

private:
    template <std::size_t... I>
    static constexpr bool allSupported(std::index_sequence<I...>) noexcept {
        return (Packing<FieldType<T, I>>::supported and ...);
    }

public:
    static constexpr bool supported = allSupported(std::make_index_sequence<Fields::count>{});
    static constexpr std::uint32_t words =
        Fields::template offset<Fields::count - 1> + Packing<FieldType<T, Fields::count - 1>>::words;

    static void store(T const &value, std::uint32_t *out) noexcept {
        auto const fields = Fields::refs(value);
        forEachIndex(
            [&](auto i) {
                constexpr auto I = decltype(i)::value;
                Packing<FieldType<T, I>>::store(std::get<I>(fields), out + Fields::template offset<I>);
            },
            std::make_index_sequence<Fields::count>{});
    }
    static void load(T &value, std::uint32_t const *in) noexcept {
        auto const fields = Fields::refs(value);
        forEachIndex(
            [&](auto i) {
                constexpr auto I = decltype(i)::value;
                Packing<FieldType<T, I>>::load(std::get<I>(fields), in + Fields::template offset<I>);
            },
            std::make_index_sequence<Fields::count>{});
    }
};

template <class V, class Codec>
struct Packing<Encoded<V, Codec>> {
    static constexpr bool supported = true;
    static constexpr std::uint32_t words = Codec::words;
    static void store(Encoded<V, Codec> const &value, std::uint32_t *out) noexcept {
        auto const packed = Codec::encode(value.value);
        std::copy(packed.begin(), packed.end(), out);
    }
    static void load(Encoded<V, Codec> &value, std::uint32_t const *in) noexcept {
        auto packed = std::array<uint, Codec::words>{};
        std::copy(in, in + Codec::words, packed.begin());
        value.value = Codec::decode(packed);
    }
};

// Whether packing loses precision (the value contains Encoded fields).
template <class T, class = void>
struct IsLossy : std::false_type {};
template <class V, class Codec>
struct IsLossy<Encoded<V, Codec>> : std::true_type {};
template <class T, std::size_t N>
struct IsLossy<std::array<T, N>> : IsLossy<T> {};
template <class T>
struct IsLossy<T, std::enable_if_t<isRecordSchema<T>>> {
private:
    template <std::size_t... I>
    static constexpr bool anyField(std::index_sequence<I...>) noexcept {
        return (IsLossy<FieldType<T, I>>::value or ...);
    }

public:
    static constexpr bool value = anyField(std::make_index_sequence<countFields<T>()>{});
};

template <class T>
T unpack(std::uint32_t const *words) noexcept {
    auto value = T{};
    Packing<T>::load(value, words);
    return value;
}

} // namespace detail

// Elements of type T packed into 32-bit words without padding, so the CPU
// and every GPU API agree on the layout whatever the alignment rules. Word 0
// of the header holds the element count (GLSL `t.length()`). Layout is Aos
// or Soa<PageSize>; switching it changes no kernel code.
template <class T, class Layout = Aos>
struct TableResource {
    static_assert(std::is_class_v<T>, "table elements are structs, vectors or matrices; store scalars in a "
                                      "BufferResource, which is already compact");
    static_assert(detail::Packing<T>::supported,
                  "table elements may hold bool, int32_t, uint32_t, float, double, 32-bit enums, tc vectors and "
                  "matrices, std::array and structs of these (no C arrays, pointers or base classes)");

    using value_type = T;
    using layout_type = Layout;
    using Fields = detail::TableFields<T>;
    using Record = std::array<std::uint32_t, detail::Packing<T>::words>;
    static constexpr std::uint32_t recordWords = detail::Packing<T>::words;
    static constexpr std::uint32_t headerWords = 4;
    // These also make the compiler instantiate the layout members that
    // tc-transpile translates and evaluates.
    static_assert(Layout::contiguousRecords or not Layout::contiguousRecords);
    static_assert(Layout::fieldAddress(0, recordWords, 0, recordWords) == 0, "a layout starts at word 0");
    static_assert(Layout::fieldAlignment(recordWords, 0, recordWords) == 1
                      or Layout::fieldAlignment(recordWords, 0, recordWords) == 2
                      or Layout::fieldAlignment(recordWords, 0, recordWords) == 4,
                  "a layout's fieldAlignment is 1, 2 or 4 words");

    static std::size_t wordsFor(std::size_t const count) noexcept {
        return headerWords + Layout::dataWords(count, recordWords);
    }

    explicit TableResource(std::size_t const count)
        : TableResource(BufferResource<std::uint32_t>{static_cast<std::int32_t>(wordsFor(count))}, count) {}

    // Adopts word storage of at least wordsFor(count) words, such as a
    // buffer from Device::createBuffer.
    TableResource(BufferResource<std::uint32_t> wordStorage, std::size_t const count)
        : storage(std::move(wordStorage)), elementCount(count) {
        assert(storage.size() >= wordsFor(count));
        storage[0] = static_cast<std::uint32_t>(count);
    }

    std::size_t size() const noexcept { return elementCount; }

    T get(std::size_t const index) const noexcept {
        auto record = packedRecord(index);
        return detail::unpack<T>(record.data());
    }

    void set(std::size_t const index, T const &value) noexcept {
        auto record = Record{};
        detail::Packing<T>::store(value, record.data());
        writeRecord(index, record);
    }

    // The word stream the devices upload, bind and download.
    BufferResource<std::uint32_t> &words() noexcept { return storage; }
    BufferResource<std::uint32_t> const &words() const noexcept { return storage; }
    ResourceIdentity const &identity() const noexcept { return storage.identity(); }
    bool usesExternalMemory() const noexcept { return storage.usesExternalMemory(); }

    // Element `index` in packed record order, whatever the layout.
    Record packedRecord(std::size_t const index) const noexcept {
        assert(index < elementCount);
        auto record = Record{};
        detail::forEachIndex(
            [&](auto i) {
                constexpr auto I = decltype(i)::value;
                auto const *source = fieldWords<I>(index);
                std::copy(source, source + fieldSize<I>, record.data() + Fields::template offset<I>);
            },
            std::make_index_sequence<Fields::count>{});
        return record;
    }

    // Writes the words of `after` that differ from `before`, so concurrent
    // CPU invocations updating different fields of one element do not race.
    void storeChangedWords(std::size_t const index, Record const &before, Record const &after) noexcept {
        assert(index < elementCount);
        detail::forEachIndex(
            [&](auto i) {
                constexpr auto I = decltype(i)::value;
                auto *target = fieldWords<I>(index);
                for (std::uint32_t w = 0; w < fieldSize<I>; ++w) {
                    auto const at = Fields::template offset<I> + w;
                    if (before[at] != after[at]) {
                        target[w] = after[at];
                    }
                }
            },
            std::make_index_sequence<Fields::count>{});
    }

private:
    template <std::size_t I>
    static constexpr std::uint32_t fieldSize = detail::Packing<typename Fields::template Type<I>>::words;

    template <std::size_t I>
    std::size_t fieldAddress(std::size_t const index) const noexcept {
        return headerWords + Layout::fieldAddress(index, recordWords, Fields::template offset<I>, fieldSize<I>);
    }
    template <std::size_t I>
    std::uint32_t const *fieldWords(std::size_t const index) const noexcept {
        return storage.data() + fieldAddress<I>(index);
    }
    template <std::size_t I>
    std::uint32_t *fieldWords(std::size_t const index) noexcept {
        return storage.data() + fieldAddress<I>(index);
    }

    void writeRecord(std::size_t const index, Record const &record) noexcept {
        assert(index < elementCount);
        detail::forEachIndex(
            [&](auto i) {
                constexpr auto I = decltype(i)::value;
                auto const *source = record.data() + Fields::template offset<I>;
                std::copy(source, source + fieldSize<I>, fieldWords<I>(index));
            },
            std::make_index_sequence<Fields::count>{});
    }

    BufferResource<std::uint32_t> storage;
    std::size_t elementCount = 0;
};

namespace detail {

// Unpacked copies of the table elements the CPU invocations on this thread
// touched. The invocations of a cooperative workgroup run on one thread and
// share them; flushTables() writes back the words that changed, so
// workgroups updating different fields of one element do not race.
template <class T, class Layout>
struct TableCache {
    struct Entry {
        TableResource<T, Layout> *table = nullptr;
        std::size_t index = 0;
        T value{};
        typename TableResource<T, Layout>::Record loaded{};
    };

    T &element(TableResource<T, Layout> &table, std::size_t const index) {
        for (auto e = used; e-- > 0;) {
            auto &entry = *entries[e];
            if (entry.index == index and entry.table == &table) {
                if constexpr (IsLossy<T>::value) {
                    // As on the GPU, a value stored earlier reads back through its codec.
                    auto record = typename TableResource<T, Layout>::Record{};
                    Packing<T>::store(entry.value, record.data());
                    Packing<T>::load(entry.value, record.data());
                }
                return entry.value;
            }
        }
        if (used == entries.size()) {
            entries.push_back(std::make_unique<Entry>()); // stable address: callers hold references
        }
        auto &entry = *entries[used++];
        entry.table = &table;
        entry.index = index;
        entry.loaded = table.packedRecord(index);
        entry.value = unpack<T>(entry.loaded.data());
        if (not registered) {
            registered = true;
            registerTableFlush(TableFlush{this, &TableCache::writeBack});
        }
        return entry.value;
    }

    static void writeBack(void *const self, bool const keepEntries) noexcept {
        auto &cache = *static_cast<TableCache *>(self);
        for (std::size_t e = 0; e < cache.used; ++e) {
            auto &entry = *cache.entries[e];
            auto current = typename TableResource<T, Layout>::Record{};
            Packing<T>::store(entry.value, current.data());
            entry.table->storeChangedWords(entry.index, entry.loaded, current);
            entry.loaded = current;
        }
        if (not keepEntries) {
            cache.used = 0;
            cache.registered = false;
        }
    }

    static TableCache &forThisThread() {
        thread_local TableCache cache;
        return cache;
    }

private:
    std::vector<std::unique_ptr<Entry>> entries;
    std::size_t used = 0;
    bool registered = false;
};

} // namespace detail

// A table in a shader storage buffer of words. In a kernel, `t[i]` is
// element i; reading `t[i].field` loads only that field on the GPU, and
// statements such as `t[i] = value;`, `t[i].field = value;` or
// `t[i].field.x += 1.0f;` store it. tc-transpile generates the GLSL that
// packs and unpacks the words. On the CPU, `t[i]` is a cached copy that is
// written back when the invocation ends.
template <class T, std::uint32_t Binding, class Layout = Aos>
struct TableBinding {
    using value_type = T;
    using Resource = TableResource<T, Layout>;
    static constexpr std::uint32_t binding = Binding;

    void attach(Resource &resource_) noexcept { resource = &resource_; }
    Resource *attachedResource() const noexcept { return resource; }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    T &operator[](I const index) const {
        assert(resource != nullptr and "TableBinding used before attach()");
        assert(static_cast<std::size_t>(index) < resource->size());
        return detail::TableCache<T, Layout>::forThisThread().element(*resource, static_cast<std::size_t>(index));
    }

    // Element count, as an int like GLSL `length()`.
    std::int32_t length() const noexcept {
        assert(resource != nullptr and "TableBinding used before attach()");
        return static_cast<std::int32_t>(resource->size());
    }

private:
    Resource *resource = nullptr;
};

} // namespace tc
