#include <tc/ray_query.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

#include <tc/device.hpp>

namespace tc::detail {

namespace {

struct Item {
    vec3 low;
    vec3 high;
    vec3 centroid;
    std::uint32_t index = 0;
};

vec3 minimum(vec3 const &a, vec3 const &b) { return vec3{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
vec3 maximum(vec3 const &a, vec3 const &b) { return vec3{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }

float halfArea(vec3 const &low, vec3 const &high) {
    auto const d = high - low;
    return d.x < 0.0f ? 0.0f : d.x * d.y + d.y * d.z + d.z * d.x;
}

// A BVH over `items`, which it reorders so that every leaf is a range.
std::vector<BvhNode> buildNodes(std::vector<Item> &items, std::uint32_t const leafSize) {
    constexpr int kBins = 12;
    auto nodes = std::vector<BvhNode>{};
    nodes.reserve(2 * items.size() + 1);
    nodes.push_back(BvhNode{});
    struct Task {
        std::uint32_t node;
        std::uint32_t begin;
        std::uint32_t end;
    };
    auto tasks = std::vector<Task>{{0, 0, static_cast<std::uint32_t>(items.size())}};
    while (not tasks.empty()) {
        auto const task = tasks.back();
        tasks.pop_back();
        auto low = vec3{std::numeric_limits<float>::max()};
        auto high = vec3{-std::numeric_limits<float>::max()};
        auto centroidLow = low;
        auto centroidHigh = high;
        for (auto i = task.begin; i < task.end; ++i) {
            low = minimum(low, items[i].low);
            high = maximum(high, items[i].high);
            centroidLow = minimum(centroidLow, items[i].centroid);
            centroidHigh = maximum(centroidHigh, items[i].centroid);
        }
        auto &node = nodes[task.node];
        node.low = low;
        node.high = high;
        auto const count = task.end - task.begin;
        auto const extent = centroidHigh - centroidLow;
        auto const axis = extent.x >= extent.y and extent.x >= extent.z ? 0 : extent.y >= extent.z ? 1 : 2;
        if (count <= leafSize or extent[axis] <= 0.0f) {
            node.first = task.begin;
            node.count = count;
            continue;
        }
        struct Bin {
            vec3 low{std::numeric_limits<float>::max()};
            vec3 high{-std::numeric_limits<float>::max()};
            std::uint32_t count = 0;
        };
        auto bins = std::array<Bin, kBins>{};
        auto const binOf = [&](Item const &item) {
            auto const at = (item.centroid[axis] - centroidLow[axis]) / extent[axis] * kBins;
            return std::min(static_cast<int>(at), kBins - 1);
        };
        for (auto i = task.begin; i < task.end; ++i) {
            auto &bin = bins[static_cast<std::size_t>(binOf(items[i]))];
            bin.low = minimum(bin.low, items[i].low);
            bin.high = maximum(bin.high, items[i].high);
            ++bin.count;
        }
        auto bestCost = std::numeric_limits<float>::max();
        auto bestSplit = kBins / 2;
        for (int split = 1; split < kBins; ++split) {
            auto leftLow = vec3{std::numeric_limits<float>::max()};
            auto leftHigh = vec3{-std::numeric_limits<float>::max()};
            auto rightLow = leftLow;
            auto rightHigh = leftHigh;
            auto left = std::uint32_t{0};
            auto right = std::uint32_t{0};
            for (int b = 0; b < kBins; ++b) {
                auto const &bin = bins[static_cast<std::size_t>(b)];
                if (bin.count == 0) {
                    continue;
                }
                if (b < split) {
                    leftLow = minimum(leftLow, bin.low);
                    leftHigh = maximum(leftHigh, bin.high);
                    left += bin.count;
                } else {
                    rightLow = minimum(rightLow, bin.low);
                    rightHigh = maximum(rightHigh, bin.high);
                    right += bin.count;
                }
            }
            if (left == 0 or right == 0) {
                continue;
            }
            auto const cost = halfArea(leftLow, leftHigh) * static_cast<float>(left)
                            + halfArea(rightLow, rightHigh) * static_cast<float>(right);
            if (cost < bestCost) {
                bestCost = cost;
                bestSplit = split;
            }
        }
        auto const middle = std::partition(items.begin() + task.begin, items.begin() + task.end,
                                           [&](Item const &item) { return binOf(item) < bestSplit; });
        auto split = static_cast<std::uint32_t>(middle - items.begin());
        if (split == task.begin or split == task.end) {
            split = task.begin + count / 2;
            std::nth_element(items.begin() + task.begin, items.begin() + split, items.begin() + task.end,
                             [axis](Item const &a, Item const &b) { return a.centroid[axis] < b.centroid[axis]; });
        }
        auto const first = static_cast<std::uint32_t>(nodes.size());
        nodes[task.node].first = first;
        nodes[task.node].count = 0;
        nodes.push_back(BvhNode{});
        nodes.push_back(BvhNode{});
        tasks.push_back(Task{first, task.begin, split});
        tasks.push_back(Task{first + 1, split, task.end});
    }
    return nodes;
}

vec3 readPosition(HostGeometry const &geometry, std::uint32_t const vertex) {
    auto xyz = std::array<float, 3>{};
    std::memcpy(xyz.data(), geometry.positions.data() + static_cast<std::size_t>(vertex) * geometry.stride, sizeof xyz);
    return vec3{xyz[0], xyz[1], xyz[2]};
}

vec3 transformPoint(std::array<vec4, 3> const &rows, vec3 const &p) {
    return vec3{dot(rows[0].xyz(), p) + rows[0].w, dot(rows[1].xyz(), p) + rows[1].w, dot(rows[2].xyz(), p) + rows[2].w};
}

vec3 transformVector(std::array<vec4, 3> const &rows, vec3 const &v) {
    return vec3{dot(rows[0].xyz(), v), dot(rows[1].xyz(), v), dot(rows[2].xyz(), v)};
}

// The inverse of an affine 3x4 transform; false when it is singular.
bool invert(std::array<vec4, 3> const &rows, std::array<vec4, 3> &inverse) {
    double m[3][4];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            m[r][c] = rows[static_cast<std::size_t>(r)][c];
        }
    }
    auto const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (det == 0.0 or not std::isfinite(det)) {
        return false;
    }
    double a[3][3];
    a[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
    a[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    a[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
    a[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    a[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
    a[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    a[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
    a[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    a[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    for (int r = 0; r < 3; ++r) {
        auto const t = -(a[r][0] * m[0][3] + a[r][1] * m[1][3] + a[r][2] * m[2][3]);
        inverse[static_cast<std::size_t>(r)] =
            vec4{static_cast<float>(a[r][0]), static_cast<float>(a[r][1]), static_cast<float>(a[r][2]), static_cast<float>(t)};
    }
    return true;
}

// The parameter at which the ray enters the box within [tMin, tMax].
bool entersBox(BvhNode const &node, vec3 const &origin, vec3 const &direction, float const tMin, float const tMax,
               float &entry) {
    auto near = static_cast<double>(tMin);
    auto far = static_cast<double>(tMax);
    for (int axis = 0; axis < 3; ++axis) {
        auto const o = static_cast<double>(origin[axis]);
        auto const d = static_cast<double>(direction[axis]);
        auto const low = static_cast<double>(node.low[axis]);
        auto const high = static_cast<double>(node.high[axis]);
        if (d == 0.0) {
            if (o < low or o > high) {
                return false;
            }
            continue;
        }
        auto t0 = (low - o) / d;
        auto t1 = (high - o) / d;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        near = std::max(near, t0);
        far = std::min(far, t1);
        if (near > far) {
            return false;
        }
    }
    entry = static_cast<float>(near);
    return true;
}

struct Crossing {
    double t = 0.0;
    double u = 0.0;
    double v = 0.0;
    double det = 0.0;
};

// Möller-Trumbore in double precision: u and v weigh corners b and c.
bool crossesTriangle(HostTriangle const &triangle, vec3 const &origin, vec3 const &direction, Crossing &crossing) {
    auto const sub = [](vec3 const &a, vec3 const &b) {
        return std::array<double, 3>{double(a.x) - double(b.x), double(a.y) - double(b.y), double(a.z) - double(b.z)};
    };
    auto const cross = [](std::array<double, 3> const &a, std::array<double, 3> const &b) {
        return std::array<double, 3>{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    auto const dot3 = [](std::array<double, 3> const &a, std::array<double, 3> const &b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto const e1 = sub(triangle.b, triangle.a);
    auto const e2 = sub(triangle.c, triangle.a);
    auto const d = std::array<double, 3>{direction.x, direction.y, direction.z};
    auto const p = cross(d, e2);
    auto const det = dot3(e1, p);
    if (det == 0.0) {
        return false;
    }
    auto const s = sub(origin, triangle.a);
    auto const u = dot3(s, p) / det;
    if (u < 0.0 or u > 1.0) {
        return false;
    }
    auto const q = cross(s, e1);
    auto const v = dot3(d, q) / det;
    if (v < 0.0 or u + v > 1.0) {
        return false;
    }
    crossing = Crossing{dot3(e2, q) / det, u, v, det};
    return true;
}

// Nearest first; at equal distance nodes before hits, and hits in instance,
// geometry, primitive order, so ties resolve the same way every time.
bool farther(rayQueryEXT const &query, RayEntry const &a, RayEntry const &b) {
    if (a.t != b.t) {
        return a.t > b.t;
    }
    auto const aHit = a.kind == 2;
    auto const bHit = b.kind == 2;
    if (aHit != bHit) {
        return aHit;
    }
    if (not aHit) {
        return a.kind > b.kind;
    }
    auto const &x = query.hits[a.node];
    auto const &y = query.hits[b.node];
    return std::tie(x.instance, x.geometry, x.primitive) > std::tie(y.instance, y.geometry, y.primitive);
}

void push(rayQueryEXT &query, RayEntry const &entry) {
    query.heap.push_back(entry);
    std::push_heap(query.heap.begin(), query.heap.end(),
                   [&query](RayEntry const &a, RayEntry const &b) { return farther(query, a, b); });
}

RayEntry pop(rayQueryEXT &query) {
    std::pop_heap(query.heap.begin(), query.heap.end(),
                  [&query](RayEntry const &a, RayEntry const &b) { return farther(query, a, b); });
    auto const entry = query.heap.back();
    query.heap.pop_back();
    return entry;
}

float limitOf(rayQueryEXT const &query) { return query.hasCommitted ? query.committed.t : query.tMax; }

// The triangles of one bottom-level leaf that the ray crosses, as hits.
void collectHits(rayQueryEXT &query, HostInstance const &instance, std::uint32_t const instanceIndex,
                 BvhNode const &leaf, vec3 const &origin, vec3 const &direction) {
    auto const flags = query.flags;
    auto const cullFacing = (instance.flags & instanceTriangleFacingCullDisable) == 0u;
    for (auto i = leaf.first; i < leaf.first + leaf.count; ++i) {
        auto const &triangle = instance.blas->triangles[i];
        auto crossing = Crossing{};
        if (not crossesTriangle(triangle, origin, direction, crossing) or crossing.t < query.tMin
            or crossing.t > limitOf(query)) {
            continue;
        }
        // Front faces wind counter-clockwise as seen from the ray's origin, in
        // object space (what NVIDIA's and Mesa's Vulkan drivers report).
        auto front = crossing.det > 0.0;
        if ((instance.flags & instanceTriangleFlipFacing) != 0u) {
            front = not front;
        }
        if (cullFacing and (((flags & gl_RayFlagsCullBackFacingTrianglesEXT) != 0u and not front)
                            or ((flags & gl_RayFlagsCullFrontFacingTrianglesEXT) != 0u and front))) {
            continue;
        }
        auto opaque = triangle.opaque;
        if ((instance.flags & instanceForceOpaque) != 0u) {
            opaque = true;
        } else if ((instance.flags & instanceForceNoOpaque) != 0u) {
            opaque = false;
        }
        if ((flags & gl_RayFlagsOpaqueEXT) != 0u) {
            opaque = true;
        } else if ((flags & gl_RayFlagsNoOpaqueEXT) != 0u) {
            opaque = false;
        }
        if (((flags & gl_RayFlagsCullOpaqueEXT) != 0u and opaque)
            or ((flags & gl_RayFlagsCullNoOpaqueEXT) != 0u and not opaque)) {
            continue;
        }
        auto hit = RayHit{};
        hit.t = static_cast<float>(crossing.t);
        hit.barycentrics = vec2{static_cast<float>(crossing.u), static_cast<float>(crossing.v)};
        hit.frontFace = front;
        hit.opaque = opaque;
        hit.instance = instanceIndex;
        hit.primitive = triangle.primitive;
        hit.geometry = triangle.geometry;
        query.hits.push_back(hit);
        push(query, RayEntry{hit.t, 2, static_cast<std::uint32_t>(query.hits.size() - 1), instanceIndex});
    }
}

} // namespace

std::shared_ptr<BottomLevelHost const> buildBottomLevel(std::vector<HostGeometry> const &geometries) {
    auto blas = std::make_shared<BottomLevelHost>();
    for (std::uint32_t g = 0; g < geometries.size(); ++g) {
        auto const &geometry = geometries[g];
        for (std::uint32_t t = 0; t < geometry.triangleCount; ++t) {
            auto corners = std::array<std::uint32_t, 3>{};
            std::memcpy(corners.data(), geometry.indices.data() + static_cast<std::size_t>(t) * sizeof corners,
                        sizeof corners);
            for (auto const corner : corners) {
                if (corner >= geometry.vertexCount) {
                    throw DeviceError{"createBottomLevel: triangle " + std::to_string(t) + " of geometry "
                                      + std::to_string(g) + " uses vertex " + std::to_string(corner) + " of "
                                      + std::to_string(geometry.vertexCount)};
                }
            }
            blas->triangles.push_back(HostTriangle{readPosition(geometry, corners[0]), readPosition(geometry, corners[1]),
                                                   readPosition(geometry, corners[2]), t, g, geometry.opaque});
        }
    }
    auto items = std::vector<Item>{};
    items.reserve(blas->triangles.size());
    for (std::uint32_t i = 0; i < blas->triangles.size(); ++i) {
        auto const &triangle = blas->triangles[i];
        auto const low = minimum(minimum(triangle.a, triangle.b), triangle.c);
        auto const high = maximum(maximum(triangle.a, triangle.b), triangle.c);
        items.push_back(Item{low, high, (low + high) * 0.5f, i});
    }
    blas->nodes = buildNodes(items, 4);
    auto ordered = std::vector<HostTriangle>{};
    ordered.reserve(items.size());
    for (auto const &item : items) {
        ordered.push_back(blas->triangles[item.index]);
    }
    blas->triangles = std::move(ordered);
    return blas;
}

std::shared_ptr<TopLevelHost const> buildTopLevel(std::vector<HostInstance> instances) {
    auto tlas = std::make_shared<TopLevelHost>();
    auto items = std::vector<Item>{};
    auto kept = std::vector<HostInstance>{};
    for (auto &instance : instances) {
        if (not invert(instance.objectToWorld, instance.worldToObject)) {
            continue; // a singular transform: nothing to hit
        }
        auto const &root = instance.blas->nodes.front();
        auto low = vec3{std::numeric_limits<float>::max()};
        auto high = vec3{-std::numeric_limits<float>::max()};
        for (int corner = 0; corner < 8; ++corner) {
            auto const p = vec3{(corner & 1) != 0 ? root.high.x : root.low.x, (corner & 2) != 0 ? root.high.y : root.low.y,
                                (corner & 4) != 0 ? root.high.z : root.low.z};
            auto const world = transformPoint(instance.objectToWorld, p);
            low = minimum(low, world);
            high = maximum(high, world);
        }
        items.push_back(Item{low, high, (low + high) * 0.5f, static_cast<std::uint32_t>(kept.size())});
        kept.push_back(std::move(instance));
    }
    if (items.empty()) {
        return tlas;
    }
    tlas->nodes = buildNodes(items, 2);
    for (auto const &item : items) {
        tlas->instances.push_back(kept[item.index]);
    }
    return tlas;
}

void rayQueryStart(rayQueryEXT &query) {
    query.hasCommitted = false;
    query.terminated = false;
    query.committed = RayHit{};
    query.candidate = RayHit{};
    query.heap.clear();
    query.hits.clear();
    if (query.scene == nullptr or query.scene->nodes.empty() or not(query.tMin <= query.tMax)) {
        query.terminated = true;
        return;
    }
    auto entry = 0.0f;
    if (entersBox(query.scene->nodes.front(), query.origin, query.direction, query.tMin, query.tMax, entry)) {
        push(query, RayEntry{entry, 0, 0, 0});
    }
}

bool rayQueryStep(rayQueryEXT &query) {
    if (query.terminated) {
        return false;
    }
    auto const &scene = *query.scene;
    while (not query.heap.empty()) {
        auto const entry = pop(query);
        auto const limit = limitOf(query);
        if (entry.kind == 2 ? (query.hasCommitted ? entry.t >= limit : entry.t > limit) : entry.t > limit) {
            query.heap.clear(); // everything left is farther
            return false;
        }
        if (entry.kind == 2) {
            auto const &hit = query.hits[entry.node];
            if (not hit.opaque) {
                query.candidate = hit;
                return true;
            }
            query.committed = hit;
            query.hasCommitted = true;
            if ((query.flags & gl_RayFlagsTerminateOnFirstHitEXT) != 0u) {
                rayQueryTerminateEXT(query);
                return false;
            }
            continue;
        }
        if (entry.kind == 0) {
            auto const &node = scene.nodes[entry.node];
            if (node.count == 0) {
                for (auto const child : {node.first, node.first + 1}) {
                    auto t = 0.0f;
                    if (entersBox(scene.nodes[child], query.origin, query.direction, query.tMin, limit, t)) {
                        push(query, RayEntry{t, 0, child, 0});
                    }
                }
                continue;
            }
            for (auto i = node.first; i < node.first + node.count; ++i) {
                auto const &instance = scene.instances[i];
                if ((instance.mask & query.cullMask) == 0u) {
                    continue;
                }
                auto const origin = transformPoint(instance.worldToObject, query.origin);
                auto const direction = transformVector(instance.worldToObject, query.direction);
                auto t = 0.0f;
                if (entersBox(instance.blas->nodes.front(), origin, direction, query.tMin, limit, t)) {
                    push(query, RayEntry{t, 1, 0, i});
                }
            }
            continue;
        }
        auto const &instance = scene.instances[entry.instance];
        auto const &node = instance.blas->nodes[entry.node];
        auto const origin = transformPoint(instance.worldToObject, query.origin);
        auto const direction = transformVector(instance.worldToObject, query.direction);
        if (node.count > 0) {
            collectHits(query, instance, entry.instance, node, origin, direction);
            continue;
        }
        for (auto const child : {node.first, node.first + 1}) {
            auto t = 0.0f;
            if (entersBox(instance.blas->nodes[child], origin, direction, query.tMin, limit, t)) {
                push(query, RayEntry{t, 1, child, entry.instance});
            }
        }
    }
    return false;
}

HostInstance const &rayQueryInstance(rayQueryEXT const &query, bool const committed) {
    static auto const none = HostInstance{};
    auto const &hit = committed ? query.committed : query.candidate;
    if (query.scene == nullptr or hit.instance >= query.scene->instances.size()) {
        return none;
    }
    return query.scene->instances[hit.instance];
}

} // namespace tc::detail
