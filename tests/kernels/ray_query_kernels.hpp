#pragma once

#include <tc/tc.hpp>

// Kernels for ray queries.
namespace tests {

using namespace tc;

// Ray i starts at rays[2i].xyz towards rays[2i + 1].xyz, up to t = rays[2i].w.
// Candidates that are not opaque are confirmed when their primitive index
// is even. hits[i] = (t, barycentrics, front face) of the committed hit, ids[i]
// = (custom index, instance, primitive, geometry); -1 and ~0u for misses.
struct TC_KERNEL TraceRays {
    static constexpr char fileLocation[] = "trace_rays";
    uvec3 local_size{64, 1, 1};

    AccelerationStructureBinding<0> scene;
    BufferBinding<vec4, 1> rays;
    BufferBinding<vec4, 2> hits;
    BufferBinding<uvec4, 3> ids;
    Uniform<uint, 0> flags{};
    Uniform<uint, 1> cullMask{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(hits.length())) {
            return;
        }
        vec4 origin = rays[2u * i];
        vec4 direction = rays[2u * i + 1u];
        rayQueryEXT query;
        rayQueryInitializeEXT(query, scene, flags.value, cullMask.value, origin.xyz(), 0.0f, direction.xyz(), origin.w);
        while (rayQueryProceedEXT(query)) {
            if ((uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)) & 1u) == 0u) {
                rayQueryConfirmIntersectionEXT(query);
            }
        }
        if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
            vec2 b = rayQueryGetIntersectionBarycentricsEXT(query, true);
            float front = rayQueryGetIntersectionFrontFaceEXT(query, true) ? 1.0f : 0.0f;
            hits[i] = vec4(rayQueryGetIntersectionTEXT(query, true), b.x, b.y, front);
            ids[i] = uvec4(uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)),
                           uint(rayQueryGetIntersectionInstanceIdEXT(query, true)),
                           uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)),
                           uint(rayQueryGetIntersectionGeometryIndexEXT(query, true)));
        } else {
            hits[i] = vec4(-1.0f);
            ids[i] = uvec4(0xFFFFFFFFu);
        }
    }
};

// Instance i of a top level: the bottom level `blas`, moved by offsets[i].xyz
// and scaled along x by offsets[i].w, custom index 100 + i, mask 1 << i.
struct TC_KERNEL WriteInstances {
    static constexpr char fileLocation[] = "write_instances";
    uvec3 local_size{64, 1, 1};

    BufferBinding<vec4, 0> offsets;
    BufferBinding<TopLevelInstance, 1> instances;
    Uniform<uvec2, 0> blas{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(offsets.length())) {
            return;
        }
        vec4 offset = offsets[i];
        TopLevelInstance instance;
        instance.transform[0] = vec4(offset.w, 0.0f, 0.0f, offset.x);
        instance.transform[1] = vec4(0.0f, 1.0f, 0.0f, offset.y);
        instance.transform[2] = vec4(0.0f, 0.0f, 1.0f, offset.z);
        instance.customIndexAndMask = (100u + i) | ((1u << i) << 24u);
        instance.offsetAndFlags = 0u;
        instance.reference = blas.value;
        instances[i] = instance;
    }
};

} // namespace tests
