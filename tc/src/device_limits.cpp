#include <tc/device.hpp>

namespace tc {

namespace {

std::string over(KernelProgram const &program, std::string const &what, std::uint64_t const used,
                 std::uint64_t const allowed) {
    return "kernel '" + std::string{program.name} + "' uses " + std::to_string(used) + " " + what
         + "; this device allows " + std::to_string(allowed);
}

char const *featureName(KernelFeature const feature) {
    switch (feature) {
    case KernelFeature::Double:
        return "double precision";
    case KernelFeature::Subgroups:
        return "subgroup operations";
    case KernelFeature::Int64:
        return "64-bit integers";
    case KernelFeature::Int64Atomics:
        return "64-bit atomics";
    }
    return "an optional feature";
}

} // namespace

std::optional<std::string> exceedsLimits(KernelProgram const &program, DeviceLimits const &limits) {
    auto buffers = std::uint32_t{0};
    auto images = std::uint32_t{0};
    auto textures = std::uint32_t{0};
    for (auto const &slot : program.resources) {
        switch (slot.kind) {
        case ResourceKind::StorageBuffer:
            ++buffers;
            break;
        case ResourceKind::StorageImage:
        case ResourceKind::ExternalImage:
            ++images;
            break;
        case ResourceKind::SampledTexture:
        case ResourceKind::ExternalTexture:
            ++textures;
            break;
        }
    }
    if (buffers > limits.storageBuffers) {
        return over(program, "storage buffers", buffers, limits.storageBuffers);
    }
    if (images > limits.storageImages) {
        return over(program, "storage images", images, limits.storageImages);
    }
    if (textures > limits.sampledTextures) {
        return over(program, "sampled textures", textures, limits.sampledTextures);
    }
    if (program.sharedBytes > limits.sharedBytes) {
        return over(program, "bytes of shared memory", program.sharedBytes, limits.sharedBytes);
    }
    auto const size = program.localSize;
    auto const invocations = std::uint64_t{size.x} * size.y * size.z;
    if (invocations > limits.workGroupInvocations) {
        return over(program, "invocations per workgroup", invocations, limits.workGroupInvocations);
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (size[axis] > limits.workGroupSize[axis]) {
            return over(program, std::string{"invocations along workgroup axis "} + "xyz"[axis], size[axis],
                        limits.workGroupSize[axis]);
        }
    }
    if (limits.pushConstantBytes and program.pushConstantSize > *limits.pushConstantBytes) {
        return over(program, "bytes of uniforms (push constants)", program.pushConstantSize,
                    *limits.pushConstantBytes);
    }
    for (auto const feature : {KernelFeature::Double, KernelFeature::Subgroups, KernelFeature::Int64,
                               KernelFeature::Int64Atomics}) {
        if (uses(program.features, feature) and not uses(limits.features, feature)) {
            return "kernel '" + std::string{program.name} + "' uses " + featureName(feature)
                 + ", which this device does not support";
        }
    }
    return std::nullopt;
}

} // namespace tc
