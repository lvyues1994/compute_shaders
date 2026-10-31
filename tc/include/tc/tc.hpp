#pragma once

// Everything a kernel header needs. Host code additionally includes
// <tc/device.hpp> and a backend factory.
#include <tc/atomic.hpp>
#include <tc/buffer.hpp>
#include <tc/builtins.hpp>
#include <tc/codec.hpp>
#include <tc/compressed.hpp>
#include <tc/device_buffer.hpp>
#include <tc/device_image.hpp>
#include <tc/dim.hpp>
#include <tc/external_image.hpp>
#include <tc/external_memory.hpp>
#include <tc/glsl_math.hpp>
#include <tc/glsl_packing.hpp>
#include <tc/gpu_format.hpp>
#include <tc/half.hpp>
#include <tc/image.hpp>
#include <tc/kernel.hpp>
#include <tc/mat.hpp>
#include <tc/mip_chain.hpp>
#include <tc/pixel.hpp>
#include <tc/shared.hpp>
#include <tc/subgroup.hpp>
#include <tc/table.hpp>
#include <tc/texture.hpp>
#include <tc/uniform.hpp>
#include <tc/vec.hpp>
#include <tc/workgroup.hpp>
