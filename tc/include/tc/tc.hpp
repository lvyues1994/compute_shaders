#pragma once

// Everything a kernel header needs. Host code additionally includes
// <tc/device.hpp> and a backend factory.
#include <tc/atomic.hpp>
#include <tc/buffer.hpp>
#include <tc/builtins.hpp>
#include <tc/dim.hpp>
#include <tc/external_image.hpp>
#include <tc/glsl_math.hpp>
#include <tc/glsl_packing.hpp>
#include <tc/gpu_format.hpp>
#include <tc/image.hpp>
#include <tc/kernel.hpp>
#include <tc/mat.hpp>
#include <tc/pixel.hpp>
#include <tc/shared.hpp>
#include <tc/texture.hpp>
#include <tc/uniform.hpp>
#include <tc/vec.hpp>
#include <tc/workgroup.hpp>
