#include "glsl_names.hpp"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>

namespace tc::transpiler {

namespace {

llvm::StringSet<> const &reservedWords() {
    static auto const words = llvm::StringSet<>{
        // Keywords and reserved words of GLSL 4.60 and GLSL ES 3.20 that are
        // valid C++ identifiers.
        "attribute", "uniform", "varying", "buffer", "shared", "coherent", "restrict", "readonly", "writeonly",
        "layout", "centroid", "flat", "smooth", "noperspective", "patch", "sample", "invariant", "precise",
        "subroutine", "in", "out", "inout", "discard", "lowp", "mediump", "highp", "precision", "common",
        "partition", "active", "resource", "noinline", "external", "interface", "half", "fixed", "superp",
        "input", "output", "filter", "cast", "packed", "atomic_uint", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4",
        "uvec2", "uvec3", "uvec4", "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4", "hvec2", "hvec3", "hvec4",
        "fvec2", "fvec3", "fvec4", "mat2", "mat3", "mat4", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3",
        "mat3x4", "mat4x2", "mat4x3", "mat4x4", "dmat2", "dmat3", "dmat4", "uint", "image1D", "image2D",
        "image3D", "uimage2D", "iimage2D", "uimage3D", "iimage3D", "sampler2D", "sampler3D", "texture", "main",
        // Built-in functions; GLSL ES forbids redeclaring them.
        "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh",
        "acosh", "atanh", "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt", "abs", "sign", "floor",
        "trunc", "round", "roundEven", "ceil", "fract", "mod", "modf", "min", "max", "clamp", "mix", "step",
        "smoothstep", "isnan", "isinf", "floatBitsToInt", "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat",
        "fma", "frexp", "ldexp", "length", "distance", "dot", "cross", "normalize", "faceforward", "reflect",
        "refract", "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual", "any",
        "all", "not", "bitCount", "findLSB", "findMSB", "bitfieldExtract", "bitfieldInsert", "bitfieldReverse",
        "imageLoad", "imageStore", "imageSize", "barrier", "memoryBarrier", "groupMemoryBarrier", "atomicAdd",
        "atomicMin", "atomicMax", "atomicAnd", "atomicOr", "atomicXor", "atomicExchange", "atomicCompSwap",
        "packUnorm4x8", "unpackUnorm4x8", "packSnorm4x8", "unpackSnorm4x8", "packUnorm2x16", "unpackUnorm2x16",
        "packSnorm2x16", "unpackSnorm2x16", "packHalf2x16", "unpackHalf2x16", "texelFetch", "textureSize",
        "textureLod", "transpose", "determinant", "inverse", "matrixCompMult", "outerProduct", "sampler2DArray",
        "usampler2D", "isampler2D", "usampler3D", "isampler3D", "samplerExternalOES"};
    return words;
}

} // namespace

std::string glslIdentifier(llvm::StringRef const name) {
    auto result = name.str();
    for (auto pos = result.find("__"); pos != std::string::npos; pos = result.find("__", pos)) {
        result.replace(pos, 2, "_x");
    }
    if (llvm::StringRef{result}.starts_with("gl_")) {
        result = "tc_" + result;
    }
    if (reservedWords().contains(result)) {
        result += "_";
    }
    return result;
}

std::optional<std::string> glslNameOfStdFunction(llvm::StringRef const name) {
    static auto const names = [] {
        auto map = llvm::StringMap<std::string>{};
        for (auto const *same : {"sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "pow", "exp",
                                 "log", "exp2", "log2", "sqrt", "abs", "floor", "ceil", "trunc", "round", "min",
                                 "max", "clamp"}) {
            map[same] = same;
            map[std::string{same} + "f"] = same;
        }
        map["fabs"] = "abs";
        map["fabsf"] = "abs";
        map["fmin"] = "min";
        map["fminf"] = "min";
        map["fmax"] = "max";
        map["fmaxf"] = "max";
        map["atan2"] = "atan";
        map["atan2f"] = "atan";
        return map;
    }();
    auto const it = names.find(name);
    if (it == names.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string glslNameOfTcFunction(llvm::StringRef const name) {
    if (name == "not_") {
        return "not";
    }
    return name.str();
}

bool isSwizzleName(llvm::StringRef const name) {
    return name.size() >= 2 and name.size() <= 4 and name.find_first_not_of("xyzw") == llvm::StringRef::npos;
}

} // namespace tc::transpiler
