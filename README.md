# compute_shaders

用纯 C++17 写 compute kernel，同一份代码既能在 CPU 上直接运行和调试，也能被转译成 GLSL / SPIR-V，在 OpenGL、OpenGL ES（Android）和 Vulkan 上运行。思路来自 CppCon 2025 的演讲 *From Pure ISO C++20 to Compute Shaders*（Koen Samyn），这里按 C++17 重新实现，并扩展到多个 GPU 后端。

```cpp
struct TC_KERNEL FloatAdder {                       // [[clang::annotate("kernel")]]
    static constexpr char fileLocation[] = "float_adder";
    uvec3 local_size{256, 1, 1};

    BufferBinding<float, 0> A;
    BufferBinding<float, 1> B;
    BufferBinding<float, 2> C;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(C.length())) return;
        C[i] = A[i] + B[i];
    }
};
```

```cpp
auto device = tc::createDevice({"vulkan"});        // 或 "cpu" / "opengl" / "gles"
device->uploadBuffer(a);  device->uploadBuffer(b);  device->uploadBuffer(c);
device->useKernel(kernel);
device->bindBuffer(kernel.A);  device->bindBuffer(kernel.B);  device->bindBuffer(kernel.C);
device->execute(kernel, tc::uvec3{n / 256, 1, 1}); // 工作组数量
device->downloadBuffer(c);
```

## 后端

| 后端 | 名字 | 平台 | 运行的代码 |
|---|---|---|---|
| CPU | `cpu` | 所有平台 | kernel struct 本身，工作组分配到多线程 |
| OpenGL 4.3 core | `opengl` | 桌面 Linux（EGL，无窗口） | `#version 430` GLSL |
| OpenGL ES 3.1 | `gles` | Android、桌面（EGL） | `#version 310 es` GLSL |
| Vulkan 1.0 | `vulkan` | Linux、Android；Windows 与 macOS（MoltenVK）未验证 | SPIR-V |

在 Linux（RTX 5070 Ti + Mesa llvmpipe/lavapipe）上，所有示例和等价性测试在 CPU、GL、GLES、Vulkan 上结果一致。`--device llvmpipe` 可以选软件实现。

## 架构

```
kernel 头文件 (C++17)
   ├─ 普通 C++ 编译 ──────────────► CPU 后端直接执行 main()
   └─ tc-transpile (Clang LibTooling，构建时运行)
        ├─ AST matcher 找 [[clang::annotate("kernel")]] struct
        ├─ 校验：new/delete、指针、异常、递归、修改常量成员……按编译器格式报错
        ├─ AST → GLSL：所有 C++ 隐式转换写成显式构造（GLSL ES 没有隐式转换）
        ├─ 推断每个 binding 的 readonly / writeonly
        ├─ std430 布局检查：buffer 里的结构体在 C++ 和 GLSL 里字节布局必须一致
        ├─ 生成三种方言：GL 430 / ES 310 / Vulkan 450（uniform 变为 push constant）
        ├─ glslangValidator 校验，并把 Vulkan 版编译成 SPIR-V
        └─ 生成 <header>.tc.cpp：内嵌 GLSL + SPIR-V + 反射数据，静态注册到 ProgramRegistry
```

| 目录 | 内容 |
|---|---|
| `tc/` | 头文件 DSL：`vec`/swizzle、GLSL 内建函数、`BufferResource`/`BufferBinding`、像素与图像格式、`ImageBinding`、`Uniform`、`Device` 抽象、`ProgramRegistry` |
| `backends/cpu` `opengl` `vulkan` | 各后端，各自一个工厂函数 `tc::<backend>::createDevice(config)` |
| `backends/devices` | 按名字创建后端：`tc::createDevice({"vulkan", "llvmpipe"})` |
| `transpiler/` | `tc-transpile` 主机工具 |
| `cmake/TcKernels.cmake` | `tc_add_kernels(<target> HEADERS ...)` |
| `examples/` | `float_adder`、`game_of_life`、`raytracer`（对应演讲的三个案例） |
| `tests/` | DSL 单元测试、跨后端等价性测试、转译器报错测试 |

生成的 GLSL 在 `build/<preset>/.../tc_generated/<target>/glsl/*.comp`，可以直接查看。

## 写 kernel

kernel 是带 `TC_KERNEL` 标注的 struct，必须有 `fileLocation`、`local_size` 和 `void main()`。可用的成员：

- `BufferBinding<T, binding>`：SSBO，`A[i]` 读写，`A.length()`。`T` 可以是标量、`vec`、或只含这些成员的结构体（布局由转译器检查）。
- `ImageBinding<GpuFormat, Dim, CpuPixel, binding>`：存储图像，配合 `imageLoad` / `imageStore` / `imageSize`。GPU 内部格式和主机像素格式可以不同（例如主机用 `cpu::R8UI`，GPU 用 `gpu::R32UI`），上传/下载时自动转换；CPU 后端读写同样经过 GPU 格式，结果与 GPU 一致。
- `Uniform<T, location>`：主机每次 dispatch 前用 `setUniform` 设置。
- 其他数据成员和 `static constexpr` 成员：变成 GLSL 常量，初始化式必须是常量表达式。

`main()` 和它调用到的成员函数、自由函数、普通结构体（含成员函数）都会被转译。支持：标量/向量运算、`if`/`for`/`while`/`do`/`switch`、`std::array`（含 range-for）、`<cmath>` 常用函数、GLSL 内建函数（`tc::` 下同名实现）。

因为 C++17 不允许字符串字面量作模板参数，swizzle 写成成员函数：`v.xy()`、`v.zyx()`（转译成 `v.xy`、`v.zyx`），单分量直接用 `v.x`。

## 构建（Linux）

依赖（Ubuntu 24.04）：

```sh
sudo apt install cmake ninja-build g++ libclang-18-dev glslang-tools \
                 libvulkan-dev mesa-vulkan-drivers libegl-dev libgles-dev libgtest-dev
```

```sh
cmake --preset linux-debug && cmake --build --preset linux-debug && ctest --preset linux-debug
./build/linux-debug/examples/float_adder  --backend vulkan
./build/linux-debug/examples/game_of_life --backend gles
./build/linux-debug/examples/raytracer    --backend opengl   # 写出 raytracer_opengl.ppm
```

## Android

转译器是主机工具，先构建 Linux release（生成 `build/linux-release/transpiler/tc-transpile`），再用 NDK 交叉编译：

```sh
export ANDROID_NDK_HOME=~/Android/Sdk/ndk/28.2.13676358
cmake --preset linux-release && cmake --build --preset linux-release
cmake --preset android-arm64 && cmake --build --preset android-arm64
```

示例是独立的命令行程序（静态 libc++），可以用 adb 在设备上直接运行，GLES 与 Vulkan 都通过无窗口方式创建：

```sh
adb shell mkdir -p /data/local/tmp/tc_compute
adb push build/android-arm64/examples/game_of_life /data/local/tmp/tc_compute/
adb shell /data/local/tmp/tc_compute/game_of_life --backend vulkan
adb shell /data/local/tmp/tc_compute/game_of_life --backend gles
adb shell rm -rf /data/local/tmp/tc_compute
```

在 App 里使用时，把 `tc`、`tc_cpu`、`tc_opengl`、`tc_vulkan` 作为子项目加入 Android Studio 的 CMake 工程，`tc_add_kernels` 通过 `TC_TRANSPILER_EXECUTABLE` 使用主机上的转译器。

为了让 kernel 在所有设备上都能运行，图像请使用 GLES 3.1 和 Vulkan 都强制支持的存储格式：`rgba8`、`rgba8ui`、`rgba8i`、`r32ui`、`r32i`、`r32f`、`rgba32ui`、`rgba32i`、`rgba32f`。使用了 `double` 或其他格式的 kernel 仍可在桌面 GL / Vulkan 上运行，`gles` 后端会给出原因并拒绝执行。

## 与演讲的差异

- C++17：concept 改成 trait + `static_assert`，swizzle 用 `v.xy()`，没有指定初始化器。
- 转译用 AST 打印而不是 Rewriter：每个隐式转换都显式输出，结果对 GLSL ES 也合法；常量初始化交给 Clang 的常量求值器。
- 一次转译产出三种方言和 SPIR-V；Vulkan 的 uniform 自动放进 push constant 块，偏移由转译器计算并写入反射数据。
- 增加了 std430 布局检查和 binding 读写推断。

## 已知限制

- 没有 `shared` 内存和 `barrier()`（CPU 后端按工作组顺序执行 invocation），没有原子操作和矩阵类型。
- swizzle 只读；写分量用 `v.x = ...`。
- 有符号整数的 `%` 在操作数为负时，GLSL 结果未定义，与 C++ 不同。
- Vulkan 后端每次传输和 dispatch 都同步等待，简单可靠但不追求吞吐。
- OpenGL 设备只能在创建它的线程上使用。
- NVIDIA 595 驱动的 Vulkan 路径会读错"含 `vec3` 的结构体常量数组"，转译器会把这类数组改为在 `main()` 开头逐元素初始化（见 `glsl_writer.cpp` 的 `defineConstant`）。
