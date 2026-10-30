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
| Vulkan 1.0（外部帧和子组需要 1.1） | `vulkan` | Linux、Android；Windows 与 macOS（MoltenVK）未验证 | SPIR-V |

在 Linux（RTX 5070 Ti + Mesa llvmpipe/lavapipe）上，所有示例和等价性测试在 CPU、GL、GLES、Vulkan 上结果一致；设备没有的可选能力（子组、64 位整数、某些压缩格式）对应的测试会跳过，见下文"设备能力"。`--device llvmpipe` 可以选软件实现。

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
        ├─ 表（TableBinding）：按紧凑字流生成逐字段的 get/set 函数，尽量用 uvec4 宽读写；
        │    布局的 fieldAddress 照常转译，fieldAlignment 在构建时由 Clang 常量求值
        ├─ 生成三种方言：GL 430 / ES 310 / Vulkan 450（uniform 变为 push constant）
        ├─ glslangValidator 校验，并把 Vulkan 版编译成 SPIR-V
        └─ 生成 <header>.tc.cpp：内嵌 GLSL + SPIR-V + 反射数据，静态注册到 ProgramRegistry；
             有表时还用 static_assert 检查主机与转译器算出的字流布局一致
```

| 目录 | 内容 |
|---|---|
| `tc/` | 头文件 DSL：`vec`/swizzle、`mat`、GLSL 内建函数（含 pack/unpack、位操作、原子操作、子组操作）、`BufferResource`/`BufferBinding`、`TableResource`/`TableBinding` 与布局 `Aos`/`Soa<P>`、`Encoded` 与编码器、像素与图像格式、`half`、`ImageBinding`、`TextureBinding`、`MipChain`、`CompressedImage`、`ExternalImage`、`Shared`/`barrier()`、`Uniform`、`Device` 抽象（含 `Submission`、`SyncFence`、`DeviceLimits`）、`ProgramRegistry` |
| `backends/cpu` `opengl` `vulkan` | 各后端，各自一个工厂函数 `tc::<backend>::createDevice(config)` |
| `backends/devices` | 按名字创建后端：`tc::createDevice({"vulkan", "llvmpipe"})` |
| `transpiler/` | `tc-transpile` 主机工具 |
| `cmake/TcKernels.cmake` | `tc_add_kernels(<target> HEADERS ...)` |
| `examples/` | `float_adder`、`game_of_life`、`raytracer`（对应演讲的三个案例）；`nv12_to_rgba`（纹理 + 矩阵 uniform 做格式转换）；`external_frame`（仅 Android，零拷贝导入 `AHardwareBuffer`） |
| `tests/` | DSL 单元测试、跨后端等价性测试、转译器报错测试 |
| `benchmarks/` | `tc_bench`：各用例在各后端上的 GPU 时间、墙钟时间和带宽 |

生成的 GLSL 在 `build/<preset>/.../tc_generated/<target>/glsl/*.comp`，可以直接查看。

## 写 kernel

kernel 是带 `TC_KERNEL` 标注的 struct，必须有 `fileLocation`、`local_size` 和 `void main()`。可用的成员：

- `BufferBinding<T, binding>`：SSBO，`A[i]` 读写，`A.length()`。`T` 可以是标量、`vec`、或只含这些成员的结构体，C++ 布局必须和 std430 一致（由转译器检查）。
- `TableBinding<T, binding, Layout = Aos>`：表，元素按紧凑字流存放，结构体可以按自然的写法定义，可以切换 AOS / SOA 布局，字段可以带编码，见下文"表"。
- `ImageBinding<GpuFormat, Dim, CpuPixel, binding>`：存储图像，配合 `imageLoad` / `imageStore` / `imageSize`。GPU 内部格式和主机像素格式可以不同（例如主机用 `cpu::R8UI`，GPU 用 `gpu::R32UI`），上传/下载时自动转换；CPU 后端读写同样经过 GPU 格式，结果与 GPU 一致。可以 attach 一个 `BufferResource`，或 `MipChain` 的某一层（见下文"纹理"）。
- `TextureBinding<GpuFormat, Dim, CpuPixel, binding, Filter, Wrap, Mip>`：采样纹理（`sampler2D` 等），配合 `texelFetch` / `texture` / `textureLod` / `textureSize`。过滤（`Nearest`/`Linear`）、寻址（`ClampToEdge`/`Repeat`/`MirroredRepeat`）和层级选择（`Mip::None`/`Nearest`/`Linear`）是类型的一部分，CPU 后端按 GL 规则实现双线性与三线性采样。纹理用 `uploadImage` 上传，用 `bindTexture` 绑定。
- `CompressedTextureBinding<binding, Filter, Wrap, Mip>`：BC、ETC2/EAC、ASTC 压缩纹理，用法同上，见下文"纹理"。
- `ExternalTextureBinding<binding>`：外部帧（见下文"零拷贝"），配合 `texture` / `textureSize`，结果是 RGB，YUV 由驱动或 CPU 转换。
- `ExternalImageBinding<GpuFormat, binding>`：把 RGBA 外部帧当作存储图像直接写入，见下文"零拷贝"。
- `Shared<T>`：工作组共享内存（GLSL `shared`），`T` 可以是标量、`vec`、`mat` 或 `std::array`。用到 `barrier()` 的 kernel 必须声明 `static constexpr bool usesBarrier = true;`，否则转译时报错。用到子组操作的 kernel 同样要声明 `usesSubgroups`（见下文"子组与 64 位整数"）。
- `Uniform<T, location>`：主机每次 dispatch 前用 `setUniform` 设置。`T` 可以是标量、`vec` 或 `mat`。
- 其他数据成员和 `static constexpr` 成员：变成 GLSL 常量，初始化式必须是常量表达式。`static constexpr tc::Precision floatPrecision = tc::Precision::Medium;` 是例外，它让 GLES 和 Vulkan 以中等精度计算（见下文"中等精度"）。

`main()` 和它调用到的成员函数、自由函数、普通结构体（含成员函数）都会被转译。支持：标量/向量/矩阵运算、`if`/`for`/`while`/`do`/`switch`、`std::array`（含 range-for）、`<cmath>` 常用函数、GLSL 内建函数（`tc::` 下同名实现）：

- 矩阵：`mat2`…`mat4x3`，列主序、布局与 std430 一致，`m[col]`、`m * v`、`transpose`、`inverse`、`determinant`、`outerProduct`、`matrixCompMult`。
- 打包与位操作：`packUnorm4x8`/`unpackUnorm4x8` 等 5 组 pack/unpack，`bitfieldExtract`/`bitfieldInsert`/`bitfieldReverse`/`bitCount`/`findLSB`/`findMSB`。
- 原子操作：`atomicAdd`/`And`/`Or`/`Xor`/`Min`/`Max`/`Exchange`/`CompSwap`，目标只能是 buffer 元素或 `Shared` 变量（转译器检查）；目标是 `tc::i64`/`tc::u64` 的 buffer 元素时为 64 位原子操作。
- 子组：`subgroupAdd`、`subgroupBallot`、`subgroupShuffle` 等，见下文。
- 同步：`barrier()`，以及 `memoryBarrier*()` / `groupMemoryBarrier()`。
- 用户类型的运算符和转换：成员或自由的 `operator+`、`operator*`、`operator==`、`operator[]`、`operator()`，以及 `explicit operator vec2()` 这样的转换函数，都转译成普通 GLSL 函数（如 `Complex_op_mul`、`Complex_to_vec2`）。返回引用的运算符（`operator+=` 返回 `*this`）不支持，转译器会提示改成按值返回。
- 函数模板：每个实例得到独立的 GLSL 名字（`scaled<2>` → `scaled_2`，`twice<float>` → `twice_float`）；函数与常量同名时自动加后缀。

因为 C++17 不允许字符串字面量作模板参数，swizzle 写成成员函数：`v.xy()`、`v.zyx()`（转译成 `v.xy`、`v.zyx`），单分量直接用 `v.x`。

kernel 本身也可以是类模板。每个实例是一个独立的程序，名字在显式实例化处给出：

```cpp
template <class Material, uint GroupSize = 32>
struct TC_KERNEL Shade {
    static constexpr char const *fileLocation = tc::instanceName<Shade>();
    uvec3 local_size{GroupSize, 1, 1};
    BufferBinding<vec4, 0> colors;
    void main() { /* 用 Material::shade(...) */ }
};
TC_KERNEL_INSTANCE("shade_lambert", Shade<Lambert>);
TC_KERNEL_INSTANCE("shade_toon", Shade<Toon, 64>);
```

`TC_KERNEL_INSTANCE` 写在模板之后、同一个命名空间里。转译器只生成头文件里列出的实例，两个实例同名时报错；主机代码使用没有列出的实例时编译失败。

CPU 后端执行 `usesBarrier` kernel 时，一个工作组的所有 invocation 作为协作纤程运行（x86_64 / aarch64 上的 Linux 与 Android，手写上下文切换，每个纤程 64 KiB 栈加保护页），`barrier()` 处让出，全部到齐后再继续；其他平台退化为每个 invocation 一个线程。如果部分 invocation 没有到达同一个 `barrier()`（GPU 上是未定义行为），CPU 后端会直接报错终止，而不是给出一个碰巧正确的结果。

## 表：紧凑布局、AOS/SOA 与字段编码

思路来自 SaeruHikari 的文章[《纯血 C++ 着色器：AOS/SOA 自动布局与加载优化》](https://zhuanlan.zhihu.com/p/2097630291724612499)：数据的含义（结构体类型）和存放方式（布局、编码）分开，kernel 代码不随存放方式改变。

`BufferBinding` 要求 C++ 布局与 std430 一致，所以 `vec3` 在 C++ 里占 16 字节，`struct { vec3 position; float mass; }` 这样自然的写法会被拒绝。`TableBinding` 不受这个限制：元素被压成没有空洞的 32 位字流（`vec3` 3 个字，`mat3` 9 个字，`bool` 1 个字），由转译器生成打包和解包的 GLSL。

```cpp
struct Spark {                       // 8 个字；C++ 里 sizeof 是 48
    vec3 position;
    float mass;
    vec3 velocity;
    uint flags;
};

struct TC_KERNEL Step {
    static constexpr char fileLocation[] = "step";
    uvec3 local_size{256, 1, 1};
    TableBinding<Spark, 0> sparks;   // 换成 TableBinding<Spark, 0, Soa<>> 即为 SOA

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(sparks.length()) or (sparks[i].flags & 1u) == 0u) return;  // 只读 flags 一个字段
        Spark s = sparks[i];                 // 读整个元素
        s.position += s.velocity;
        sparks[i] = s;                       // 写整个元素
        sparks[i].velocity.y -= 9.81f;       // 改一个字段的一个分量
    }
};
```

```cpp
auto sparks = tc::TableResource<Spark>{n};      // 或 device->createTable<Spark>(n)，零拷贝
sparks.set(i, Spark{...});   Spark s = sparks.get(i);
device->uploadTable(sparks);  device->bindTable(kernel.sparks);  device->downloadTable(sparks);
```

规则：

- 在 kernel 里，`t[i]` 读整个元素，`t[i].field` 在 GPU 上只读这个字段。写入必须是单独的一条语句：`t[i] = v;`、`t[i].field = v;` 直接写；`t[i].field.x += v;` 这类部分写入会展开成"读出字段、修改、写回"。表元素不能绑定到引用，不能传给非 const 引用参数，也不能做原子操作。
- 元素可以含 `bool`、`int32_t`、`uint32_t`、`float`、`double`、32 位枚举、`vec`/`mat`、`std::array`、嵌套结构体和 `Encoded` 字段；不能有 C 数组、指针或基类。C++17 没有反射，主机端通过聚合初始化和结构化绑定逐个找到字段，最多 32 个。
- 缓冲区开头有 4 个字的头部，第 0 个字是元素个数（`t.length()`）。
- CPU 后端上，`t[i]` 返回当前线程缓存的元素副本，每个 invocation 结束时（用 `barrier()` 的 kernel 在工作组结束时）只把改动过的字写回，所以并发改同一元素不同字段的工作组不会互相覆盖。

### 布局

布局是一个只含 constexpr 函数的结构体，主机直接调用；转译器把 `fieldAddress` 当作普通代码转译成 GLSL，并在构建时求值 `fieldAlignment`，以决定能否用 `uvec2`/`uvec4` 宽读写。

| 布局 | 存放方式 | 适合 |
|---|---|---|
| `Aos` | 元素一个接一个 | 整个元素都要读写 |
| `Soa<P>` | 每 P 个元素一页，页内每个字段连续成一列（AoSoA） | 只读部分字段、提前返回 |
| 自定义 | 实现 `contiguousRecords`、`dataWords`、`fieldAddress`、`fieldAlignment`（接口见 `tc/table.hpp`） | 例如把元素补齐到 16 字节以便宽读写（`tests/kernels/feature_kernels.hpp` 的 `PaddedAos`） |

### 字段编码

`Encoded<V, Codec>` 字段在 kernel 和主机代码里是普通的 `V`（`.value` 或隐式转换），存进表时由编码器压缩，读出时还原。编码器的 `encode`/`decode` 也是普通的 kernel C++，用户可以自己写（测试里的 `Unorm10x3` 就是一个例子）。只有表会应用编码；局部变量和 std430 buffer 里存的还是完整的 `V`。

| 编码器 | 值 | 字数 |
|---|---|---|
| `Unorm8x4` / `Snorm8x4` | `vec4` | 1（原本 4） |
| `Unorm16x2` / `Half2` | `vec2` | 1（原本 2） |
| `Half4` | `vec4` | 2（原本 4） |
| `Affine3x4` | `mat4`，最后一行固定为 (0, 0, 0, 1) | 12（原本 16） |
| `OctahedralNormal` | 单位 `vec3` | 1（原本 3），误差小于 1e-4 |

## 纹理：mip、数组、半精度与压缩格式

`MipChain<Pixel, Dim>` 是带 mip 层的图像：第 0 层是给定的尺寸，之后每层减半，直到 1 个像素（默认生成完整的链）。纹理 binding 采样整条链，图像 binding 写其中一层，所以可以用 kernel 逐层生成 mip（测试里的 `Downsample`）：

```cpp
auto chain = tc::MipChain<cpu::RGBA8, Dim::D2>{ivec2{256, 256}};    // 9 层
chain.level(0)[ivec2{x, y}] = ...;
kernel.source.attach(chain);         // TextureBinding<..., Mip::Nearest>：读第 n - 1 层
kernel.target.attach(chain, n);      // ImageBinding：写第 n 层
device->uploadImage(kernel.source);  // 所有层；uploadImage(binding, level) 只传一层
```

- `textureLod(t, uv, lod)` 按 binding 的 `Mip` 参数选层：`None` 只用第 0 层，`Nearest` 取最近的层，`Linear` 在相邻两层之间插值。compute shader 没有导数，`texture()` 等于 `lod = 0`。`texelFetch` 和 `textureSize` 的最后一个参数是层号。
- `Dim::D2Array` 是 2D 纹理数组（`sampler2DArray` / `image2DArray`）：资源尺寸为 `ivec3{宽, 高, 层数}`，`texture` 的坐标是 `vec3{u, v, 层}`（层号四舍五入并限制在有效范围内），`texelFetch` / `imageStore` 用 `ivec3{x, y, 层}`。
- 16 位浮点格式 `gpu::R16F` / `RG16F` / `RGBA16F`：主机像素可以是 `cpu::RGBA16F`（分量为 `tc::half`），也可以是 `cpu::RGBA32F`，上传和下载时转换。`rgba16f` 是 GLES 3.1 和 Vulkan 都强制支持的存储图像格式。

压缩纹理放在 `CompressedImage` 里，每层的数据是按行排列的压缩块，也就是 KTX、DDS 文件里存的原始内容；kernel 通过 `CompressedTextureBinding` 采样，得到归一化的 RGBA：

```cpp
using tc::CompressedFormat;
auto image = tc::CompressedImage{CompressedFormat::Astc6x6, ivec2{1024, 1024}, levels};
std::memcpy(image.blocks(0).data(), level0, image.blocks(0).size());   // 每层的大小已分配好
if (device->info().limits.compressedFormats & tc::compressedBit(CompressedFormat::Astc6x6)) {
    kernel.albedo.attach(image);
    device->uploadImage(kernel.albedo);
    device->bindTexture(kernel.albedo);
}
```

支持 BC1/3/4/5/7、ETC2 RGB8/RGB8A1/RGBA8、EAC R11/RG11、ASTC 4x4 到 12x12（都是 LDR、非 sRGB）。GPU 后端把压缩块直接交给驱动，设备不支持的格式在 `uploadImage` 时报 `DeviceError`。CPU 后端在上传时解码成 RGBA32F 的 mip 链再采样，解码器为 [bcdec](https://github.com/iOrange/bcdec)、[etcdec](https://github.com/iOrange/etcdec) 和 [astc-encoder](https://github.com/ARM-software/astc-encoder)，构建时通过 `FetchContent` 下载并校验哈希。

## 子组与 64 位整数

声明了 `static constexpr bool usesSubgroups = true;` 的 kernel 可以使用 `GL_KHR_shader_subgroup` 的 basic、vote、arithmetic、ballot、shuffle 和 relative shuffle 操作：`subgroupElect`、`subgroupAll` / `Any` / `AllEqual`、`subgroupAdd` / `Mul` / `Min` / `Max` / `And` / `Or` / `Xor` 以及对应的 `subgroupInclusive*` / `subgroupExclusive*` 扫描、`subgroupBallot` 系列、`subgroupBroadcast` / `BroadcastFirst`、`subgroupShuffle` / `ShuffleXor` / `ShuffleUp` / `ShuffleDown`，还有 `gl_SubgroupSize`、`gl_SubgroupInvocationID`、`gl_SubgroupEqMask` 等内建变量。没有声明就使用时，转译器报错。

```cpp
struct TC_KERNEL GroupSums {
    static constexpr char fileLocation[] = "group_sums";
    static constexpr bool usesSubgroups = true;
    uvec3 local_size{64, 1, 1};
    BufferBinding<uint, 0> values;
    BufferBinding<uint, 1> total;

    void main() {
        uint sum = subgroupAdd(values[gl_GlobalInvocationID.x]);
        if (subgroupElect()) {
            atomicAdd(total[0], sum);
        }
    }
};
```

GPU 设备在 `DeviceLimits::features` 含 `KernelFeature::Subgroups` 时才能运行这类 kernel，子组大小见 `DeviceLimits::subgroupSize`。CPU 后端把一个工作组的 invocation 作为纤程运行，按 `CpuDeviceConfig::subgroupSize`（默认 32，取 1 到 128 之间的 2 的幂）分成子组；每个子组操作要等组里所有活跃的 invocation 都到达后才一起计算，提前 `return` 的 invocation 不再参与，和 GPU 一样。ballot 的位、shuffle 等结果依赖子组大小，在不同设备上会不同，kernel 里应该用 `gl_SubgroupSize` 而不是写死 32。

`tc::i64` / `tc::u64`（即 `long long`）转译为 GLSL 的 `int64_t` / `uint64_t`，可以用在 buffer 元素、局部变量和运算中，buffer 里的 64 位元素也可以做原子操作。`std::int64_t` 在 Linux 上是 `long`，和 `std::size_t` 一样在 GPU 上仍是 32 位，转译器会对这样的局部变量给出警告。用到它们的 kernel 需要 `KernelFeature::Int64`，原子操作还需要 `KernelFeature::Int64Atomics`。GLSL ES 没有 64 位整数，这类 kernel 在 `gles` 后端上会被拒绝。

## 中等精度

```cpp
struct TC_KERNEL Tonemap {
    static constexpr tc::Precision floatPrecision = tc::Precision::Medium;
    ...
};
```

GLES 版本的 GLSL 在函数之前加上 `precision mediump float;`，Vulkan 的 SPIR-V 中对应的运算带 `RelaxedPrecision` 修饰。在移动 GPU 上，这通常意味着用 16 位浮点计算，寄存器占用和功耗都更低。buffer、图像、uniform、共享变量和常量仍保持完整精度。桌面 OpenGL 和 CPU 后端照常用 32 位计算，所以结果会有 fp16 舍入量级的差异（测试中的容差为 1.5e-2）。显式的 fp16 运算（`float16_t`）目前还不支持。

## 调度、提交与同步

**间接调度**：工作组数量可以由前一个 kernel 写进 buffer，主机不需要读回。

```cpp
auto args = device->createBuffer<tc::DispatchIndirectCommand>(1);   // {x, y = 1, z = 1}
cull.args.attach(args);           // CullSpheres 每找到 64 个可见球体，就 atomicAdd(args[0].x, 1u)
device->execute(cull, uvec3{n / 64, 1, 1});
device->executeIndirect(shade, args);                               // 读 args[0]
```

**部分传输与区间绑定**：`uploadBuffer(resource, first, count)` / `downloadBuffer(resource, first, count)` 只传输其中一段元素，表也有对应的 `uploadTable` / `downloadTable`。`BufferBinding::attach(resource, first, count)` 只把一段元素绑定给 kernel，kernel 里的下标从这一段的开头算起，`length()` 是这一段的长度；在 GPU 后端上，起点的字节偏移必须是 `DeviceLimits::storageBufferOffsetAlignment` 的倍数（规范允许的最大值是 256）。SOA 表的一个元素分散在整页里，部分上传或下载时，同一页其他元素的字也会一起传输。

**设备能力**：`device->info().limits` 给出每个 kernel 可用的 buffer、图像和纹理数量、共享内存、工作组大小、push constant 大小、可选能力（`KernelFeature`）和支持的压缩格式。转译器记录每个 kernel 用到了什么，`device->checkKernel("name")` 返回 kernel 不能在这台设备上运行的原因（超出限制，或用到了设备没有的能力），可以运行时返回 `nullopt`；在启动时对 `ProgramRegistry::instance().all()` 逐个检查，可以在任何工作开始前发现问题。`useKernel` 也会做同样的检查，失败时抛出 `DeviceError`。已测试设备的可选能力：

| 设备 | 子组 | 64 位整数 / 原子 | 压缩格式 | sync file |
|---|---|---|---|---|
| RTX 5070 Ti，`opengl` | 有（32） | 有 / 有（`GL_NV_shader_atomic_int64`） | BC、ETC2/EAC | 有 |
| RTX 5070 Ti，`gles` | 有（32） | 无 / 无 | BC、ETC2/EAC、ASTC | 有 |
| RTX 5070 Ti，`vulkan` | 有（32） | 有 / 有 | BC | 有 |
| llvmpipe，`opengl` / `gles` | 无 | 有 / 无（`gles` 均无） | BC、ETC2/EAC、ASTC | 无 |
| lavapipe，`vulkan` | 有 | 有 / 有 | BC | 无 |
| Adreno 830，`gles` | 无 | 无 / 无 | BC、ETC2/EAC、ASTC | 有 |
| Adreno 830，`vulkan` | 有 | 有 / 有 | BC、ETC2/EAC、ASTC | 有 |
| `cpu` | 有（可配置） | 有 / 有 | 全部 | 不需要 |

**异步提交**：`device->submit()` 把目前排队的工作交给设备，不等待完成，返回一个 `Submission`；`isComplete()` 查询，`wait()` 等待，任何线程都可以使用，并且可以比设备活得更久。设备执行这一帧时，主机可以准备下一帧，只要不写还在被读的内存即可：对 `createBuffer` 分配的内存，每一帧在途时用环形缓冲区的不同一段。`Submission` 完成后，`createBuffer` 内存里就是结果，所以它也是异步读回的方式；`downloadBuffer` / `downloadImage` 总是会等待。Vulkan 后端每次提交都有自己的命令缓冲区、暂存内存和 fence，GL 后端用 EGL fence sync。

**按需屏障**：后端记录每次 dispatch 和传输读写了哪些资源的哪段字节，只在真正有读后写、写后读或写后写冲突时才插入屏障（Vulkan 的 pipeline barrier，GL 的 `glMemoryBarrier`）。互不相关的连续 dispatch 之间没有屏障，可以重叠执行。

**CPU 线程**：CPU 后端默认使用自己的常驻线程池。引擎可以实现 `tc::cpu::ParallelExecutor`（`concurrency()` 和 `bulk(count, task)` 两个函数），通过 `CpuDeviceConfig::executor` 交给 CPU 后端，让 kernel 和引擎共用线程；`tc::cpu::makeThreadPool(n)` 返回 tc 自带的实现，可以由多个设备共享。

## 开发时重新加载 kernel

开发时改了 kernel 的函数体，不需要重启程序：重新构建，`tc-transpile` 会重写 `tc_generated/<target>/glsl` 里的 GLSL 和 SPIR-V，然后在运行中的程序里调用

```cpp
auto const report = tc::ProgramRegistry::instance().reloadFrom(".../tc_generated/<target>/glsl");
// report.reloaded：替换了的程序；report.rejected：程序名和原因
```

GPU 设备在下一次 `useKernel` 时改用新程序。只有接口不变的程序才会被替换：转译器在 GLSL 旁边写出 `<name>.interface`，记录工作组大小、binding（含推断出的读写方式）、uniform、push constant、共享内存和用到的可选能力；新增 binding、改变工作组大小等需要重新编译主机代码。CPU 后端始终执行编译进来的 C++。CMake 选项 `TC_PROGRAM_RELOAD`（默认 `ON`）关闭后，`reloadFrom` 不读磁盘，并在报告里说明原因，发布版本应关闭它。

## 性能测量

`device->enableTimers(true)` 之后，每次 `execute()` 都在设备上计时：Vulkan 用时间戳查询，桌面 GL 用 `GL_TIME_ELAPSED`，GLES 用 `GL_EXT_disjoint_timer_query`，CPU 后端用主机时钟。`device->takeTimings()` 等待已提交的工作完成，按执行顺序返回每次 dispatch 的耗时；GLES 报告时钟不连续时，对应的结果为空。`DeviceInfo::timers` 表示设备是否支持计时。

`tc_bench` 在所有可用后端上跑一组用例，先预热，再按时间预算决定迭代次数，报告 GPU 时间中位数、最小值、墙钟时间和有效带宽：

```sh
./build/linux-release/benchmarks/tc_bench                         # 全部用例、全部后端
./build/linux-release/benchmarks/tc_bench --backend vulkan --case cull,shade --csv out.csv
./build/linux-release/benchmarks/tc_bench --scale 0.25            # 手机上缩小问题规模
```

RTX 5070 Ti、Vulkan 上的部分结果（GPU 时间，毫秒；OpenGL 和 GLES 基本相同）：

| 对比 | 用例 | 时间 |
|---|---|---|
| 打包本身没有开销（同为每个粒子 32 字节） | `particles_std430` / `particles_table` / `particles_soa` | 0.171 / 0.171 / 0.171 |
| 只读部分字段时 SOA 快约 2 倍 | `cull_aos` / `cull_soa` | 0.169 / 0.079 |
| 编码后的字段读得更快（每个元素 25 → 15 个字） | `shade_plain` / `shade_encoded` | 0.292 / 0.189 |
| `vec3` 不再补齐到 16 字节 | `points_vec4_std430` / `points_vec3_table` | 0.171 / 0.097 |

同一组对比在 Adreno 830 的 Vulkan 上（`--scale 0.25`）也成立：`particles_*` 0.503 / 0.517 / 0.507，`cull_aos` / `cull_soa` 0.647 / 0.299，`shade_plain` / `shade_encoded` 0.888 / 0.576，`points_vec4_std430` / `points_vec3_table` 0.498 / 0.370。该机 GLES 的 `GL_EXT_disjoint_timer_query` 会给出接近 0 的时间，带宽请看墙钟或改用 Vulkan 时间戳。

一开始表的访问函数按 32 位逐字读取，`particles_table` 比 std430 慢 30%～50%（Vulkan / GL）；改成按 `uvec4` 读写后才持平。是否能宽读写由布局的 `fieldAlignment` 决定。

## 零拷贝与外部帧

`device->createBuffer<T>(n)` 返回的 `BufferResource` 直接指向设备分配、映射到主机的内存：Vulkan 用 host-visible 的设备内存，GL/GLES 用持久映射（`glBufferStorage`，GLES 需要 `GL_EXT_buffer_storage`），CPU 后端就是普通内存。这样的 buffer 在主机上直接读写，`uploadBuffer` / `downloadBuffer` 只负责同步，不再复制数据；不调用它们而直接读写时，先 `device->finish()`。驱动不支持映射时自动退回普通 buffer，接口不变。`DeviceInfo::unifiedMemory` 表示主机写入的内存是否就是 GPU 读取的内存（手机等统一内存架构，或开启 ReBAR 的独立显卡）。

Vulkan 后端会把连续的传输和 dispatch 录进同一个命令缓冲区，到 `submit()` 时提交，或者在需要把结果交给主机时（`downloadBuffer`/`downloadImage`/`finish()`）提交并等待。`device->finish()` 等待所有已提交的工作完成。

在 Android 上，`tc::ExternalImage::fromHardwareBuffer(ahb)` 包装相机、解码器或其他来源的 `AHardwareBuffer`（RGBA 或 YUV 420），用 `bindExternalTexture` 绑定到 `ExternalTextureBinding`，不经过任何复制：

| 后端 | 导入方式 | YUV → RGB |
|---|---|---|
| `gles` | `EGLImage` → `GL_TEXTURE_EXTERNAL_OES`，GLSL 中为 `samplerExternalOES` | 驱动 |
| `vulkan` | `VK_ANDROID_external_memory_android_hardware_buffer`，`VkSamplerYcbcrConversion`（BT.601 有限范围） | 采样器硬件 |
| `cpu` | 每次 dispatch 前锁定 buffer 读取一份快照 | BT.601 有限范围 |

`examples/external_frame` 演示了完整流程：分配 YUV 和 RGBA 两个 `AHardwareBuffer`，填入测试图，转换成 RGBA8 写进 `createBuffer` 分配的 buffer，并与期望值比较；然后用下文的 `ExternalImageBinding` 连续渲染 3 帧到一个 RGBA `AHardwareBuffer`，帧与帧之间用 sync file 交接。在 Adreno 830 上，三个后端对两种格式的结果都正确（RGBA 平均误差 0.08/255，YUV 在 GPU 上 0.84/255），写入外部帧的结果三个后端一致，GLES 与 Vulkan 的 fence 都在设备上等待和发出。

Adreno 的 GLES 驱动会在一个 program 第一次 dispatch 时，固定其中 `samplerExternalOES` 的颜色转换方式，之后换成别的格式的帧，仍然按第一帧的格式转换。为此，GL 后端对每种外部帧格式组合各链接一次同一份着色器。

### 写入外部帧与 fence

RGBA 外部帧（`AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM` 或 `R16G16B16A16_FLOAT`，分配时带 `GPU_SAMPLED_IMAGE | GPU_FRAMEBUFFER` 用途）还可以通过 `ExternalImageBinding<gpu::RGBA8, binding>` 作为存储图像直接写入，例如 kernel 直接渲染进合成器要显示的 buffer。与合成器、相机等其他使用者之间用 sync file 同步：

```cpp
auto frame = tc::ExternalImage::fromHardwareBuffer(ahb);
kernel.target.attach(frame);                     // ExternalImageBinding<gpu::RGBA8, 0>
device->waitFor(std::move(releaseFence));        // 合成器还给我们的 release fence
device->useKernel(kernel);
device->bindExternalImage(kernel.target);
device->execute(kernel, groups);
tc::SyncFence done = device->submitWithFence();  // 交给 ASurfaceTransaction_setBuffer 作为 acquire fence
```

| 后端 | 写入外部帧 | `waitFor` / `submitWithFence` |
|---|---|---|
| `gles` | `EGLImage` + `GL_EXT_EGL_image_storage` | `EGL_ANDROID_native_fence_sync` |
| `vulkan` | 以 `STORAGE` 用途导入 `AHardwareBuffer`，在 foreign 队列族之间转移所有权 | `VK_KHR_external_semaphore_fd` 的 sync fd 导入与导出 |
| `cpu` | 绑定时读一份快照，每次 `execute()` 后写回 | 在主机上等待；返回空 fence |

`DeviceInfo::syncFiles` 为假时，`waitFor` 在主机上等待，`submitWithFence` 等工作完成后返回空 fence。`SyncFence` 只是 Linux / Android 的 sync file 描述符，桌面 Linux 上同样可用（NVIDIA 的三种 API 都支持导入和导出）。

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

CPU 后端的压缩纹理解码器（bcdec、etcdec、astc-encoder 5.7.0）在配置时通过 `FetchContent` 下载，版本和哈希都已固定。发布版本请加 `-DTC_PROGRAM_RELOAD=OFF`，去掉从磁盘重新加载 kernel 的功能。

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

`android-arm64` 预设同时构建测试（GoogleTest 通过 `FetchContent` 下载），`build/android-arm64/tests/` 下的 `tc_core_tests` 和 `tc_equivalence_tests` 用同样的方式推到设备上运行。在 Adreno 830 上，`tc_core_tests` 35 项通过（含 aarch64 纤程上的 barrier 和子组），`tc_equivalence_tests` 的 GLES 与 Vulkan 用例 105 项通过。按设计跳过的有：llvmpipe 设备、GLES 上读写非 r32 图像、GLES 上的子组与 64 位整数，以及需要构建目录的重新加载测试。

在 App 里使用时，把 `tc`、`tc_cpu`、`tc_opengl`、`tc_vulkan` 作为子项目加入 Android Studio 的 CMake 工程，`tc_add_kernels` 通过 `TC_TRANSPILER_EXECUTABLE` 使用主机上的转译器。

为了让 kernel 在所有设备上都能运行，图像请使用 GLES 3.1 和 Vulkan 都强制支持的存储格式：`rgba8`、`rgba8ui`、`rgba8i`、`r32ui`、`r32i`、`r32f`、`rgba16f`、`rgba32ui`、`rgba32i`、`rgba32f`。使用了 `double`、64 位整数或其他格式的 kernel 仍可在桌面 GL / Vulkan 上运行，`gles` 后端会给出原因并拒绝执行。子组、64 位原子和压缩格式因设备而异，启动时用 `checkKernel` 检查。

## 与演讲的差异

- C++17：concept 改成 trait + `static_assert`，swizzle 用 `v.xy()`，没有指定初始化器。
- 转译用 AST 打印而不是 Rewriter：每个隐式转换都显式输出，结果对 GLSL ES 也合法；常量初始化交给 Clang 的常量求值器。
- 一次转译产出三种方言和 SPIR-V；Vulkan 的 uniform 自动放进 push constant 块，偏移由转译器计算并写入反射数据。
- 增加了 std430 布局检查和 binding 读写推断。
- 增加了表（紧凑字流、AOS/SOA、字段编码），思路来自上文提到的知乎文章；布局和编码器写成普通 C++，由转译器照常转译。
- 增加了 kernel 模板、间接调度、mip / 数组 / 16 位浮点 / 压缩纹理、子组与 64 位整数、中等精度、异步提交与按需屏障、外部帧写入与 sync file、开发时重新加载，以及可替换的 CPU 线程池。

## 已知限制

- swizzle 只读；写分量用 `v.x = ...`。
- kernel 里的结构体不能有基类；运算符不能返回引用。
- 表字段不能做原子操作；CPU 后端读写表比读写 std430 buffer 慢约 3 倍（要逐字打包，还要经过元素缓存）。
- 有符号整数的 `%` 在操作数为负时，GLSL 结果未定义，与 C++ 不同。
- CPU 后端的原子操作是 relaxed 顺序，与 GLSL 一致；不同工作组之间没有执行顺序保证，这一点在所有后端上都一样。
- `Shared` 总量超过 16 KiB（GLES 3.1 与 Vulkan 的最低保证）时转译器给出警告。
- 外部帧的 YUV 转换在 GPU 上由驱动或采样器硬件完成，色度插值方式各家不同，与 CPU 结果有约 1/255 的平均差异。
- OpenGL 设备只能在创建它的线程上使用。
- NVIDIA 595 驱动的 Vulkan 路径会读错"含 `vec3` 的结构体常量数组"，转译器会把这类数组改为在 `main()` 开头逐元素初始化（见 `glsl_writer.cpp` 的 `defineConstant`）。
- 部分 GLES 实现（Adreno 830）的 `GL_EXT_disjoint_timer_query` 几乎总是返回接近 0 的时间且不置 disjoint；`tc_bench` 的 GPU 列会偏小，带宽请用墙钟或 Vulkan。
- CPU 后端只支持在子组一致的控制流里调用子组操作；同一子组的 invocation 停在不同的子组操作上时直接报错终止。
- llvmpipe 的 `texelFetch` / `textureSize` 在同一次 dispatch 里各 invocation 的层号不同时，会读第 0 层；层号对所有 invocation 相同时结果正确。
- `std::int64_t` 和 `std::size_t` 在 GPU 上是 32 位，64 位整数请用 `tc::i64` / `tc::u64`。
- 桌面 GL 的 64 位原子操作依赖 `GL_NV_shader_atomic_int64`；glslang 不能离线校验这类桌面 GLSL，错误会在驱动编译时才报告。
- GPU 解码 BC1～BC5 的舍入方式与 CPU 解码器不同（端点插值），NVIDIA 上最多相差约 6/255，Adreno 830 上 1/255；BC7、ETC2/EAC 一致，ASTC 在 1/255 以内。
- Adreno 830 的 GLES 驱动在一次 dispatch 同时采样一个纹理、又把它第 n > 0 层作为存储图像写入之后，读这个纹理的第 k 层会得到第 k + 1 层（dispatch 内和之后都是）。GL 后端因此通过纹理视图（`glTextureView`，GLES 上需要 `GL_OES_texture_view` 或 `GL_EXT_texture_view`）绑定第 0 层以外的存储图像；没有视图扩展的 GLES 设备仍直接绑定。
- 重新加载只接受函数体的改动，接口变化必须重新编译。
- Vulkan 第一次以存储图像使用一个新导入的外部帧时，按 `UNDEFINED` 布局取得，规范允许驱动丢弃帧里原有的内容，所以第一次写入时 kernel 应写满整帧。
- 不支持显式的 fp16 运算（`float16_t`），只有整个 kernel 的中等精度。
