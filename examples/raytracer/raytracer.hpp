#pragma once

#include <array>

#include <tc/tc.hpp>

namespace examples {

using namespace tc;

// Plain structs and free functions are transpiled along with the kernel:
// structs become GLSL structs, member functions become `Ray_at(ray, t)`.
struct Ray {
    vec3 origin;
    vec3 direction;

    vec3 at(float t) const { return origin + direction * t; }
};

struct Sphere {
    vec3 center;
    float radius;
    vec3 albedo;
    float reflectivity;
};

struct Hit {
    float t;
    vec3 normal;
    int sphere;
};

// Distance along `ray` to the nearest intersection with `sphere`, or -1.
inline float intersect(Sphere const &sphere, Ray const &ray) {
    vec3 oc = ray.origin - sphere.center;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0f) {
        return -1.0f;
    }
    float root = sqrt(discriminant);
    float t = -b - root;
    return t > 0.001f ? t : -b + root;
}

// The raytracer demo from the talk, reduced to spheres, hard shadows and
// mirror bounces. One invocation shades one pixel.
struct TC_KERNEL Raytracer {
    static constexpr char fileLocation[] = "raytracer";
    uvec3 local_size{8, 8, 1};

    ImageBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 0> image;
    Uniform<float, 0> time{0.0f};

    static constexpr int kBounces = 3;
    static constexpr std::array<Sphere, 4> kSpheres{
        Sphere{vec3{0.0f, -1000.0f, 0.0f}, 999.0f, vec3{0.8f, 0.8f, 0.7f}, 0.1f},
        Sphere{vec3{0.0f, 0.0f, 0.0f}, 1.0f, vec3{0.9f, 0.3f, 0.2f}, 0.4f},
        Sphere{vec3{-2.2f, -0.3f, 0.5f}, 0.7f, vec3{0.2f, 0.5f, 0.9f}, 0.2f},
        Sphere{vec3{2.0f, -0.5f, -0.4f}, 0.5f, vec3{0.9f, 0.9f, 0.9f}, 0.8f},
    };

    Hit trace(Ray const &ray) const {
        Hit hit{1e30f, vec3{0.0f}, -1};
        for (int i = 0; i < 4; ++i) {
            float t = intersect(kSpheres[i], ray);
            if (t > 0.001f and t < hit.t) {
                hit.t = t;
                hit.sphere = i;
                hit.normal = normalize(ray.at(t) - kSpheres[i].center);
            }
        }
        return hit;
    }

    vec3 sky(vec3 direction) const {
        float h = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
        return mix(vec3{1.0f, 1.0f, 1.0f}, vec3{0.4f, 0.6f, 1.0f}, h);
    }

    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = imageSize(image);
        if (pixel.x >= size.x or pixel.y >= size.y) {
            return;
        }
        vec2 uv = (vec2(pixel) + 0.5f) / vec2(size) * 2.0f - 1.0f;
        uv.x *= float(size.x) / float(size.y);
        uv.y = -uv.y;

        float angle = time * 0.5f;
        vec3 eye{4.0f * sin(angle), 1.2f, 4.0f * cos(angle)};
        vec3 forward = normalize(vec3{0.0f, 0.0f, 0.0f} - eye);
        vec3 right = normalize(cross(forward, vec3{0.0f, 1.0f, 0.0f}));
        vec3 up = cross(right, forward);
        Ray ray{eye, normalize(forward * 1.5f + right * uv.x + up * uv.y)};

        vec3 light = normalize(vec3{1.0f, 1.0f, 0.5f});
        vec3 color{0.0f};
        vec3 throughput{1.0f};
        for (int bounce = 0; bounce < kBounces; ++bounce) {
            Hit hit = trace(ray);
            if (hit.sphere < 0) {
                color += throughput * sky(ray.direction);
                break;
            }
            Sphere sphere = kSpheres[hit.sphere];
            vec3 point = ray.at(hit.t) + hit.normal * 0.001f;
            float diffuse = max(dot(hit.normal, light), 0.0f);
            if (trace(Ray{point, light}).sphere >= 0) {
                diffuse *= 0.2f;
            }
            color += throughput * sphere.albedo * (0.1f + 0.9f * diffuse) * (1.0f - sphere.reflectivity);
            throughput *= sphere.albedo * sphere.reflectivity;
            ray = Ray{point, reflect(ray.direction, hit.normal)};
        }
        vec3 corrected = pow(color, vec3{1.0f / 2.2f});
        imageStore(image, pixel, vec4(corrected, 1.0f));
    }
};

} // namespace examples
