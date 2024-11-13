#version 460

#pragma vscode_glsllint_stage : comp
#pragma shader_stage(compute)

#include "common.glsl"
#include "relativity.glsl"

#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) buffer LightSources {
    //
};

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
// layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;

// gl_GlobalInvocationID
void main() {
    //
}

// RESOURCES:
//
// https://edw.is/learning-vulkan/
// ^ this one is standout (bindless textures + buffer addresses + other stuff); no raytracing help though
//
// ash/rust full pipeline | https://github.com/hatoo/ash-raytracing-example/blob/master/ash-raytracing-example/src/main.rs
//
// fancier illumination | https://developer.nvidia.com/blog/rtx-global-illumination-part-i/
//
// nvpro samples
//  https://github.com/nvpro-samples/nvpro_core/tree/master/nvvk
//  https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/master/ray_tracing_reflections/shaders/raytrace.rgen
//  https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/vkrt_tutorial.md.html
//
// vulkano docs | https://docs.rs/vulkano/latest/vulkano/acceleration_structure/index.html
