Spacetime Softbody Raytracer
=

THIS IS A WORK IN PROGRESS

## Build Instructions

To build and run the program, run `cargo run`.

Your working directory should be the root of this repository, the directory `spacetime-raytracer`.

### Dependencies

Required dependencies
- Vulkan
- Rust + Cargo
- A sane window manager

Optional dependencies
- Vulkan validation layers
- shaderc

Uses custom vulkano and egui/egui_winit_vulkano forks (TODO add additional remote for them in ccs-cs1l-f24 so yall can also build)

### Building on MacOS

If you're running MacOS, you will need to have MoltenVK installed for Vulkan to work.

Applications using Vulkan will typically bundle MoltenVK, but I don't have a reference mac to test on.

You can either install it [through brew](https://formulae.brew.sh/formula/molten-vk) or as part of the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#mac).

