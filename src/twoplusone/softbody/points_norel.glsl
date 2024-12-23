#version 460

#ifdef VERTEX_SHADER

    #pragma vscode_glsllint_stage : vert
    #pragma shader_stage(vertex)

    // we just read from the particles buffer used in the rk4
    // with appropriate stride and offset
    // keeps it simple :)
    layout(location = 0) in vec2 pos;
    layout(location = 1) in uint object_index;

    layout(location = 0) out flat uint out_object_index;

    // we can use the rest_mass parameter of the particles
    // to indicate whether it's been culled
    // when we're testing our culler :D
    //layout(location = 1) in float was_culled_indicator;

    layout(push_constant) uniform Info {
        vec2 worldspace_to_screenspace;
        vec2 camera_position;
    };

    void main() {
        // gl_Position = vec4(float(gl_VertexIndex)/10.0, 0.0, 0.0, 1.0);
        gl_Position = vec4(worldspace_to_screenspace * (pos - camera_position), 0.0, 1.0);
        gl_PointSize = 1.0;
        out_object_index = object_index;
    }

#endif

#ifdef FRAGMENT_SHADER

    #pragma vscode_glsllint_stage : frag
    #pragma shader_stage(fragment)

    layout(location = 0) in flat uint object_index;

    layout(location = 0) out vec4 out_color;

    void main() {
        // temporary, for debug purposes
        if (object_index == 0)
            out_color = vec4(0.0, 0.0, 1.0, 1.0);
        else
            out_color = vec4(1.0, 0.0, 0.0, 1.0);
    }

#endif