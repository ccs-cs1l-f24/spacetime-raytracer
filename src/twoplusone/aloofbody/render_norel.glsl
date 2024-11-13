#version 460

#ifdef VERTEX_SHADER

    #pragma vscode_glsllint_stage : vert
    #pragma shader_stage(vertex)

    layout(location = 0) in vec3 pos;

    layout(push_constant) uniform Info {
        mat4 projview_matrix;
    };

    void main() {
        gl_Position = projview_matrix * vec4(pos, 1.0);
    }

#endif

#ifdef FRAGMENT_SHADER

    #pragma vscode_glsllint_stage : frag
    #pragma shader_stage(fragment)

    layout(location = 0) out vec4 out_color;

    void main() {
        out_color = vec4(0.0, 0.0, 1.0, 1.0);
    }

#endif