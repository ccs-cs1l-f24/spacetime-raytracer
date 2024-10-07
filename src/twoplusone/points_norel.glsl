#version 460

#ifdef VERTEX_SHADER

    #pragma vscode_glsllint_stage : vert

    // we just read from the particles buffer used in the rk4
    // with appropriate stride and offset
    // keeps it simple :)
    layout(location = 0) in vec2 pos;

    layout(push_constant) uniform Info {
        vec2 worldspaceToScreenspace;
    };

    void main() {
        // gl_Position = vec4(float(gl_VertexIndex)/10.0, 0.0, 0.0, 1.0);
        gl_Position = vec4(worldspaceToScreenspace * pos, 0.0, 1.0);
        gl_PointSize = 1.0;
    }

#endif

#ifdef FRAGMENT_SHADER

    #pragma vscode_glsllint_stage : frag

    layout(location = 0) out vec4 out_color;

    void main() {
        out_color = vec4(0.0, 0.0, 1.0, 1.0);
    }

#endif