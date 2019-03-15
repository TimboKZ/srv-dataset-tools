#version 130

#define ShaderMode_3D 0
#define ShaderMode_Texture 1

// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 proj_ModelViewMatrix;
uniform mat4 proj_ProjectionMatrix;
uniform int ShaderMode;

// Vertex inputs
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
in vec2 p3d_MultiTexCoord1;

// Output to fragment shader
out vec4 vertex_pos;
out vec2 model_texcoord;
out vec2 projection_texcoord;

void main() {
    // Figure out the texcoord on the camera image
    vertex_pos = proj_ProjectionMatrix * proj_ModelViewMatrix * p3d_Vertex;
    vec2 vertex_pos_2d = vec2(vertex_pos[0], vertex_pos[1]) / vertex_pos[3];
    projection_texcoord = (vertex_pos_2d + 1.0) / 2.0;

    // Figure out the positon on screen based on model texcoord
    model_texcoord = p3d_MultiTexCoord0;

    if (ShaderMode == ShaderMode_3D) {
        gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    } else {
        vec2 tex = model_texcoord * 2.0 - 1.0;
        gl_Position = vec4(tex, 0, 1);
    }
}