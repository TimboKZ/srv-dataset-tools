#version 130

#define ShaderViewMode_3D 0
#define ShaderViewMode_Texture 1
#define ShaderTextureMode_Default 0
#define ShaderTextureMode_Projection 1
#define ShaderTextureMode_Normal 2
#define ShaderTextureMode_Mask 3
#define ShaderTextureMode_Visibility 4

// Shader parameters
uniform int ShaderViewMode;
uniform int ShaderTextureMode;

// Uniform inputs
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 proj_ModelViewMatrix;
uniform mat4 proj_ProjectionMatrix;

// Vertex inputs
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

// Output to fragment shader
out vec4 vertex_pos_proj;
out vec4 vertex_pos_3d;
out vec3 vertex_normal;
out vec2 model_texcoord;
out vec2 projection_texcoord;

void main() {
    // Positions of the vertex in different perspectives
    vertex_pos_3d = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vertex_pos_proj = proj_ProjectionMatrix  * proj_ModelViewMatrix  * p3d_Vertex;
    vertex_normal = p3d_Normal;

    // Figure out the texcoord on the camera image
    vec2 proj_tex_pos = (vertex_pos_proj.xyz / vertex_pos_proj.w).xy;
    projection_texcoord = (proj_tex_pos + 1.0) / 2.0;

    // Figure out the positon on screen based on model texcoord
    model_texcoord = p3d_MultiTexCoord0;

    if (ShaderViewMode == ShaderViewMode_3D) {
        gl_Position = vertex_pos_3d;
    } else {
        vec2 tex = model_texcoord * 2.0 - 1.0;
        gl_Position = vec4(tex, 0.0, 1.0);
    }

    if (ShaderTextureMode == ShaderTextureMode_Projection) {
        // Necessary to get perspective-correct texturing
        gl_Position /= vertex_pos_proj.w;
    }
}