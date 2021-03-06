#version 130

#define ShaderViewMode_3D 0
#define ShaderViewMode_Texture 1
#define ShaderTextureMode_Default 0
#define ShaderTextureMode_Projection 1
#define ShaderTextureMode_Normal 2
#define ShaderTextureMode_Mask 3
#define ShaderTextureMode_Visibility 4
#define ShaderTextureMode_Frustum 5
#define ShaderTextureMode_Light 6

// Shader parameters
uniform int ShaderViewMode;
uniform int ShaderTextureMode;

// Uniform inputs
uniform sampler2D ModelTexture;
uniform sampler2D ProjectionTexture;

// Input from vertex shader
in vec4 vertex_pos_proj;
in vec4 vertex_pos_3d;
in vec3 vertex_normal;
in vec3 vertex_to_light;
in vec2 model_texcoord;
in vec2 projection_texcoord;

// Output into the renderer
out vec4 frag_color;

bool valid_texcoord(vec2 texcoord) {
    if (texcoord[0] < 0.0 || texcoord[0] > 1.0) {
        return false;
    }
    if (texcoord[1] < 0.0 || texcoord[1] > 1.0) {
        return false;
    }

    return true;
}

void main() {
    vec4 model_color = texture2D(ModelTexture, model_texcoord);

    //    vec2 projection_texcoord = vec2(vertex_pos);
    vec4 projection_color;
    vec4 projection_mask;
    if (valid_texcoord(projection_texcoord)) {
        projection_color = texture2D(ProjectionTexture, projection_texcoord);
        projection_mask = vec4(1, 1, 1, 1);
    } else {
        projection_color = vec4(0, 0, 0, 1);
        projection_mask = vec4(0, 0, 0, 1);
    }

    vec3 proj_frustum = (vertex_pos_proj.xyz / vertex_pos_proj.w).xyz;
    vec3 proj_frustum_0_to_1 = (proj_frustum + 1) / 2.0;

    if (ShaderTextureMode == ShaderTextureMode_Projection) {
        frag_color = projection_color.rgba;

    } else if (ShaderTextureMode == ShaderTextureMode_Normal) {
        vec3 normal_0_to_1 = (vertex_normal + 1) / 2.0;
        frag_color = vec4(normal_0_to_1, 1.0);

    } else if (ShaderTextureMode == ShaderTextureMode_Mask) {
        frag_color = vec4(1.0, 1.0, 1.0, 1.0);

    } else if (ShaderTextureMode == ShaderTextureMode_Visibility) {
        frag_color = projection_mask;

    } else if (ShaderTextureMode == ShaderTextureMode_Frustum) {
        if (valid_texcoord(proj_frustum_0_to_1.xy)) {
            frag_color = vec4(proj_frustum_0_to_1, 1.0);
        } else {
            frag_color = vec4(0);
        }

    } else if (ShaderTextureMode == ShaderTextureMode_Light) {
        vec3 to_light_0_to_1 = (vertex_to_light + 1) / 2.0;
        frag_color = vec4(to_light_0_to_1, 1.0);

    } else {
        // ShaderTextureMode == ShaderTextureMode_Default
        frag_color = model_color.rgba;
    }

}