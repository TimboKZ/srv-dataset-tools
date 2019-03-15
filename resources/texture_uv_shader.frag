#version 130

uniform sampler2D ModelTexture;
uniform sampler2D ProjectionTexture;

// Input from vertex shader
in vec4 vertex_pos;
in vec2 model_texcoord;
in vec2 projection_texcoord;

// Output into the renderer
out vec4 frag_color;

void main() {
    vec4 model_color = texture(ModelTexture, model_texcoord);

//    vec2 projection_texcoord = vec2(vertex_pos);
    vec4 projection_color = texture(ProjectionTexture, projection_texcoord);

//    frag_color = model_color.rgba + projection_color.rgba;
    frag_color = projection_color.rgba;

}