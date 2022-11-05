#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform readonly image2D src;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D dst;
layout(set = 0, binding = 2) uniform Settings {
    float diffuseRate;
    float evaporationRate;
    float velocity;
    float turnSpeed;
    int sensorSize;
    float sensorAngleDegrees;
    float sensorDistance;
} settings;
layout(set = 0, binding = 3) uniform Data{
    int time;
    float deltaTime;
} timeBuf;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(src);

    vec4 originalValue = imageLoad(src, pos);

    vec4 sum = vec4(0.0);
    int n = 0;
    int offsetX, offsetY;
    for (offsetX = -1; offsetX <= 1; offsetX++) {
        for (offsetY = -1; offsetY <= 1; offsetY++) {
            int sampleX = pos.x + offsetX;
            int sampleY = pos.y + offsetY;

            if (sampleX >= 0 && sampleX < size.x && sampleY >= 0 && sampleY < size.y) {
                sum += imageLoad(src, ivec2(sampleX, sampleY));
                n++;
            }
        }
    }

    vec4 blurResult = sum / n;

    vec4 diffusedValue = mix(originalValue, blurResult, settings.diffuseRate * timeBuf.deltaTime);

    imageStore(dst, pos, vec4(max(diffusedValue.rgb - settings.evaporationRate * timeBuf.deltaTime, 0), 1.0));
}
