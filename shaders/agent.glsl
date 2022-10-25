#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct Agent {
    vec2 position;
    vec2 direction;
    // float velocity;
} ;

layout(set = 0, binding = 0) buffer Agents
{
    Agent agents[];
} buf;

layout(set = 0, binding = 1, rgba8) uniform writeonly image2D dst;
layout(set = 0, binding = 2, rgba8) uniform readonly image2D src;
layout(set = 0, binding = 3) uniform Data{
    int time;
} timeBuf;

float scaleToRange01(uint state)
{
    return state / 4294967295.0;
}

float hash(uint state)
{
    state ^= 2747636419u;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    return scaleToRange01(state);
}

float sense(Agent agent, float angle, int sensorSize, float sensorAngleOffset) {
    vec2 size = imageSize(dst);
    float sensorAngle = angle + sensorAngleOffset; 
    vec2 sensorDir = vec2(cos(sensorAngle), sin(sensorAngle));


	vec2 sensorPos = agent.position + sensorDir * 15;
	int sensorCentreX = int(sensorPos.x);
	int sensorCentreY = int(sensorPos.y);

    float sum = 0.0;
    for (int offsetX = -sensorSize; offsetX <= sensorSize; offsetX ++) {
		for (int offsetY = -sensorSize; offsetY <= sensorSize; offsetY ++) {
			int sampleX = int(min(size.x - 1, max(0, sensorCentreX + offsetX)));
			int sampleY = int(min(size.y - 1, max(0, sensorCentreY + offsetY)));
			sum += dot(vec4(1.0), imageLoad(src, ivec2(sampleX, sampleY)));
		}
	}

    return sum;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    vec2 size = imageSize(dst);

    float turnSpeed = 1;

    float randomSteer = hash(int(buf.agents[idx].position.y * size.x + buf.agents[idx].position.x + timeBuf.time));

    float angle = atan(buf.agents[idx].direction.y, buf.agents[idx].direction.x);
    float sensorAngleOffset = 30 * (3.1415 / 180);
    int sensorSize = 4;
    float weightForward = sense(buf.agents[idx], angle, sensorSize, 0);
    float weightLeft = sense(buf.agents[idx], angle, sensorSize, -sensorAngleOffset);
    float weightRight = sense(buf.agents[idx], angle, sensorSize, sensorAngleOffset);

    if (weightForward >= weightLeft && weightForward >= weightRight) {
        angle += (randomSteer - 0.5) * 2 * turnSpeed;

    } else if(weightLeft >= weightRight) {
        angle -= randomSteer * turnSpeed;
    } else {
        angle += randomSteer * turnSpeed;
    }

    buf.agents[idx].direction = vec2(cos(angle), sin(angle));

    vec2 newPos = buf.agents[idx].position + buf.agents[idx].direction;
    if (newPos.x < 0 || newPos.x >= size.x) {
        buf.agents[idx].direction.x *= -1;
        newPos.x = newPos.x < 0 ? 0 : size.x;
    }

    if (newPos.y < 0 || newPos.y >= size.y) {
        buf.agents[idx].direction.y *= -1;
        newPos.y = newPos.y < 0 ? 0 : size.y;
    }


    buf.agents[idx].position = newPos;
    imageStore(dst, ivec2(buf.agents[idx].position.xy), vec4(1.0));
}
