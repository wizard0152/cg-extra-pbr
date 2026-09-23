cbuffer DrawConstants : register(b0)
{
    float4x4 gWorldViewProjection;
    float4x4 gWorld;
    float4 gUvTransform;
    float4 gMaterialColor;
};

Texture2D gDiffuseTexture : register(t0);
SamplerState gSampler : register(s0);

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
    float4 tangent : TANGENT;
};
struct GeometryInput { float4 position : SV_POSITION; float3 worldPosition : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD; };
struct GBufferOutput { float4 albedo : SV_Target0; float4 normal : SV_Target1; float4 worldPosition : SV_Target2; };

GeometryInput GeometryVS(VertexInput input)
{
    GeometryInput output;
    output.position = mul(float4(input.position, 1.0f), gWorldViewProjection);
    output.worldPosition = mul(float4(input.position, 1.0f), gWorld).xyz;
    output.normal = normalize(mul(float4(input.normal, 0.0f), gWorld).xyz);
    output.uv = input.uv * gUvTransform.xy + gUvTransform.zw;
    return output;
}

GBufferOutput GeometryPS(GeometryInput input)
{
    GBufferOutput output;
    float4 texel = gDiffuseTexture.Sample(gSampler, input.uv) * gMaterialColor;
    clip(texel.a - 0.15f);
    output.albedo = float4(texel.rgb, 0.0f);
    output.normal = float4(normalize(input.normal), 0.72f);
    output.worldPosition = float4(input.worldPosition, 1.0f);
    return output;
}

struct PointLight { float4 positionRadius; float4 colorIntensity; };
struct SpotLight { float4 positionRange; float4 directionCosOuter; float4 colorIntensity; float4 parameters; };

cbuffer LightConstants : register(b0)
{
    float4 gCameraPosition;
    float4 gAmbientColor;
    float4 gDirectionalDirectionIntensity;
    float4 gDirectionalColor;
    uint4 gLightCountsAndMode;
    PointLight gPointLights[16];
    SpotLight gSpotLights[8];
    float4x4 gShadowViewProjection[4];
    float4 gCascadeSplits;
    float4 gShadowParameters; // x=enabled, y=visualize cascades, z=bias, w=texel size
    float4 gCameraForward;
};

Texture2D<float4> gAlbedo : register(t0);
Texture2D<float4> gNormal : register(t1);
Texture2D<float4> gWorldPosition : register(t2);
Texture2DArray<float> gShadowMap : register(t3);
TextureCube<float4> gIrradianceMap : register(t4);
Texture2D<float2> gBrdfIntegrationMap : register(t5);
TextureCube<float4> gPrefilteredEnvironmentMap : register(t6);
SamplerComparisonState gShadowSampler : register(s1);
SamplerState gIblSampler : register(s2);

struct FullScreenInput { float4 position : SV_POSITION; float2 uv : TEXCOORD; };

FullScreenInput LightingVS(uint id : SV_VertexID)
{
    FullScreenInput output;
    output.uv = float2((id << 1) & 2, id & 2);
    output.position = float4(output.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}

static const float PI = 3.14159265359f;

float DistributionGGX(float3 normal, float3 halfway, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = saturate(dot(normal, halfway));
    float denominator = nDotH * nDotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * denominator * denominator, 0.000001f);
}

float GeometrySchlickGGX(float nDotDirection, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f;
    return nDotDirection / max(nDotDirection * (1.0f - k) + k, 0.000001f);
}

float GeometrySmith(float3 normal, float3 viewDirection, float3 lightDirection, float roughness)
{
    return GeometrySchlickGGX(saturate(dot(normal, viewDirection)), roughness) *
        GeometrySchlickGGX(saturate(dot(normal, lightDirection)), roughness);
}

float3 FresnelSchlick(float cosine, float3 f0)
{
    return f0 + (1.0f - f0) * pow(saturate(1.0f - cosine), 5.0f);
}

float3 FresnelSchlickRoughness(float cosine, float3 f0, float roughness)
{
    return f0 + (max(1.0f - roughness, f0) - f0) * pow(saturate(1.0f - cosine), 5.0f);
}

float3 EvaluatePbrLight(float3 albedo, float metallic, float roughness, float3 normal,
    float3 viewDirection, float3 lightDirection, float3 radiance)
{
    float3 halfway = normalize(viewDirection + lightDirection);
    float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 fresnel = FresnelSchlick(saturate(dot(halfway, viewDirection)), f0);
    float distribution = DistributionGGX(normal, halfway, roughness);
    float geometry = GeometrySmith(normal, viewDirection, lightDirection, roughness);
    float3 specular = distribution * geometry * fresnel /
        max(4.0f * saturate(dot(normal, viewDirection)) * saturate(dot(normal, lightDirection)), 0.001f);
    float3 diffuseWeight = (1.0f - fresnel) * (1.0f - metallic);
    return (diffuseWeight * albedo / PI + specular) * radiance * saturate(dot(normal, lightDirection));
}

uint SelectCascade(float3 worldPosition)
{
    float viewDepth = dot(worldPosition - gCameraPosition.xyz, normalize(gCameraForward.xyz));
    uint cascadeIndex = viewDepth > gCascadeSplits.x ? 1 : 0;
    cascadeIndex = viewDepth > gCascadeSplits.y ? 2 : cascadeIndex;
    cascadeIndex = viewDepth > gCascadeSplits.z ? 3 : cascadeIndex;
    return cascadeIndex;
}

float DirectionalShadow(float3 worldPosition, uint cascadeIndex)
{
    cascadeIndex = min(cascadeIndex, 3u);
    float result = 1.0f;
    float4 lightClip = mul(float4(worldPosition, 1.0f), gShadowViewProjection[cascadeIndex]);
    float3 projected = lightClip.xyz / lightClip.w;
    float2 uv = projected.xy * float2(0.5f, -0.5f) + 0.5f;
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f)
        return result;

    float visibility = 0.0f;
    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            visibility += gShadowMap.SampleCmpLevelZero(gShadowSampler,
                float3(uv, cascadeIndex), projected.z - gShadowParameters.z,
                int2(x, y));
        }
    }
    result = visibility / 9.0f;
    return result;
}

float GBufferOutline(int2 pixel)
{
    uint width, height;
    gNormal.GetDimensions(width, height);
    const int2 maximum = int2(width - 1, height - 1);
    const int2 offsets[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    const float4 centerNormal = gNormal.Load(int3(pixel, 0));
    const float4 centerPosition = gWorldPosition.Load(int3(pixel, 0));
    float edge = 0.0f;
    [unroll] for (uint i = 0; i < 4; ++i)
    {
        const int2 samplePixel = clamp(pixel + offsets[i] * 2, int2(0, 0), maximum);
        const float4 sampleNormal = gNormal.Load(int3(samplePixel, 0));
        const float4 samplePosition = gWorldPosition.Load(int3(samplePixel, 0));
        if (centerPosition.w != samplePosition.w)
            edge = 1.0f;
        else if (centerPosition.w > 0.0f)
        {
            const float normalDifference = length(centerNormal.xyz - sampleNormal.xyz);
            const float positionDifference = length(centerPosition.xyz - samplePosition.xyz);
            const float distanceScale = 1.0f + length(centerPosition.xyz - gCameraPosition.xyz) * 0.04f;
            edge = max(edge, saturate(normalDifference * 1.8f + positionDifference * 0.22f / distanceScale));
        }
    }
    return smoothstep(0.18f, 0.7f, edge);
}

float VignetteFactor(float2 uv)
{
    const float2 centered = uv * 2.0f - 1.0f;
    const float radialDistance = dot(centered, centered);
    return 1.0f - smoothstep(0.35f, 1.35f, radialDistance) * 0.68f;
}

float4 LightingPS(FullScreenInput input) : SV_Target
{
    int2 pixel = int2(input.position.xy);
    float4 albedo = gAlbedo.Load(int3(pixel, 0));
    float4 normalSample = gNormal.Load(int3(pixel, 0));
    float4 positionSample = gWorldPosition.Load(int3(pixel, 0));
    if (positionSample.w == 0.0f) return float4(0.018f, 0.025f, 0.045f, 1.0f);
    float3 normal = normalize(normalSample.xyz);
    float3 worldPosition = positionSample.xyz;
    float metallic = saturate(albedo.a);
    float roughness = clamp(normalSample.a, 0.05f, 1.0f);
    uint mode = gLightCountsAndMode.z;
    if (mode == 1) return float4(albedo.rgb, 1.0f);
    if (mode == 2) return float4(normal * 0.5f + 0.5f, 1.0f);
    if (mode == 3) return float4(frac(abs(worldPosition) * 0.05f), 1.0f);

    uint cascadeIndex = SelectCascade(worldPosition);
    float shadow = 1.0f;
    if (gShadowParameters.x > 0.5f)
        shadow = DirectionalShadow(worldPosition, cascadeIndex);
    float3 viewDirection = normalize(gCameraPosition.xyz - worldPosition);
    float3 result = 0.0f;
    const float3 directionalDirection = normalize(-gDirectionalDirectionIntensity.xyz);
    result += EvaluatePbrLight(albedo.rgb, metallic, roughness, normal, viewDirection,
        directionalDirection, gDirectionalColor.rgb * gDirectionalDirectionIntensity.w) * shadow;
    [loop] for (uint i = 0; i < gLightCountsAndMode.x; ++i) {
        float3 toLight = gPointLights[i].positionRadius.xyz - worldPosition;
        float distanceToLight = length(toLight);
        float attenuation = saturate(1.0f - distanceToLight / gPointLights[i].positionRadius.w);
        attenuation *= attenuation;
        result += EvaluatePbrLight(albedo.rgb, metallic, roughness, normal, viewDirection,
            toLight / max(distanceToLight, 0.0001f),
            gPointLights[i].colorIntensity.rgb * gPointLights[i].colorIntensity.w * attenuation);
    }
    [loop] for (uint j = 0; j < gLightCountsAndMode.y; ++j) {
        float3 toLight = gSpotLights[j].positionRange.xyz - worldPosition;
        float distanceToLight = length(toLight);
        float3 directionToLight = toLight / max(distanceToLight, 0.0001f);
        float cone = dot(-directionToLight, normalize(gSpotLights[j].directionCosOuter.xyz));
        float coneFactor = smoothstep(gSpotLights[j].directionCosOuter.w, gSpotLights[j].parameters.x, cone);
        float attenuation = saturate(1.0f - distanceToLight / gSpotLights[j].positionRange.w);
        attenuation *= attenuation * coneFactor;
        result += EvaluatePbrLight(albedo.rgb, metallic, roughness, normal, viewDirection,
            directionToLight, gSpotLights[j].colorIntensity.rgb *
            gSpotLights[j].colorIntensity.w * attenuation);
    }
    if ((gLightCountsAndMode.w & 4u) != 0u)
    {
        float nDotV = saturate(dot(normal, viewDirection));
        float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo.rgb, metallic);
        float3 fresnel = FresnelSchlickRoughness(nDotV, f0, roughness);
        float3 diffuseWeight = (1.0f - fresnel) * (1.0f - metallic);
        float3 irradiance = gIrradianceMap.SampleLevel(gIblSampler, normal, 0.0f).rgb;
        float3 diffuseIbl = irradiance * albedo.rgb;
        float3 reflection = reflect(-viewDirection, normal);
        float3 prefiltered = gPrefilteredEnvironmentMap.SampleLevel(
            gIblSampler, reflection, roughness * 11.0f).rgb;
        float2 brdf = gBrdfIntegrationMap.SampleLevel(gIblSampler,
            float2(nDotV, roughness), 0.0f).rg;
        float3 specularIbl = prefiltered * (fresnel * brdf.x + brdf.y);
        result += diffuseWeight * diffuseIbl + specularIbl;
    }
    else
    {
        result += gAmbientColor.rgb * albedo.rgb;
    }
    if (gShadowParameters.y > 0.5f) {
        const float3 cascadeColors[4] = {
            float3(1.0f, 0.35f, 0.35f), float3(0.35f, 1.0f, 0.35f),
            float3(0.35f, 0.55f, 1.0f), float3(1.0f, 0.85f, 0.3f)};
        result = lerp(result, result * cascadeColors[cascadeIndex], 0.35f);
    }
    // Post-processing flags are stored in gLightCountsAndMode.w:
    // bit 0 = camera vignette, bit 1 = G-buffer outline.
    if ((gLightCountsAndMode.w & 2u) != 0u)
        result = lerp(result, float3(0.008f, 0.012f, 0.02f), GBufferOutline(pixel) * 0.88f);
    if ((gLightCountsAndMode.w & 1u) != 0u)
        result *= VignetteFactor(input.uv);
    result = result / (result + 1.0f);
    result = pow(max(result, 0.0f), 1.0f / 2.2f);
    return float4(result, 1.0f);
}

cbuffer TessellationConstants : register(b0)
{
    float4x4 gTessWorldViewProjection;
    float4x4 gTessWorld;
    float4 gTessEyePosition;
    float4 gTessParameters; // x=max factor, y=near distance, z=far distance, w=height scale
    float4 gTessOptions;    // x=normal map enabled, y=displacement enabled
};

Texture2D gTessAlbedo : register(t0);
Texture2D gTessNormal : register(t1);
Texture2D gTessDisplacement : register(t2);

struct TessControlPoint
{
    float3 worldPosition : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD;
};

TessControlPoint TessellationVS(VertexInput input)
{
    TessControlPoint output;
    output.worldPosition = mul(float4(input.position, 1.0f), gTessWorld).xyz;
    output.normal = normalize(mul(float4(input.normal, 0.0f), gTessWorld).xyz);
    output.tangent = float4(normalize(mul(float4(input.tangent.xyz, 0.0f), gTessWorld).xyz), input.tangent.w);
    output.uv = input.uv;
    return output;
}

struct TessFactors
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

float DistanceTessellationFactor(float3 position)
{
    float distanceFromCamera = distance(position, gTessEyePosition.xyz);
    float blend = saturate((distanceFromCamera - gTessParameters.y) /
        max(gTessParameters.z - gTessParameters.y, 0.001f));
    return lerp(gTessParameters.x, 1.0f, blend);
}

TessFactors TessellationPatchConstants(InputPatch<TessControlPoint, 3> patch)
{
    TessFactors output;
    output.edges[0] = DistanceTessellationFactor((patch[1].worldPosition + patch[2].worldPosition) * 0.5f);
    output.edges[1] = DistanceTessellationFactor((patch[2].worldPosition + patch[0].worldPosition) * 0.5f);
    output.edges[2] = DistanceTessellationFactor((patch[0].worldPosition + patch[1].worldPosition) * 0.5f);
    output.inside = DistanceTessellationFactor(
        (patch[0].worldPosition + patch[1].worldPosition + patch[2].worldPosition) / 3.0f);
    return output;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("TessellationPatchConstants")]
[maxtessfactor(16.0f)]
TessControlPoint TessellationHS(InputPatch<TessControlPoint, 3> patch, uint id : SV_OutputControlPointID)
{
    return patch[id];
}

struct TessPixelInput
{
    float4 position : SV_POSITION;
    float3 worldPosition : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
    float2 uv : TEXCOORD;
};

[domain("tri")]
TessPixelInput TessellationDS(TessFactors factors, float3 barycentric : SV_DomainLocation,
    const OutputPatch<TessControlPoint, 3> patch)
{
    TessPixelInput output;
    output.worldPosition = barycentric.x * patch[0].worldPosition +
        barycentric.y * patch[1].worldPosition + barycentric.z * patch[2].worldPosition;
    output.normal = normalize(barycentric.x * patch[0].normal +
        barycentric.y * patch[1].normal + barycentric.z * patch[2].normal);
    float4 tangent = barycentric.x * patch[0].tangent +
        barycentric.y * patch[1].tangent + barycentric.z * patch[2].tangent;
    output.tangent = float4(normalize(tangent.xyz), tangent.w < 0.0f ? -1.0f : 1.0f);
    output.uv = barycentric.x * patch[0].uv + barycentric.y * patch[1].uv + barycentric.z * patch[2].uv;
    float height = gTessDisplacement.SampleLevel(gSampler, output.uv, 0).r;
    output.worldPosition += output.normal *
        ((height - 0.5f) * gTessParameters.w * gTessOptions.y);
    output.position = mul(float4(output.worldPosition, 1.0f), gTessWorldViewProjection);
    return output;
}

GBufferOutput TessellationPS(TessPixelInput input)
{
    GBufferOutput output;
    float3 N = normalize(input.normal);
    float3 T = normalize(input.tangent.xyz - dot(input.tangent.xyz, N) * N);
    float3 B = cross(N, T) * input.tangent.w;
    float3 tangentNormal = float3(0.0f, 0.0f, 1.0f);
    if (gTessOptions.x > 0.5f) {
        tangentNormal = gTessNormal.Sample(gSampler, input.uv).xyz * 2.0f - 1.0f;
        tangentNormal.y = -tangentNormal.y;
    }
    float3 worldNormal = normalize(mul(tangentNormal, float3x3(T, B, N)));
    output.albedo = float4(gTessAlbedo.Sample(gSampler, input.uv).rgb, 0.0f);
    output.normal = float4(worldNormal, 0.48f);
    output.worldPosition = float4(input.worldPosition, 1.0f);
    return output;
}

cbuffer InstancePassConstants : register(b0)
{
    float4x4 gInstanceViewProjection;
};

struct InstancedVertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
    float4 tangent : TANGENT;
    float3 instancePosition : INSTANCE_POSITION;
    float3 instanceScale : INSTANCE_SCALE;
    float4 instanceColor : INSTANCE_COLOR;
    float instanceRoughness : INSTANCE_ROUGHNESS;
};

struct InstancedPixelInput
{
    float4 position : SV_POSITION;
    float3 worldPosition : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR;
    float roughness : ROUGHNESS;
};

InstancedPixelInput InstancingVS(InstancedVertexInput input)
{
    InstancedPixelInput output;
    output.worldPosition = input.position * input.instanceScale + input.instancePosition;
    output.position = mul(float4(output.worldPosition, 1.0f), gInstanceViewProjection);
    output.normal = input.normal;
    output.color = input.instanceColor;
    output.roughness = input.instanceRoughness;
    return output;
}

GBufferOutput InstancingPS(InstancedPixelInput input)
{
    GBufferOutput output;
    output.albedo = input.color;
    output.normal = float4(normalize(input.normal), input.roughness);
    output.worldPosition = float4(input.worldPosition, 1.0f);
    return output;
}

cbuffer ShadowPassConstants : register(b0)
{
    float4x4 gLightViewProjection;
};

float4 ShadowVS(InstancedVertexInput input) : SV_POSITION
{
    float3 worldPosition = input.position * input.instanceScale + input.instancePosition;
    return mul(float4(worldPosition, 1.0f), gLightViewProjection);
}

// GPU particle system. The compute pass consumes every particle from one
// structured buffer and appends the updated particle to the other buffer.
struct Particle
{
    float3 position;
    float age;
    float3 velocity;
    float lifetime;
    float4 color;
    float size;
    float3 padding;
};

cbuffer ParticleConstants : register(b0)
{
    float4x4 gParticleViewProjection;
    float4 gParticleCameraRight;
    float4 gParticleCameraUp;
    float4 gParticleEmitterAndTime; // xyz = emitter, w = delta time
};

ConsumeStructuredBuffer<Particle> gParticleInput : register(u0);
AppendStructuredBuffer<Particle> gParticleOutput : register(u1);

float ParticleRandom(uint value)
{
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return (value & 0x00ffffffu) / 16777215.0f;
}

[numthreads(64, 1, 1)]
void ParticleCS(uint3 dispatchId : SV_DispatchThreadID)
{
    Particle particle = gParticleInput.Consume();
    const float dt = gParticleEmitterAndTime.w;
    particle.age += dt;
    particle.velocity += float3(0.0f, -5.2f, 0.0f) * dt;
    particle.position += particle.velocity * dt;

    if (particle.age >= particle.lifetime || particle.position.y < 0.05f)
    {
        uint seed = dispatchId.x * 747796405u + asuint(particle.age * 1000.0f) + 2891336453u;
        float angle = ParticleRandom(seed) * 6.2831853f;
        float radius = sqrt(ParticleRandom(seed ^ 0x68bc21ebu)) * 1.25f;
        float speed = 4.5f + ParticleRandom(seed ^ 0x02e5be93u) * 5.5f;
        particle.position = gParticleEmitterAndTime.xyz + float3(cos(angle) * radius, 0.0f, sin(angle) * radius);
        particle.velocity = float3(cos(angle) * speed * 0.22f, speed, sin(angle) * speed * 0.22f);
        particle.age = 0.0f;
        particle.lifetime = 1.6f + ParticleRandom(seed ^ 0x967a889bu) * 1.8f;
        particle.size = 0.12f + ParticleRandom(seed ^ 0x368cc8b7u) * 0.18f;
        particle.color = lerp(float4(1.0f, 0.22f, 0.04f, 1.0f),
            float4(1.0f, 0.9f, 0.18f, 1.0f), ParticleRandom(seed ^ 0xa511e9b3u));
    }
    gParticleOutput.Append(particle);
}

StructuredBuffer<Particle> gParticles : register(t0);

struct ParticleVertexOutput
{
    float3 position : POSITION;
    float4 color : COLOR;
    float size : PSIZE;
};

ParticleVertexOutput ParticleVS(uint id : SV_VertexID)
{
    ParticleVertexOutput output;
    Particle particle = gParticles[id];
    output.position = particle.position;
    output.color = particle.color;
    output.size = particle.size;
    return output;
}

struct ParticlePixelInput
{
    float4 position : SV_POSITION;
    float3 worldPosition : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR;
};

[maxvertexcount(4)]
void ParticleGS(point ParticleVertexOutput input[1], inout TriangleStream<ParticlePixelInput> stream)
{
    const float3 right = gParticleCameraRight.xyz * input[0].size;
    const float3 up = gParticleCameraUp.xyz * input[0].size;
    const float3 normal = normalize(cross(gParticleCameraRight.xyz, gParticleCameraUp.xyz));
    const float2 corners[4] = {float2(-1,-1), float2(-1,1), float2(1,-1), float2(1,1)};
    [unroll] for (uint i = 0; i < 4; ++i)
    {
        ParticlePixelInput output;
        output.worldPosition = input[0].position + right * corners[i].x + up * corners[i].y;
        output.position = mul(float4(output.worldPosition, 1.0f), gParticleViewProjection);
        output.normal = normal;
        output.color = input[0].color;
        stream.Append(output);
    }
}

GBufferOutput ParticlePS(ParticlePixelInput input)
{
    GBufferOutput output;
    output.albedo = float4(input.color.rgb, 0.15f);
    output.normal = float4(normalize(input.normal), 0.28f);
    output.worldPosition = float4(input.worldPosition, 1.0f);
    return output;
}
