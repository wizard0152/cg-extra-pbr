#include <windows.h>
#include <wincodec.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <DirectXCollision.h>
#include "GBuffer.h"
#include "SpatialIndex.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
namespace fs = std::filesystem;

static void Check(HRESULT hr, const char* message)
{
    if (FAILED(hr)) throw std::runtime_error(message);
}

static D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
{
    D3D12_HEAP_PROPERTIES p{};
    p.Type = type;
    p.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    p.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    p.CreationNodeMask = 1;
    p.VisibleNodeMask = 1;
    return p;
}

static D3D12_RESOURCE_DESC BufferDescription(UINT64 size)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return b;
}

struct CascadeMatrices
{
    std::array<XMMATRIX, 4> viewProjection;
    std::array<float, 4> splitDistances{};
};

static CascadeMatrices CalculateCascades(FXMVECTOR cameraPosition, FXMVECTOR cameraForward,
    float verticalFov, float aspectRatio, float nearPlane, float farPlane)
{
    CascadeMatrices result{};
    constexpr float splitLambda = 0.72f;
    for (UINT cascade = 0; cascade < 4; ++cascade) {
        const float fraction = static_cast<float>(cascade + 1) / 4.0f;
        const float logarithmic = nearPlane * std::pow(farPlane / nearPlane, fraction);
        const float uniform = nearPlane + (farPlane - nearPlane) * fraction;
        result.splitDistances[cascade] = splitLambda * logarithmic + (1.0f - splitLambda) * uniform;
    }

    const XMVECTOR forward = XMVector3Normalize(cameraForward);
    const XMVECTOR worldUp = XMVectorSet(0, 1, 0, 0);
    const XMVECTOR right = XMVector3Normalize(XMVector3Cross(worldUp, forward));
    const XMVECTOR up = XMVector3Normalize(XMVector3Cross(forward, right));
    const XMVECTOR lightDirection = XMVector3Normalize(XMVectorSet(0.3f, -0.85f, 0.25f, 0));
    const float tangent = std::tan(verticalFov * 0.5f);
    float cascadeNear = nearPlane;

    for (UINT cascade = 0; cascade < 4; ++cascade) {
        const float cascadeFar = result.splitDistances[cascade];
        std::array<XMVECTOR, 8> corners;
        for (UINT plane = 0; plane < 2; ++plane) {
            const float distance = plane == 0 ? cascadeNear : cascadeFar;
            const float halfHeight = tangent * distance;
            const float halfWidth = halfHeight * aspectRatio;
            const XMVECTOR center = cameraPosition + forward * distance;
            corners[plane * 4 + 0] = center - right * halfWidth - up * halfHeight;
            corners[plane * 4 + 1] = center + right * halfWidth - up * halfHeight;
            corners[plane * 4 + 2] = center + right * halfWidth + up * halfHeight;
            corners[plane * 4 + 3] = center - right * halfWidth + up * halfHeight;
        }
        XMVECTOR center = XMVectorZero();
        for (FXMVECTOR corner : corners) center += corner;
        center /= 8.0f;
        const XMVECTOR lightPosition = center - lightDirection * 180.0f;
        const XMMATRIX lightView = XMMatrixLookAtLH(lightPosition, center, worldUp);
        XMFLOAT3 minimum{FLT_MAX, FLT_MAX, FLT_MAX};
        XMFLOAT3 maximum{-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (FXMVECTOR corner : corners) {
            XMFLOAT3 p;
            XMStoreFloat3(&p, XMVector3TransformCoord(corner, lightView));
            minimum.x = std::min(minimum.x, p.x); minimum.y = std::min(minimum.y, p.y); minimum.z = std::min(minimum.z, p.z);
            maximum.x = std::max(maximum.x, p.x); maximum.y = std::max(maximum.y, p.y); maximum.z = std::max(maximum.z, p.z);
        }
        const float padding = 4.0f;
        const XMMATRIX lightProjection = XMMatrixOrthographicOffCenterLH(
            minimum.x - padding, maximum.x + padding, minimum.y - padding, maximum.y + padding,
            std::max(0.1f, minimum.z - 80.0f), maximum.z + 80.0f);
        result.viewProjection[cascade] = lightView * lightProjection;
        cascadeNear = cascadeFar;
    }
    return result;
}

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
    XMFLOAT2 uv;
    XMFLOAT4 tangent{1, 0, 0, 1};
};

struct Material
{
    std::string name;
    XMFLOAT4 color{1, 1, 1, 1};
    fs::path texturePath;
    UINT srvIndex = 0;
};

struct Submesh
{
    UINT firstIndex = 0;
    UINT indexCount = 0;
    UINT materialIndex = 0;
};

struct Model
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Material> materials;
    std::vector<Submesh> submeshes;
};

static std::vector<Material> LoadMtl(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open MTL file");
    std::vector<Material> result;
    Material* current = nullptr;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::string command;
        stream >> command;
        if (command == "newmtl") {
            result.emplace_back();
            current = &result.back();
            stream >> current->name;
        } else if (command == "Kd" && current) {
            stream >> current->color.x >> current->color.y >> current->color.z;
        } else if (command == "map_Kd" && current) {
            std::string filename;
            std::getline(stream >> std::ws, filename);
            current->texturePath = path.parent_path() / fs::u8path(filename);
        }
    }
    return result;
}

struct ObjIndex { int p = 0, t = 0, n = 0; };

static ObjIndex ParseObjIndex(const std::string& token)
{
    ObjIndex result{};
    std::stringstream s(token);
    std::string part;
    if (std::getline(s, part, '/') && !part.empty()) result.p = std::stoi(part);
    if (std::getline(s, part, '/') && !part.empty()) result.t = std::stoi(part);
    if (std::getline(s, part, '/') && !part.empty()) result.n = std::stoi(part);
    return result;
}

static int ResolveObjIndex(int index, size_t count)
{
    if (index > 0) return index - 1;
    if (index < 0) return static_cast<int>(count) + index;
    return -1;
}

static Model LoadObj(const fs::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open OBJ file");
    std::vector<XMFLOAT3> positions;
    std::vector<XMFLOAT3> normals;
    std::vector<XMFLOAT2> texcoords;
    Model model;
    std::unordered_map<std::string, UINT> materialByName;
    UINT currentMaterial = 0;

    auto finishSubmesh = [&] {
        if (!model.submeshes.empty()) {
            auto& s = model.submeshes.back();
            s.indexCount = static_cast<UINT>(model.indices.size()) - s.firstIndex;
            if (s.indexCount == 0) model.submeshes.pop_back();
        }
    };

    std::string line;
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::string command;
        stream >> command;
        if (command == "v") {
            XMFLOAT3 v{}; stream >> v.x >> v.y >> v.z; positions.push_back(v);
        } else if (command == "vt") {
            XMFLOAT2 v{}; stream >> v.x >> v.y; texcoords.push_back(v);
        } else if (command == "vn") {
            XMFLOAT3 v{}; stream >> v.x >> v.y >> v.z; normals.push_back(v);
        } else if (command == "mtllib") {
            std::string filename; std::getline(stream >> std::ws, filename);
            model.materials = LoadMtl(path.parent_path() / fs::u8path(filename));
            for (UINT i = 0; i < model.materials.size(); ++i) materialByName[model.materials[i].name] = i;
        } else if (command == "usemtl") {
            finishSubmesh();
            std::string name; stream >> name;
            auto found = materialByName.find(name);
            currentMaterial = found == materialByName.end() ? 0 : found->second;
            model.submeshes.push_back({static_cast<UINT>(model.indices.size()), 0, currentMaterial});
        } else if (command == "f") {
            std::vector<ObjIndex> face;
            std::string token;
            while (stream >> token) face.push_back(ParseObjIndex(token));
            if (face.size() < 3) continue;
            if (model.submeshes.empty()) model.submeshes.push_back({0, 0, currentMaterial});
            for (size_t tri = 1; tri + 1 < face.size(); ++tri) {
                const ObjIndex corners[3] = {face[0], face[tri], face[tri + 1]};
                for (const auto& c : corners) {
                    Vertex v{};
                    int pi = ResolveObjIndex(c.p, positions.size());
                    int ti = ResolveObjIndex(c.t, texcoords.size());
                    int ni = ResolveObjIndex(c.n, normals.size());
                    if (pi >= 0) v.position = positions[pi];
                    v.normal = ni >= 0 ? normals[ni] : XMFLOAT3{0, 0, -1};
                    v.uv = ti >= 0 ? texcoords[ti] : XMFLOAT2{0, 0};
                    model.vertices.push_back(v);
                    model.indices.push_back(static_cast<uint32_t>(model.vertices.size() - 1));
                }
            }
        }
    }
    finishSubmesh();
    if (model.materials.empty()) model.materials.push_back({"Default"});
    return model;
}

struct ImageData
{
    UINT width = 0;
    UINT height = 0;
    std::vector<uint8_t> pixels;
};

static ImageData LoadImageWic(const fs::path& path)
{
    ComPtr<IWICImagingFactory> factory;
    Check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory)), "Cannot create WIC factory");
    ComPtr<IWICBitmapDecoder> decoder;
    Check(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
        WICDecodeMetadataCacheOnLoad, &decoder), "Cannot decode texture");
    ComPtr<IWICBitmapFrameDecode> frame;
    Check(decoder->GetFrame(0, &frame), "Cannot read texture frame");
    ImageData image;
    Check(frame->GetSize(&image.width, &image.height), "Cannot read texture dimensions");
    ComPtr<IWICFormatConverter> converter;
    Check(factory->CreateFormatConverter(&converter), "Cannot create WIC converter");
    Check(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom), "Cannot convert texture");
    image.pixels.resize(static_cast<size_t>(image.width) * image.height * 4);
    Check(converter->CopyPixels(nullptr, image.width * 4,
        static_cast<UINT>(image.pixels.size()), image.pixels.data()), "Cannot copy texture pixels");
    return image;
}

static ImageData MakeFallbackImage()
{
    ImageData image;
    image.width = image.height = 2;
    image.pixels = {
        255, 0, 255, 255, 35, 35, 35, 255,
        35, 35, 35, 255, 255, 0, 255, 255};
    return image;
}

struct alignas(256) DrawConstants
{
    XMFLOAT4X4 worldViewProjection;
    XMFLOAT4X4 world;
    XMFLOAT4 uvTransform;
    XMFLOAT4 materialColor;
};

struct PointLightData { XMFLOAT4 positionRadius; XMFLOAT4 colorIntensity; };
struct SpotLightData
{
    XMFLOAT4 positionRange;
    XMFLOAT4 directionCosOuter;
    XMFLOAT4 colorIntensity;
    XMFLOAT4 parameters;
};

struct alignas(256) LightConstants
{
    XMFLOAT4 cameraPosition;
    XMFLOAT4 ambientColor;
    XMFLOAT4 directionalDirectionIntensity;
    XMFLOAT4 directionalColor;
    std::array<UINT, 4> lightCountsAndMode{};
    std::array<PointLightData, 16> pointLights{};
    std::array<SpotLightData, 8> spotLights{};
    std::array<XMFLOAT4X4, 4> shadowViewProjection{};
    XMFLOAT4 cascadeSplits{};
    XMFLOAT4 shadowParameters{};
    XMFLOAT4 cameraForward{};
};

struct alignas(256) TessellationConstants
{
    XMFLOAT4X4 worldViewProjection;
    XMFLOAT4X4 world;
    XMFLOAT4 eyePosition;
    XMFLOAT4 parameters;
    XMFLOAT4 options;
};

struct InstanceData
{
    XMFLOAT3 position;
    float padding0 = 0.0f;
    XMFLOAT3 scale;
    float padding1 = 0.0f;
    XMFLOAT4 color;
    float roughness = 0.34f;
    XMFLOAT3 padding2{};
};

struct alignas(256) InstancePassConstants
{
    XMFLOAT4X4 viewProjection;
};

struct alignas(256) ShadowPassConstants
{
    XMFLOAT4X4 lightViewProjection;
};

struct ParticleData
{
    XMFLOAT3 position;
    float age;
    XMFLOAT3 velocity;
    float lifetime;
    XMFLOAT4 color;
    float size;
    XMFLOAT3 padding{};
};

struct alignas(256) ParticleConstants
{
    XMFLOAT4X4 viewProjection;
    XMFLOAT4 cameraRight;
    XMFLOAT4 cameraUp;
    XMFLOAT4 emitterAndTime;
};

class RenderingSystem
{
public:
    int Run(HINSTANCE instance);

private:
    static constexpr UINT FrameCount = 2;
    static constexpr UINT CascadeCount = 4;
    static constexpr UINT ShadowMapSize = 2048;
    static constexpr UINT ParticleCount = 4096;
    HWND window_ = nullptr;
    UINT width_ = 1280;
    UINT height_ = 720;
    bool paused_ = false;
    bool showTessellation_ = false;
    bool showSponza_ = false;
    bool tessellationWireframe_ = false;
    bool useNormalMap_ = true;
    bool useDisplacement_ = true;
    bool frustumCulling_ = true;
    bool octreeCulling_ = true;
    bool shadowsEnabled_ = true;
    bool visualizeCascades_ = false;
    bool particlesEnabled_ = false;
    bool vignetteEnabled_ = true;
    bool outlinesEnabled_ = true;
    bool iblEnabled_ = true;
    bool materialGridEnabled_ = true;
    float tiling_ = 2.0f;
    float animationTime_ = 0.0f;
    XMFLOAT3 cameraPosition_{0.0f, 8.0f, -25.0f};
    float cameraYaw_ = 0.0f;
    float cameraPitch_ = 0.0f;
    std::chrono::steady_clock::time_point previousTime_{};
    float titleUpdateTime_ = 0.0f;
    uint32_t titleFrameCount_ = 0;
    float displayedFps_ = 0.0f;

    ComPtr<ID3D12Device> device_;
    ComPtr<IDXGISwapChain3> swapChain_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12GraphicsCommandList> commandList_;
    std::array<ComPtr<ID3D12CommandAllocator>, FrameCount> allocators_;
    std::array<ComPtr<ID3D12Resource>, FrameCount> backBuffers_;
    ComPtr<ID3D12DescriptorHeap> rtvHeap_, dsvHeap_, srvHeap_;
    ComPtr<ID3D12Resource> depthBuffer_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fenceEvent_ = nullptr;
    std::array<UINT64, FrameCount> fenceValues_{};
    UINT frameIndex_ = 0;
    UINT rtvSize_ = 0;
    UINT dsvSize_ = 0;
    UINT srvSize_ = 0;

    ComPtr<ID3D12RootSignature> geometryRootSignature_, lightingRootSignature_;
    ComPtr<ID3D12PipelineState> geometryPipeline_, lightingPipeline_;
    ComPtr<ID3D12RootSignature> tessellationRootSignature_;
    ComPtr<ID3D12PipelineState> tessellationPipeline_, tessellationWireframePipeline_;
    ComPtr<ID3D12RootSignature> instancingRootSignature_;
    ComPtr<ID3D12PipelineState> instancingPipeline_;
    ComPtr<ID3D12RootSignature> shadowRootSignature_;
    ComPtr<ID3D12PipelineState> shadowPipeline_;
    ComPtr<ID3D12RootSignature> particleComputeRootSignature_, particleRenderRootSignature_;
    ComPtr<ID3D12PipelineState> particleComputePipeline_, particleRenderPipeline_;
    GBuffer gBuffer_;
    ComPtr<ID3D12Resource> vertexBuffer_, indexBuffer_;
    ComPtr<ID3D12Resource> tessellationVertexBuffer_;
    ComPtr<ID3D12Resource> cubeVertexBuffer_, cubeIndexBuffer_;
    ComPtr<ID3D12Resource> sphereVertexBuffer_, sphereIndexBuffer_;
    ComPtr<ID3D12Resource> shadowMap_, shadowInstanceBuffer_;
    std::array<ComPtr<ID3D12Resource>, 2> particleBuffers_, particleCounters_;
    D3D12_VERTEX_BUFFER_VIEW vertexView_{};
    D3D12_INDEX_BUFFER_VIEW indexView_{};
    D3D12_VERTEX_BUFFER_VIEW tessellationVertexView_{};
    D3D12_VERTEX_BUFFER_VIEW cubeVertexView_{};
    D3D12_INDEX_BUFFER_VIEW cubeIndexView_{};
    D3D12_VERTEX_BUFFER_VIEW sphereVertexView_{};
    D3D12_INDEX_BUFFER_VIEW sphereIndexView_{};
    UINT sphereIndexCount_ = 0;
    D3D12_VERTEX_BUFFER_VIEW shadowInstanceView_{};
    std::vector<ComPtr<ID3D12Resource>> textures_;
    std::vector<ComPtr<ID3D12Resource>> initializationUploads_;
    Model model_;
    std::array<ComPtr<ID3D12Resource>, FrameCount> constantBuffers_;
    std::array<uint8_t*, FrameCount> mappedConstants_{};
    std::array<ComPtr<ID3D12Resource>, FrameCount> lightBuffers_;
    std::array<uint8_t*, FrameCount> mappedLights_{};
    std::array<ComPtr<ID3D12Resource>, FrameCount> tessellationBuffers_;
    std::array<uint8_t*, FrameCount> mappedTessellation_{};
    std::array<ComPtr<ID3D12Resource>, FrameCount> instanceBuffers_;
    std::array<uint8_t*, FrameCount> mappedInstances_{};
    std::array<ComPtr<ID3D12Resource>, FrameCount> instancePassBuffers_;
    std::array<uint8_t*, FrameCount> mappedInstancePass_{};
    std::array<ComPtr<ID3D12Resource>, FrameCount> shadowPassBuffers_;
    std::array<uint8_t*, FrameCount> mappedShadowPass_{};
    std::array<ComPtr<ID3D12Resource>, FrameCount> particleConstantBuffers_;
    std::array<uint8_t*, FrameCount> mappedParticleConstants_{};
    std::vector<SceneObject> sceneObjects_;
    std::vector<uint32_t> visibleObjects_;
    Octree octree_;
    uint32_t cullingTests_ = 0;
    UINT gBufferSrvStart_ = 0;
    UINT tessellationSrvStart_ = 0;
    UINT shadowSrvIndex_ = 0;
    UINT particleUavStart_ = 0;
    UINT particleSrvStart_ = 0;
    UINT iblSrvStart_ = 0;
    UINT particleSource_ = 0;
    UINT displayMode_ = 0;

    void CreateWindowHandle(HINSTANCE instance);
    void CreateDeviceObjects();
    void CreateSizeDependentObjects();
    void CreatePipeline();
    void LoadScene();
    void Render();
    void Flush();
    void MoveToNextFrame();
    ComPtr<ID3D12Resource> CreateDefaultBuffer(const void* data, UINT64 size);
    ComPtr<ID3D12Resource> CreateTexture(const ImageData& image, UINT srvIndex);
    ComPtr<ID3D12Resource> CreateDdsTexture(const fs::path& path, UINT srvIndex);
    static LRESULT CALLBACK WindowProcedure(HWND, UINT, WPARAM, LPARAM);
};

LRESULT CALLBACK RenderingSystem::WindowProcedure(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    auto* app = reinterpret_cast<RenderingSystem*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    if (message == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
        app = static_cast<RenderingSystem*>(cs->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
    }
    if (app && message == WM_KEYDOWN) {
        if (wParam == VK_ESCAPE) DestroyWindow(hwnd);
        const bool firstPress = (lParam & (1LL << 30)) == 0;
        if (firstPress && wParam == VK_SPACE) app->paused_ = !app->paused_;
        if (firstPress && wParam == 'R') {
            app->cameraPosition_ = {0.0f, 8.0f, -25.0f};
            app->cameraYaw_ = 0.0f;
            app->cameraPitch_ = app->materialGridEnabled_ ? 0.0f : -0.12f;
        }
        if (firstPress && wParam == 'C') app->frustumCulling_ = !app->frustumCulling_;
        if (firstPress && wParam == 'O') app->octreeCulling_ = !app->octreeCulling_;
        if (firstPress && wParam == 'B') app->showSponza_ = !app->showSponza_;
        if (firstPress && wParam == 'T') app->showTessellation_ = !app->showTessellation_;
        if (firstPress && wParam == 'F') app->tessellationWireframe_ = !app->tessellationWireframe_;
        if (firstPress && wParam == 'N') app->useNormalMap_ = !app->useNormalMap_;
        if (firstPress && wParam == 'H') app->useDisplacement_ = !app->useDisplacement_;
        if (firstPress && wParam == 'X') app->shadowsEnabled_ = !app->shadowsEnabled_;
        if (firstPress && wParam == 'V') app->visualizeCascades_ = !app->visualizeCascades_;
        if (firstPress && wParam == 'P') app->particlesEnabled_ = !app->particlesEnabled_;
        if (firstPress && wParam == 'J') app->vignetteEnabled_ = !app->vignetteEnabled_;
        if (firstPress && wParam == 'K') app->outlinesEnabled_ = !app->outlinesEnabled_;
        if (firstPress && wParam == 'I') app->iblEnabled_ = !app->iblEnabled_;
        if (firstPress && wParam == 'G') {
            app->materialGridEnabled_ = !app->materialGridEnabled_;
            app->cameraPosition_ = {0.0f, 8.0f, -25.0f};
            app->cameraYaw_ = 0.0f;
            app->cameraPitch_ = app->materialGridEnabled_ ? 0.0f : -0.12f;
        }
        if (wParam >= '1' && wParam <= '4') app->displayMode_ = static_cast<UINT>(wParam - '1');
        return 0;
    }
    if (message == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd, message, wParam, lParam);
}

void RenderingSystem::CreateWindowHandle(HINSTANCE instance)
{
    WNDCLASSEX wc{sizeof(WNDCLASSEX)};
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProcedure;
    wc.hInstance = instance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"PbrIblHomeworkWindow";
    RegisterClassEx(&wc);
    RECT rect{0, 0, static_cast<LONG>(width_), static_cast<LONG>(height_)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    window_ = CreateWindowEx(0, wc.lpszClassName, L"Direct3D 12: Physically Based Rendering and IBL",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
        nullptr, nullptr, instance, this);
    if (!window_) throw std::runtime_error("Cannot create window");
    ShowWindow(window_, SW_SHOW);
}

void RenderingSystem::CreateDeviceObjects()
{
#if defined(_DEBUG)
    ComPtr<ID3D12Debug> debug;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
#endif
    ComPtr<IDXGIFactory6> factory;
    UINT flags = 0;
#if defined(_DEBUG)
    flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
    Check(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)), "Cannot create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)))) break;
        adapter.Reset();
    }
    if (!device_) Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)), "Cannot create D3D12 device");

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Check(device_->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue_)), "Cannot create command queue");
    for (auto& allocator : allocators_)
        Check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "Cannot create allocator");
    Check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators_[0].Get(), nullptr,
        IID_PPV_ARGS(&commandList_)), "Cannot create command list");
    Check(commandList_->Close(), "Cannot close command list");

    DXGI_SWAP_CHAIN_DESC1 swapDesc{};
    swapDesc.Width = width_; swapDesc.Height = height_;
    swapDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapDesc.SampleDesc.Count = 1;
    swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.BufferCount = FrameCount;
    swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain1> swapChain1;
    Check(factory->CreateSwapChainForHwnd(queue_.Get(), window_, &swapDesc, nullptr, nullptr, &swapChain1), "Cannot create swap chain");
    Check(swapChain1.As(&swapChain_), "Cannot query swap chain");
    frameIndex_ = swapChain_->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.NumDescriptors = FrameCount; rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    Check(device_->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap_)), "Cannot create RTV heap");
    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{};
    dsvDesc.NumDescriptors = 1 + CascadeCount; dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    Check(device_->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dsvHeap_)), "Cannot create DSV heap");
    rtvSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    dsvSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    srvSize_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CreateSizeDependentObjects();

    Check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "Cannot create fence");
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent_) throw std::runtime_error("Cannot create fence event");
}

void RenderingSystem::CreateSizeDependentObjects()
{
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < FrameCount; ++i) {
        Check(swapChain_->GetBuffer(i, IID_PPV_ARGS(&backBuffers_[i])), "Cannot get back buffer");
        device_->CreateRenderTargetView(backBuffers_[i].Get(), nullptr, rtv);
        rtv.ptr += rtvSize_;
    }
    D3D12_RESOURCE_DESC depthDesc{};
    depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDesc.Width = width_; depthDesc.Height = height_;
    depthDesc.DepthOrArraySize = 1; depthDesc.MipLevels = 1;
    depthDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{}; clear.Format = DXGI_FORMAT_D32_FLOAT; clear.DepthStencil.Depth = 1.0f;
    auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    Check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &depthDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(&depthBuffer_)), "Cannot create depth buffer");
    device_->CreateDepthStencilView(depthBuffer_.Get(), nullptr, dsvHeap_->GetCPUDescriptorHandleForHeapStart());
}

void RenderingSystem::CreatePipeline()
{
    D3D12_DESCRIPTOR_RANGE textureRange{};
    textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    textureRange.NumDescriptors = 1;
    textureRange.BaseShaderRegister = 0;
    textureRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER geometryParameters[2]{};
    geometryParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    geometryParameters[0].Descriptor.ShaderRegister = 0;
    geometryParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    geometryParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    geometryParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    geometryParameters[1].DescriptorTable.pDescriptorRanges = &textureRange;
    geometryParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 2; rootDesc.pParameters = geometryParameters;
    rootDesc.NumStaticSamplers = 1; rootDesc.pStaticSamplers = &sampler;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> serialized, error;
    Check(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &error), "Cannot serialize geometry root signature");
    Check(device_->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&geometryRootSignature_)), "Cannot create geometry root signature");

    D3D12_DESCRIPTOR_RANGE gBufferRange{};
    gBufferRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    gBufferRange.NumDescriptors = GBuffer::TargetCount;
    gBufferRange.BaseShaderRegister = 0;
    gBufferRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_DESCRIPTOR_RANGE shadowRange{};
    shadowRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    shadowRange.NumDescriptors = 1;
    shadowRange.BaseShaderRegister = 3;
    shadowRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_DESCRIPTOR_RANGE iblRange{};
    iblRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    iblRange.NumDescriptors = 3;
    iblRange.BaseShaderRegister = 4;
    iblRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER lightingParameters[4]{};
    lightingParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    lightingParameters[0].Descriptor.ShaderRegister = 0;
    lightingParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    lightingParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    lightingParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    lightingParameters[1].DescriptorTable.pDescriptorRanges = &gBufferRange;
    lightingParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    lightingParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    lightingParameters[2].DescriptorTable.NumDescriptorRanges = 1;
    lightingParameters[2].DescriptorTable.pDescriptorRanges = &shadowRange;
    lightingParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    lightingParameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    lightingParameters[3].DescriptorTable.NumDescriptorRanges = 1;
    lightingParameters[3].DescriptorTable.pDescriptorRanges = &iblRange;
    lightingParameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC shadowSampler{};
    shadowSampler.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
    shadowSampler.AddressU = shadowSampler.AddressV = shadowSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    shadowSampler.MipLODBias = 0.0f;
    shadowSampler.MaxAnisotropy = 1;
    shadowSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    shadowSampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE;
    shadowSampler.MinLOD = 0.0f;
    shadowSampler.MaxLOD = D3D12_FLOAT32_MAX;
    shadowSampler.ShaderRegister = 1;
    shadowSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC iblSampler{};
    iblSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    iblSampler.AddressU = iblSampler.AddressV = iblSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    iblSampler.MaxLOD = D3D12_FLOAT32_MAX;
    iblSampler.ShaderRegister = 2;
    iblSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    const D3D12_STATIC_SAMPLER_DESC lightingSamplers[]{shadowSampler, iblSampler};
    D3D12_ROOT_SIGNATURE_DESC lightingRootDesc{};
    lightingRootDesc.NumParameters = 4;
    lightingRootDesc.pParameters = lightingParameters;
    lightingRootDesc.NumStaticSamplers = _countof(lightingSamplers);
    lightingRootDesc.pStaticSamplers = lightingSamplers;
    serialized.Reset(); error.Reset();
    Check(D3D12SerializeRootSignature(&lightingRootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &error), "Cannot serialize lighting root signature");
    Check(device_->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&lightingRootSignature_)), "Cannot create lighting root signature");

    D3D12_DESCRIPTOR_RANGE tessellationRange{};
    tessellationRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    tessellationRange.NumDescriptors = 3;
    tessellationRange.BaseShaderRegister = 0;
    tessellationRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER tessellationParameters[2]{};
    tessellationParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    tessellationParameters[0].Descriptor.ShaderRegister = 0;
    tessellationParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    tessellationParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    tessellationParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    tessellationParameters[1].DescriptorTable.pDescriptorRanges = &tessellationRange;
    tessellationParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC tessellationRootDesc{};
    tessellationRootDesc.NumParameters = 2;
    tessellationRootDesc.pParameters = tessellationParameters;
    tessellationRootDesc.NumStaticSamplers = 1;
    D3D12_STATIC_SAMPLER_DESC tessellationSampler = sampler;
    tessellationSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    tessellationRootDesc.pStaticSamplers = &tessellationSampler;
    tessellationRootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    serialized.Reset(); error.Reset();
    Check(D3D12SerializeRootSignature(&tessellationRootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &error), "Cannot serialize tessellation root signature");
    Check(device_->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&tessellationRootSignature_)), "Cannot create tessellation root signature");

    D3D12_ROOT_PARAMETER instancingParameter{};
    instancingParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    instancingParameter.Descriptor.ShaderRegister = 0;
    instancingParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC instancingRootDesc{};
    instancingRootDesc.NumParameters = 1;
    instancingRootDesc.pParameters = &instancingParameter;
    instancingRootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS;
    serialized.Reset(); error.Reset();
    Check(D3D12SerializeRootSignature(&instancingRootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &error), "Cannot serialize instancing root signature");
    Check(device_->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&instancingRootSignature_)), "Cannot create instancing root signature");

    D3D12_DESCRIPTOR_RANGE particleInputRange{};
    particleInputRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    particleInputRange.NumDescriptors = 1;
    particleInputRange.BaseShaderRegister = 0;
    D3D12_DESCRIPTOR_RANGE particleOutputRange = particleInputRange;
    particleOutputRange.BaseShaderRegister = 1;
    D3D12_ROOT_PARAMETER particleComputeParameters[3]{};
    particleComputeParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    particleComputeParameters[0].Descriptor.ShaderRegister = 0;
    particleComputeParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    particleComputeParameters[1].DescriptorTable = {1, &particleInputRange};
    particleComputeParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    particleComputeParameters[2].DescriptorTable = {1, &particleOutputRange};
    D3D12_ROOT_SIGNATURE_DESC particleComputeDesc{};
    particleComputeDesc.NumParameters = _countof(particleComputeParameters);
    particleComputeDesc.pParameters = particleComputeParameters;
    serialized.Reset(); error.Reset();
    Check(D3D12SerializeRootSignature(&particleComputeDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &error), "Cannot serialize particle compute root signature");
    Check(device_->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&particleComputeRootSignature_)), "Cannot create particle compute root signature");

    D3D12_DESCRIPTOR_RANGE particleSrvRange{};
    particleSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    particleSrvRange.NumDescriptors = 1;
    particleSrvRange.BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER particleRenderParameters[2]{};
    particleRenderParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    particleRenderParameters[0].Descriptor.ShaderRegister = 0;
    particleRenderParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    particleRenderParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    particleRenderParameters[1].DescriptorTable = {1, &particleSrvRange};
    particleRenderParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC particleRenderDesc{};
    particleRenderDesc.NumParameters = _countof(particleRenderParameters);
    particleRenderDesc.pParameters = particleRenderParameters;
    particleRenderDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    serialized.Reset(); error.Reset();
    Check(D3D12SerializeRootSignature(&particleRenderDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized, &error), "Cannot serialize particle render root signature");
    Check(device_->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
        IID_PPV_ARGS(&particleRenderRootSignature_)), "Cannot create particle render root signature");

    UINT shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
    shaderFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
    ComPtr<ID3DBlob> geometryVs, geometryPs, lightingVs, lightingPs, instancingVs, instancingPs, shadowVs;
    ComPtr<ID3DBlob> tessellationVs, tessellationHs, tessellationDs, tessellationPs;
    ComPtr<ID3DBlob> particleCs, particleVs, particleGs, particlePs;
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "GeometryVS", "vs_5_1", shaderFlags, 0, &geometryVs, &error), "Cannot compile geometry vertex shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "GeometryPS", "ps_5_1", shaderFlags, 0, &geometryPs, &error), "Cannot compile geometry pixel shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "LightingVS", "vs_5_1", shaderFlags, 0, &lightingVs, &error), "Cannot compile lighting vertex shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "LightingPS", "ps_5_1", shaderFlags, 0, &lightingPs, &error), "Cannot compile lighting pixel shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "TessellationVS", "vs_5_1", shaderFlags, 0, &tessellationVs, &error), "Cannot compile tessellation vertex shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "TessellationHS", "hs_5_1", shaderFlags, 0, &tessellationHs, &error), "Cannot compile hull shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "TessellationDS", "ds_5_1", shaderFlags, 0, &tessellationDs, &error), "Cannot compile domain shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "TessellationPS", "ps_5_1", shaderFlags, 0, &tessellationPs, &error), "Cannot compile tessellation pixel shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "InstancingVS", "vs_5_1", shaderFlags, 0, &instancingVs, &error), "Cannot compile instancing vertex shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "InstancingPS", "ps_5_1", shaderFlags, 0, &instancingPs, &error), "Cannot compile instancing pixel shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "ShadowVS", "vs_5_1", shaderFlags, 0, &shadowVs, &error), "Cannot compile shadow vertex shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "ParticleCS", "cs_5_1", shaderFlags, 0, &particleCs, &error), "Cannot compile particle compute shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "ParticleVS", "vs_5_1", shaderFlags, 0, &particleVs, &error), "Cannot compile particle vertex shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "ParticleGS", "gs_5_1", shaderFlags, 0, &particleGs, &error), "Cannot compile particle geometry shader");
    error.Reset();
    Check(D3DCompileFromFile(L"Textured.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
        "ParticlePS", "ps_5_1", shaderFlags, 0, &particlePs, &error), "Cannot compile particle pixel shader");
    D3D12_INPUT_ELEMENT_DESC input[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, normal), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(Vertex, uv), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(Vertex, tangent), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = geometryRootSignature_.Get();
    pso.VS = {geometryVs->GetBufferPointer(), geometryVs->GetBufferSize()};
    pso.PS = {geometryPs->GetBufferPointer(), geometryPs->GetBufferSize()};
    pso.InputLayout = {input, _countof(input)};
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.DepthStencilState.DepthEnable = TRUE;
    pso.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    pso.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = GBuffer::TargetCount;
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    pso.RTVFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    pso.SampleDesc.Count = 1;
    Check(device_->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&geometryPipeline_)), "Cannot create geometry pipeline state");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC lightPso{};
    lightPso.pRootSignature = lightingRootSignature_.Get();
    lightPso.VS = {lightingVs->GetBufferPointer(), lightingVs->GetBufferSize()};
    lightPso.PS = {lightingPs->GetBufferPointer(), lightingPs->GetBufferSize()};
    lightPso.SampleMask = UINT_MAX;
    lightPso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    lightPso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    lightPso.RasterizerState.DepthClipEnable = TRUE;
    lightPso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    lightPso.DepthStencilState.DepthEnable = FALSE;
    lightPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    lightPso.NumRenderTargets = 1;
    lightPso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    lightPso.SampleDesc.Count = 1;
    Check(device_->CreateGraphicsPipelineState(&lightPso, IID_PPV_ARGS(&lightingPipeline_)), "Cannot create lighting pipeline state");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC tessellationPso = pso;
    tessellationPso.pRootSignature = tessellationRootSignature_.Get();
    tessellationPso.VS = {tessellationVs->GetBufferPointer(), tessellationVs->GetBufferSize()};
    tessellationPso.HS = {tessellationHs->GetBufferPointer(), tessellationHs->GetBufferSize()};
    tessellationPso.DS = {tessellationDs->GetBufferPointer(), tessellationDs->GetBufferSize()};
    tessellationPso.PS = {tessellationPs->GetBufferPointer(), tessellationPs->GetBufferSize()};
    tessellationPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    Check(device_->CreateGraphicsPipelineState(&tessellationPso, IID_PPV_ARGS(&tessellationPipeline_)),
        "Cannot create tessellation pipeline state");
    tessellationPso.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    Check(device_->CreateGraphicsPipelineState(&tessellationPso, IID_PPV_ARGS(&tessellationWireframePipeline_)),
        "Cannot create tessellation wireframe pipeline state");

    D3D12_INPUT_ELEMENT_DESC instancingInput[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, position), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, normal), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(Vertex, uv), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TANGENT", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(Vertex, tangent), D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"INSTANCE_POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, offsetof(InstanceData, position), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_SCALE", 0, DXGI_FORMAT_R32G32B32_FLOAT, 1, offsetof(InstanceData, scale), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, offsetof(InstanceData, color), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        {"INSTANCE_ROUGHNESS", 0, DXGI_FORMAT_R32_FLOAT, 1, offsetof(InstanceData, roughness), D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1}
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC instancingPso = pso;
    instancingPso.pRootSignature = instancingRootSignature_.Get();
    instancingPso.VS = {instancingVs->GetBufferPointer(), instancingVs->GetBufferSize()};
    instancingPso.PS = {instancingPs->GetBufferPointer(), instancingPs->GetBufferSize()};
    instancingPso.InputLayout = {instancingInput, _countof(instancingInput)};
    instancingPso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    Check(device_->CreateGraphicsPipelineState(&instancingPso, IID_PPV_ARGS(&instancingPipeline_)),
        "Cannot create instancing pipeline state");

    shadowRootSignature_ = instancingRootSignature_;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC shadowPso = instancingPso;
    shadowPso.pRootSignature = shadowRootSignature_.Get();
    shadowPso.VS = {shadowVs->GetBufferPointer(), shadowVs->GetBufferSize()};
    shadowPso.PS = {};
    shadowPso.NumRenderTargets = 0;
    for (auto& format : shadowPso.RTVFormats) format = DXGI_FORMAT_UNKNOWN;
    shadowPso.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    shadowPso.RasterizerState.DepthBias = 1200;
    shadowPso.RasterizerState.SlopeScaledDepthBias = 1.5f;
    shadowPso.RasterizerState.DepthBiasClamp = 0.01f;
    Check(device_->CreateGraphicsPipelineState(&shadowPso, IID_PPV_ARGS(&shadowPipeline_)),
        "Cannot create shadow pipeline state");

    D3D12_COMPUTE_PIPELINE_STATE_DESC particleComputePso{};
    particleComputePso.pRootSignature = particleComputeRootSignature_.Get();
    particleComputePso.CS = {particleCs->GetBufferPointer(), particleCs->GetBufferSize()};
    Check(device_->CreateComputePipelineState(&particleComputePso, IID_PPV_ARGS(&particleComputePipeline_)),
        "Cannot create particle compute pipeline state");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC particleRenderPso = pso;
    particleRenderPso.pRootSignature = particleRenderRootSignature_.Get();
    particleRenderPso.VS = {particleVs->GetBufferPointer(), particleVs->GetBufferSize()};
    particleRenderPso.GS = {particleGs->GetBufferPointer(), particleGs->GetBufferSize()};
    particleRenderPso.PS = {particlePs->GetBufferPointer(), particlePs->GetBufferSize()};
    particleRenderPso.InputLayout = {nullptr, 0};
    particleRenderPso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    particleRenderPso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    Check(device_->CreateGraphicsPipelineState(&particleRenderPso, IID_PPV_ARGS(&particleRenderPipeline_)),
        "Cannot create particle render pipeline state");
}

ComPtr<ID3D12Resource> RenderingSystem::CreateDefaultBuffer(const void* data, UINT64 size)
{
    auto defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto uploadHeap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    auto desc = BufferDescription(size);
    ComPtr<ID3D12Resource> destination, upload;
    Check(device_->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&destination)), "Cannot create GPU buffer");
    Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "Cannot create upload buffer");
    void* mapped = nullptr; Check(upload->Map(0, nullptr, &mapped), "Cannot map upload buffer");
    memcpy(mapped, data, static_cast<size_t>(size)); upload->Unmap(0, nullptr);
    commandList_->CopyBufferRegion(destination.Get(), 0, upload.Get(), 0, size);
    auto barrier = Transition(destination.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
    commandList_->ResourceBarrier(1, &barrier);
    initializationUploads_.push_back(upload);
    return destination;
}

ComPtr<ID3D12Resource> RenderingSystem::CreateTexture(const ImageData& image, UINT srvIndex)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = image.width; desc.Height = image.height;
    desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    auto defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    ComPtr<ID3D12Resource> texture;
    Check(device_->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)), "Cannot create texture");
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0; UINT64 rowSize = 0, totalSize = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowSize, &totalSize);
    auto uploadHeap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    auto uploadDesc = BufferDescription(totalSize);
    ComPtr<ID3D12Resource> upload;
    Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "Cannot create texture upload buffer");
    uint8_t* mapped = nullptr; Check(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Cannot map texture upload");
    const UINT sourcePitch = image.width * 4;
    for (UINT y = 0; y < image.height; ++y)
        memcpy(mapped + footprint.Offset + static_cast<size_t>(y) * footprint.Footprint.RowPitch,
            image.pixels.data() + static_cast<size_t>(y) * sourcePitch, sourcePitch);
    upload->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = upload.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = texture.Get(); destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    commandList_->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    const auto shaderResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    auto barrier = Transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderResourceState);
    commandList_->ResourceBarrier(1, &barrier);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = desc.Format; srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    auto handle = srvHeap_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(srvIndex) * srvSize_;
    device_->CreateShaderResourceView(texture.Get(), &srv, handle);
    initializationUploads_.push_back(upload);
    return texture;
}

ComPtr<ID3D12Resource> RenderingSystem::CreateDdsTexture(const fs::path& path, UINT srvIndex)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Cannot open DDS texture");
    const size_t fileSize = static_cast<size_t>(input.tellg());
    input.seekg(0);
    std::vector<uint8_t> bytes(fileSize);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    auto read32 = [&](size_t offset) {
        if (offset + sizeof(uint32_t) > bytes.size()) throw std::runtime_error("Invalid DDS header");
        uint32_t value = 0; memcpy(&value, bytes.data() + offset, sizeof(value)); return value;
    };
    if (bytes.size() < 148 || read32(0) != 0x20534444u || read32(4) != 124u || read32(84) != 0x30315844u)
        throw std::runtime_error("Only DDS files with a DX10 header are supported");
    const UINT height = read32(12);
    const UINT width = read32(16);
    const UINT mipLevels = std::max<UINT>(1, read32(28));
    const DXGI_FORMAT format = static_cast<DXGI_FORMAT>(read32(128));
    const bool isCube = (read32(136) & 0x4u) != 0;
    const UINT arraySize = std::max<UINT>(1, read32(140));
    if (arraySize != 1 || (format != DXGI_FORMAT_BC6H_UF16 && format != DXGI_FORMAT_R32G32_FLOAT))
        throw std::runtime_error("Unsupported DDS format or array size");
    const UINT slices = isCube ? 6u : 1u;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = static_cast<UINT16>(slices);
    desc.MipLevels = static_cast<UINT16>(mipLevels);
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    auto defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    ComPtr<ID3D12Resource> texture;
    Check(device_->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)), "Cannot create DDS texture");

    const UINT subresourceCount = slices * mipLevels;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(subresourceCount);
    std::vector<UINT> rowCounts(subresourceCount);
    std::vector<UINT64> rowSizes(subresourceCount);
    UINT64 uploadSize = 0;
    device_->GetCopyableFootprints(&desc, 0, subresourceCount, 0, layouts.data(),
        rowCounts.data(), rowSizes.data(), &uploadSize);
    auto uploadHeap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    auto uploadDesc = BufferDescription(uploadSize);
    ComPtr<ID3D12Resource> upload;
    Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload)), "Cannot create DDS upload buffer");
    uint8_t* mapped = nullptr;
    Check(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "Cannot map DDS upload buffer");
    size_t sourceOffset = 148;
    for (UINT slice = 0; slice < slices; ++slice) {
        for (UINT mip = 0; mip < mipLevels; ++mip) {
            const UINT subresource = slice * mipLevels + mip;
            const UINT mipWidth = std::max<UINT>(1, width >> mip);
            const UINT mipHeight = std::max<UINT>(1, height >> mip);
            const UINT sourceRows = format == DXGI_FORMAT_BC6H_UF16 ? std::max<UINT>(1, (mipHeight + 3) / 4) : mipHeight;
            const UINT sourceRowBytes = format == DXGI_FORMAT_BC6H_UF16 ?
                std::max<UINT>(1, (mipWidth + 3) / 4) * 16u : mipWidth * 8u;
            const size_t subresourceBytes = static_cast<size_t>(sourceRows) * sourceRowBytes;
            if (sourceOffset + subresourceBytes > bytes.size()) throw std::runtime_error("DDS data is truncated");
            for (UINT row = 0; row < sourceRows; ++row)
                memcpy(mapped + layouts[subresource].Offset + static_cast<size_t>(row) * layouts[subresource].Footprint.RowPitch,
                    bytes.data() + sourceOffset + static_cast<size_t>(row) * sourceRowBytes, sourceRowBytes);
            sourceOffset += subresourceBytes;
        }
    }
    upload->Unmap(0, nullptr);
    for (UINT subresource = 0; subresource < subresourceCount; ++subresource) {
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = upload.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = layouts[subresource];
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = texture.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = subresource;
        commandList_->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    }
    const auto shaderResourceState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    auto barrier = Transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderResourceState);
    commandList_->ResourceBarrier(1, &barrier);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = format;
    if (isCube) {
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
        srv.TextureCube.MipLevels = mipLevels;
    } else {
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = mipLevels;
    }
    auto handle = srvHeap_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(srvIndex) * srvSize_;
    device_->CreateShaderResourceView(texture.Get(), &srv, handle);
    initializationUploads_.push_back(upload);
    return texture;
}

void RenderingSystem::LoadScene()
{
    model_ = LoadObj(fs::path(L"assets") / L"sponza" / L"sponza.obj");
    gBufferSrvStart_ = static_cast<UINT>(model_.materials.size());
    tessellationSrvStart_ = gBufferSrvStart_ + GBuffer::TargetCount;
    shadowSrvIndex_ = tessellationSrvStart_ + 3;
    particleUavStart_ = shadowSrvIndex_ + 1;
    particleSrvStart_ = particleUavStart_ + 2;
    iblSrvStart_ = particleSrvStart_ + 2;
    D3D12_DESCRIPTOR_HEAP_DESC srvDesc{};
    srvDesc.NumDescriptors = std::max<UINT>(1, static_cast<UINT>(model_.materials.size())) +
        GBuffer::TargetCount + 11;
    srvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Check(device_->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&srvHeap_)), "Cannot create SRV heap");
    gBuffer_.Create(device_.Get(), width_, height_, srvHeap_->GetCPUDescriptorHandleForHeapStart(),
        srvSize_, gBufferSrvStart_);

    D3D12_RESOURCE_DESC shadowDesc{};
    shadowDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    shadowDesc.Width = ShadowMapSize;
    shadowDesc.Height = ShadowMapSize;
    shadowDesc.DepthOrArraySize = CascadeCount;
    shadowDesc.MipLevels = 1;
    shadowDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    shadowDesc.SampleDesc.Count = 1;
    shadowDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    shadowDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE shadowClear{};
    shadowClear.Format = DXGI_FORMAT_D32_FLOAT;
    shadowClear.DepthStencil.Depth = 1.0f;
    auto shadowHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    Check(device_->CreateCommittedResource(&shadowHeap, D3D12_HEAP_FLAG_NONE, &shadowDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &shadowClear, IID_PPV_ARGS(&shadowMap_)),
        "Cannot create cascaded shadow map");
    for (UINT cascade = 0; cascade < CascadeCount; ++cascade) {
        D3D12_DEPTH_STENCIL_VIEW_DESC shadowDsv{};
        shadowDsv.Format = DXGI_FORMAT_D32_FLOAT;
        shadowDsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        shadowDsv.Texture2DArray.FirstArraySlice = cascade;
        shadowDsv.Texture2DArray.ArraySize = 1;
        shadowDsv.Texture2DArray.MipSlice = 0;
        auto shadowDsvHandle = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
        shadowDsvHandle.ptr += static_cast<SIZE_T>(1 + cascade) * dsvSize_;
        device_->CreateDepthStencilView(shadowMap_.Get(), &shadowDsv, shadowDsvHandle);
    }
    D3D12_SHADER_RESOURCE_VIEW_DESC shadowSrv{};
    shadowSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    shadowSrv.Format = DXGI_FORMAT_R32_FLOAT;
    shadowSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    shadowSrv.Texture2DArray.MipLevels = 1;
    shadowSrv.Texture2DArray.ArraySize = CascadeCount;
    auto shadowSrvHandle = srvHeap_->GetCPUDescriptorHandleForHeapStart();
    shadowSrvHandle.ptr += static_cast<SIZE_T>(shadowSrvIndex_) * srvSize_;
    device_->CreateShaderResourceView(shadowMap_.Get(), &shadowSrv, shadowSrvHandle);
    Check(allocators_[frameIndex_]->Reset(), "Cannot reset init allocator");
    Check(commandList_->Reset(allocators_[frameIndex_].Get(), nullptr), "Cannot reset init command list");
    vertexBuffer_ = CreateDefaultBuffer(model_.vertices.data(), model_.vertices.size() * sizeof(Vertex));
    indexBuffer_ = CreateDefaultBuffer(model_.indices.data(), model_.indices.size() * sizeof(uint32_t));
    vertexView_ = {vertexBuffer_->GetGPUVirtualAddress(), static_cast<UINT>(model_.vertices.size() * sizeof(Vertex)), sizeof(Vertex)};
    indexView_ = {indexBuffer_->GetGPUVirtualAddress(), static_cast<UINT>(model_.indices.size() * sizeof(uint32_t)), DXGI_FORMAT_R32_UINT};
    const std::array<Vertex, 6> tessellationVertices{{
        {{-3.5f, 1.5f, 0}, {0, 0, -1}, {0, 2}, {1, 0, 0, -1}},
        {{ 3.5f, 1.5f, 0}, {0, 0, -1}, {3, 2}, {1, 0, 0, -1}},
        {{ 3.5f, 6.5f, 0}, {0, 0, -1}, {3, 0}, {1, 0, 0, -1}},
        {{-3.5f, 1.5f, 0}, {0, 0, -1}, {0, 2}, {1, 0, 0, -1}},
        {{ 3.5f, 6.5f, 0}, {0, 0, -1}, {3, 0}, {1, 0, 0, -1}},
        {{-3.5f, 6.5f, 0}, {0, 0, -1}, {0, 0}, {1, 0, 0, -1}}}};
    tessellationVertexBuffer_ = CreateDefaultBuffer(tessellationVertices.data(), sizeof(tessellationVertices));
    tessellationVertexView_ = {tessellationVertexBuffer_->GetGPUVirtualAddress(),
        static_cast<UINT>(sizeof(tessellationVertices)), sizeof(Vertex)};

    const std::array<Vertex, 24> cubeVertices{{
        {{-0.5f,-0.5f,-0.5f},{ 0, 0,-1},{0,1}}, {{ 0.5f,-0.5f,-0.5f},{ 0, 0,-1},{1,1}},
        {{ 0.5f, 0.5f,-0.5f},{ 0, 0,-1},{1,0}}, {{-0.5f, 0.5f,-0.5f},{ 0, 0,-1},{0,0}},
        {{ 0.5f,-0.5f, 0.5f},{ 0, 0, 1},{0,1}}, {{-0.5f,-0.5f, 0.5f},{ 0, 0, 1},{1,1}},
        {{-0.5f, 0.5f, 0.5f},{ 0, 0, 1},{1,0}}, {{ 0.5f, 0.5f, 0.5f},{ 0, 0, 1},{0,0}},
        {{-0.5f,-0.5f, 0.5f},{-1, 0, 0},{0,1}}, {{-0.5f,-0.5f,-0.5f},{-1, 0, 0},{1,1}},
        {{-0.5f, 0.5f,-0.5f},{-1, 0, 0},{1,0}}, {{-0.5f, 0.5f, 0.5f},{-1, 0, 0},{0,0}},
        {{ 0.5f,-0.5f,-0.5f},{ 1, 0, 0},{0,1}}, {{ 0.5f,-0.5f, 0.5f},{ 1, 0, 0},{1,1}},
        {{ 0.5f, 0.5f, 0.5f},{ 1, 0, 0},{1,0}}, {{ 0.5f, 0.5f,-0.5f},{ 1, 0, 0},{0,0}},
        {{-0.5f, 0.5f,-0.5f},{ 0, 1, 0},{0,1}}, {{ 0.5f, 0.5f,-0.5f},{ 0, 1, 0},{1,1}},
        {{ 0.5f, 0.5f, 0.5f},{ 0, 1, 0},{1,0}}, {{-0.5f, 0.5f, 0.5f},{ 0, 1, 0},{0,0}},
        {{-0.5f,-0.5f, 0.5f},{ 0,-1, 0},{0,1}}, {{ 0.5f,-0.5f, 0.5f},{ 0,-1, 0},{1,1}},
        {{ 0.5f,-0.5f,-0.5f},{ 0,-1, 0},{1,0}}, {{-0.5f,-0.5f,-0.5f},{ 0,-1, 0},{0,0}}
    }};
    const std::array<uint16_t, 36> cubeIndices{{
         0, 1, 2,  0, 2, 3,  4, 5, 6,  4, 6, 7,
         8, 9,10,  8,10,11, 12,13,14, 12,14,15,
        16,17,18, 16,18,19, 20,21,22, 20,22,23
    }};
    cubeVertexBuffer_ = CreateDefaultBuffer(cubeVertices.data(), sizeof(cubeVertices));
    cubeIndexBuffer_ = CreateDefaultBuffer(cubeIndices.data(), sizeof(cubeIndices));
    cubeVertexView_ = {cubeVertexBuffer_->GetGPUVirtualAddress(),
        static_cast<UINT>(sizeof(cubeVertices)), sizeof(Vertex)};
    cubeIndexView_ = {cubeIndexBuffer_->GetGPUVirtualAddress(),
        static_cast<UINT>(sizeof(cubeIndices)), DXGI_FORMAT_R16_UINT};

    // UV sphere used by the metallic/roughness material chart.
    constexpr UINT sphereSlices = 32;
    constexpr UINT sphereStacks = 20;
    std::vector<Vertex> sphereVertices;
    std::vector<uint32_t> sphereIndices;
    sphereVertices.reserve((sphereSlices + 1) * (sphereStacks + 1));
    sphereIndices.reserve(sphereSlices * sphereStacks * 6);
    for (UINT stack = 0; stack <= sphereStacks; ++stack) {
        const float v = static_cast<float>(stack) / sphereStacks;
        const float phi = v * XM_PI;
        for (UINT slice = 0; slice <= sphereSlices; ++slice) {
            const float u = static_cast<float>(slice) / sphereSlices;
            const float theta = u * XM_2PI;
            const XMFLOAT3 normal{sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)};
            sphereVertices.push_back({normal, normal, {u, v}});
        }
    }
    for (UINT stack = 0; stack < sphereStacks; ++stack) {
        for (UINT slice = 0; slice < sphereSlices; ++slice) {
            const uint32_t a = stack * (sphereSlices + 1) + slice;
            const uint32_t b = a + 1;
            const uint32_t c = a + sphereSlices + 1;
            const uint32_t d = c + 1;
            sphereIndices.insert(sphereIndices.end(), {a, c, b, b, c, d});
        }
    }
    sphereVertexBuffer_ = CreateDefaultBuffer(sphereVertices.data(), sphereVertices.size() * sizeof(Vertex));
    sphereIndexBuffer_ = CreateDefaultBuffer(sphereIndices.data(), sphereIndices.size() * sizeof(uint32_t));
    sphereVertexView_ = {sphereVertexBuffer_->GetGPUVirtualAddress(),
        static_cast<UINT>(sphereVertices.size() * sizeof(Vertex)), sizeof(Vertex)};
    sphereIndexView_ = {sphereIndexBuffer_->GetGPUVirtualAddress(),
        static_cast<UINT>(sphereIndices.size() * sizeof(uint32_t)), DXGI_FORMAT_R32_UINT};
    sphereIndexCount_ = static_cast<UINT>(sphereIndices.size());

    std::vector<ParticleData> initialParticles(ParticleCount);
    for (UINT i = 0; i < ParticleCount; ++i) {
        const float a = static_cast<float>((i * 16807u) % 10000u) / 10000.0f;
        const float b = static_cast<float>((i * 48271u + 17u) % 10000u) / 10000.0f;
        float age = b * 2.8f;
        const float angle = a * XM_2PI;
        const float speed = 4.5f + 5.5f * static_cast<float>((i * 69621u + 31u) % 10000u) / 10000.0f;
        XMFLOAT3 velocity{cosf(angle) * speed * 0.22f, speed, sinf(angle) * speed * 0.22f};
        XMFLOAT3 position{velocity.x * age, 0.1f + velocity.y * age - 2.6f * age * age,
            8.0f + velocity.z * age};
        if (position.y < 0.05f) { position = {0.0f, 0.1f, 8.0f}; age = 0.0f; }
        initialParticles[i] = {position, age, velocity, 3.0f,
            {1.0f, 0.25f + 0.65f * a, 0.04f, 1.0f}, 0.12f + 0.18f * b, {}};
    }
    auto gpuHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    auto uploadHeapForParticles = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    auto particleDesc = BufferDescription(static_cast<UINT64>(ParticleCount) * sizeof(ParticleData));
    particleDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto particleUploadDesc = BufferDescription(particleDesc.Width);
    ComPtr<ID3D12Resource> particleUpload;
    Check(device_->CreateCommittedResource(&uploadHeapForParticles, D3D12_HEAP_FLAG_NONE, &particleUploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&particleUpload)), "Cannot create particle upload buffer");
    void* particleMapped = nullptr;
    Check(particleUpload->Map(0, nullptr, &particleMapped), "Cannot map particle upload buffer");
    memcpy(particleMapped, initialParticles.data(), static_cast<size_t>(particleDesc.Width));
    particleUpload->Unmap(0, nullptr);
    Check(device_->CreateCommittedResource(&gpuHeap, D3D12_HEAP_FLAG_NONE, &particleDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&particleBuffers_[0])), "Cannot create particle buffer A");
    Check(device_->CreateCommittedResource(&gpuHeap, D3D12_HEAP_FLAG_NONE, &particleDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&particleBuffers_[1])), "Cannot create particle buffer B");
    commandList_->CopyBufferRegion(particleBuffers_[0].Get(), 0, particleUpload.Get(), 0, particleDesc.Width);
    auto particleReady = Transition(particleBuffers_[0].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    commandList_->ResourceBarrier(1, &particleReady);
    initializationUploads_.push_back(particleUpload);

    auto counterDesc = BufferDescription(sizeof(UINT));
    counterDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    for (UINT i = 0; i < 2; ++i) {
        Check(device_->CreateCommittedResource(&gpuHeap, D3D12_HEAP_FLAG_NONE, &counterDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&particleCounters_[i])),
            "Cannot create particle counter");
        auto counterUploadDesc = BufferDescription(sizeof(UINT));
        ComPtr<ID3D12Resource> counterUpload;
        Check(device_->CreateCommittedResource(&uploadHeapForParticles, D3D12_HEAP_FLAG_NONE, &counterUploadDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&counterUpload)),
            "Cannot create particle counter upload");
        UINT* counterValue = nullptr;
        Check(counterUpload->Map(0, nullptr, reinterpret_cast<void**>(&counterValue)), "Cannot map particle counter");
        *counterValue = i == 0 ? ParticleCount : 0;
        counterUpload->Unmap(0, nullptr);
        commandList_->CopyBufferRegion(particleCounters_[i].Get(), 0, counterUpload.Get(), 0, sizeof(UINT));
        auto counterReady = Transition(particleCounters_[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        commandList_->ResourceBarrier(1, &counterReady);
        initializationUploads_.push_back(counterUpload);
    }
    for (UINT i = 0; i < 2; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.Buffer.NumElements = ParticleCount;
        uav.Buffer.StructureByteStride = sizeof(ParticleData);
        auto uavHandle = srvHeap_->GetCPUDescriptorHandleForHeapStart();
        uavHandle.ptr += static_cast<SIZE_T>(particleUavStart_ + i) * srvSize_;
        device_->CreateUnorderedAccessView(particleBuffers_[i].Get(), particleCounters_[i].Get(), &uav, uavHandle);

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Format = DXGI_FORMAT_UNKNOWN;
        srv.Buffer.NumElements = ParticleCount;
        srv.Buffer.StructureByteStride = sizeof(ParticleData);
        auto srvHandle = srvHeap_->GetCPUDescriptorHandleForHeapStart();
        srvHandle.ptr += static_cast<SIZE_T>(particleSrvStart_ + i) * srvSize_;
        device_->CreateShaderResourceView(particleBuffers_[i].Get(), &srv, srvHandle);
    }

    sceneObjects_.reserve(64 * 64);
    for (uint32_t z = 0; z < 64; ++z) {
        for (uint32_t x = 0; x < 64; ++x) {
            const uint32_t hash = x * 73856093u ^ z * 19349663u;
            const float scale = 0.75f + static_cast<float>(hash % 100u) / 100.0f;
            const float red = 0.2f + static_cast<float>((hash >> 3) % 70u) / 100.0f;
            const float green = 0.2f + static_cast<float>((hash >> 10) % 70u) / 100.0f;
            const float blue = 0.2f + static_cast<float>((hash >> 17) % 70u) / 100.0f;
            sceneObjects_.push_back({
                {(static_cast<float>(x) - 31.5f) * 2.5f, scale * 0.5f,
                    (static_cast<float>(z) - 31.5f) * 2.5f + 30.0f},
                scale, {red, green, blue, 0.72f}});
        }
    }
    visibleObjects_.reserve(sceneObjects_.size());
    octree_.Build(sceneObjects_, BoundingBox({0.0f, 5.0f, 30.0f}, {82.0f, 10.0f, 82.0f}));
    std::vector<InstanceData> shadowInstances;
    shadowInstances.reserve(sceneObjects_.size() + 1);
    for (const SceneObject& object : sceneObjects_)
        shadowInstances.push_back({object.position, 0.0f,
            {object.scale, object.scale, object.scale}, 0.0f, object.color});
    shadowInstances.push_back({{0.0f, -0.6f, 30.0f}, 0.0f,
        {165.0f, 1.0f, 165.0f}, 0.0f, {0.32f, 0.34f, 0.38f, 0.0f}});
    shadowInstanceBuffer_ = CreateDefaultBuffer(shadowInstances.data(),
        shadowInstances.size() * sizeof(InstanceData));
    shadowInstanceView_ = {shadowInstanceBuffer_->GetGPUVirtualAddress(),
        static_cast<UINT>(shadowInstances.size() * sizeof(InstanceData)), sizeof(InstanceData)};
    for (UINT i = 0; i < model_.materials.size(); ++i) {
        model_.materials[i].srvIndex = i;
        const auto& path = model_.materials[i].texturePath;
        ImageData image = !path.empty() && fs::exists(path) ? LoadImageWic(path) : MakeFallbackImage();
        textures_.push_back(CreateTexture(image, i));
    }
    const fs::path tessellationAssets = fs::path(L"assets") / L"tessellation";
    textures_.push_back(CreateTexture(LoadImageWic(tessellationAssets / L"bricks2.jpg"), tessellationSrvStart_));
    textures_.push_back(CreateTexture(LoadImageWic(tessellationAssets / L"bricks2_normal.jpg"), tessellationSrvStart_ + 1));
    textures_.push_back(CreateTexture(LoadImageWic(tessellationAssets / L"bricks2_disp.jpg"), tessellationSrvStart_ + 2));
    const fs::path iblAssets = fs::path(L"assets") / L"ibl";
    textures_.push_back(CreateDdsTexture(iblAssets / L"IrradianceMap_BC6U.dds", iblSrvStart_));
    textures_.push_back(CreateDdsTexture(iblAssets / L"IntegrationMap.dds", iblSrvStart_ + 1));
    textures_.push_back(CreateDdsTexture(iblAssets / L"PreFilteredEnvMap_BC6U.dds", iblSrvStart_ + 2));
    const UINT64 constantsSize = std::max<size_t>(1, model_.submeshes.size()) * sizeof(DrawConstants);
    auto uploadHeap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    auto bufferDesc = BufferDescription(constantsSize);
    for (UINT i = 0; i < FrameCount; ++i) {
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&constantBuffers_[i])), "Cannot create constant buffer");
        Check(constantBuffers_[i]->Map(0, nullptr, reinterpret_cast<void**>(&mappedConstants_[i])), "Cannot map constant buffer");
        auto lightDesc = BufferDescription(sizeof(LightConstants));
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &lightDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&lightBuffers_[i])), "Cannot create light buffer");
        Check(lightBuffers_[i]->Map(0, nullptr, reinterpret_cast<void**>(&mappedLights_[i])), "Cannot map light buffer");
        auto tessellationDesc = BufferDescription(sizeof(TessellationConstants));
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &tessellationDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&tessellationBuffers_[i])),
            "Cannot create tessellation constant buffer");
        Check(tessellationBuffers_[i]->Map(0, nullptr, reinterpret_cast<void**>(&mappedTessellation_[i])),
            "Cannot map tessellation constant buffer");
        auto instanceDesc = BufferDescription((sceneObjects_.size() + 1) * sizeof(InstanceData));
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &instanceDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&instanceBuffers_[i])),
            "Cannot create instance buffer");
        Check(instanceBuffers_[i]->Map(0, nullptr, reinterpret_cast<void**>(&mappedInstances_[i])),
            "Cannot map instance buffer");
        auto instancePassDesc = BufferDescription(sizeof(InstancePassConstants));
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &instancePassDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&instancePassBuffers_[i])),
            "Cannot create instance pass buffer");
        Check(instancePassBuffers_[i]->Map(0, nullptr, reinterpret_cast<void**>(&mappedInstancePass_[i])),
            "Cannot map instance pass buffer");
        auto shadowPassDesc = BufferDescription(CascadeCount * sizeof(ShadowPassConstants));
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &shadowPassDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&shadowPassBuffers_[i])),
            "Cannot create shadow pass buffer");
        Check(shadowPassBuffers_[i]->Map(0, nullptr, reinterpret_cast<void**>(&mappedShadowPass_[i])),
            "Cannot map shadow pass buffer");
        auto particleConstantDesc = BufferDescription(sizeof(ParticleConstants));
        Check(device_->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &particleConstantDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&particleConstantBuffers_[i])),
            "Cannot create particle constant buffer");
        Check(particleConstantBuffers_[i]->Map(0, nullptr,
            reinterpret_cast<void**>(&mappedParticleConstants_[i])), "Cannot map particle constant buffer");
    }
    Check(commandList_->Close(), "Cannot close init command list");
    ID3D12CommandList* lists[] = {commandList_.Get()}; queue_->ExecuteCommandLists(1, lists);
    Flush();
    initializationUploads_.clear();
}

void RenderingSystem::Render()
{
    auto now = std::chrono::steady_clock::now();
    float delta = std::chrono::duration<float>(now - previousTime_).count();
    previousTime_ = now;
    if (!paused_) animationTime_ += std::min(delta, 0.1f);
    const float movementSpeed = (GetAsyncKeyState(VK_SHIFT) & 0x8000) ? 9.0f : 4.0f;
    const float rotationSpeed = 1.35f;
    if (GetAsyncKeyState(VK_LEFT) & 0x8000) cameraYaw_ -= rotationSpeed * delta;
    if (GetAsyncKeyState(VK_RIGHT) & 0x8000) cameraYaw_ += rotationSpeed * delta;
    if (GetAsyncKeyState(VK_UP) & 0x8000) cameraPitch_ += rotationSpeed * delta;
    if (GetAsyncKeyState(VK_DOWN) & 0x8000) cameraPitch_ -= rotationSpeed * delta;
    cameraPitch_ = std::clamp(cameraPitch_, -1.35f, 1.35f);
    XMVECTOR cameraPositionVector = XMLoadFloat3(&cameraPosition_);
    XMVECTOR flatForward = XMVectorSet(sinf(cameraYaw_), 0, cosf(cameraYaw_), 0);
    XMVECTOR right = XMVectorSet(cosf(cameraYaw_), 0, -sinf(cameraYaw_), 0);
    if (GetAsyncKeyState('W') & 0x8000) cameraPositionVector += flatForward * (movementSpeed * delta);
    if (GetAsyncKeyState('S') & 0x8000) cameraPositionVector -= flatForward * (movementSpeed * delta);
    if (GetAsyncKeyState('D') & 0x8000) cameraPositionVector += right * (movementSpeed * delta);
    if (GetAsyncKeyState('A') & 0x8000) cameraPositionVector -= right * (movementSpeed * delta);
    if (GetAsyncKeyState('E') & 0x8000) cameraPositionVector += XMVectorSet(0, movementSpeed * delta, 0, 0);
    if (GetAsyncKeyState('Q') & 0x8000) cameraPositionVector -= XMVectorSet(0, movementSpeed * delta, 0, 0);
    XMStoreFloat3(&cameraPosition_, cameraPositionVector);
    Check(allocators_[frameIndex_]->Reset(), "Cannot reset frame allocator");
    Check(commandList_->Reset(allocators_[frameIndex_].Get(), geometryPipeline_.Get()), "Cannot reset command list");
    auto rtv = rtvHeap_->GetCPUDescriptorHandleForHeapStart(); rtv.ptr += static_cast<SIZE_T>(frameIndex_) * rtvSize_;
    auto dsv = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
    commandList_->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width_), static_cast<float>(height_), 0, 1};
    D3D12_RECT scissor{0, 0, static_cast<LONG>(width_), static_cast<LONG>(height_)};
    commandList_->RSSetViewports(1, &viewport); commandList_->RSSetScissorRects(1, &scissor);
    ID3D12DescriptorHeap* heaps[] = {srvHeap_.Get()}; commandList_->SetDescriptorHeaps(1, heaps);

    const XMVECTOR camera = XMVectorSet(cameraPosition_.x, cameraPosition_.y, cameraPosition_.z, 1);
    const float cosPitch = cosf(cameraPitch_);
    const XMVECTOR lookDirection = XMVectorSet(
        sinf(cameraYaw_) * cosPitch, sinf(cameraPitch_), cosf(cameraYaw_) * cosPitch, 0);
    const XMMATRIX view = XMMatrixLookToLH(camera, lookDirection, XMVectorSet(0, 1, 0, 0));
    const float verticalFov = XMConvertToRadians(62.0f);
    const float aspectRatio = float(width_) / height_;
    const XMMATRIX projection = XMMatrixPerspectiveFovLH(verticalFov, aspectRatio, 0.1f, 250.0f);
    const CascadeMatrices cascades = CalculateCascades(camera, lookDirection,
        verticalFov, aspectRatio, 0.1f, 250.0f);

    const UINT particleDestination = 1u - particleSource_;
    ParticleConstants particleConstants{};
    XMStoreFloat4x4(&particleConstants.viewProjection, XMMatrixTranspose(view * projection));
    XMStoreFloat4(&particleConstants.cameraRight, XMVectorSetW(XMVector3Normalize(right), 0.0f));
    const XMVECTOR cameraUp = XMVector3Normalize(XMVector3Cross(lookDirection, right));
    XMStoreFloat4(&particleConstants.cameraUp, XMVectorSetW(cameraUp, 0.0f));
    particleConstants.emitterAndTime = {0.0f, 0.1f, 8.0f, paused_ ? 0.0f : std::min(delta, 0.033f)};
    memcpy(mappedParticleConstants_[frameIndex_], &particleConstants, sizeof(particleConstants));
    commandList_->SetPipelineState(particleComputePipeline_.Get());
    commandList_->SetComputeRootSignature(particleComputeRootSignature_.Get());
    commandList_->SetComputeRootConstantBufferView(0,
        particleConstantBuffers_[frameIndex_]->GetGPUVirtualAddress());
    auto particleInput = srvHeap_->GetGPUDescriptorHandleForHeapStart();
    particleInput.ptr += static_cast<UINT64>(particleUavStart_ + particleSource_) * srvSize_;
    auto particleOutput = srvHeap_->GetGPUDescriptorHandleForHeapStart();
    particleOutput.ptr += static_cast<UINT64>(particleUavStart_ + particleDestination) * srvSize_;
    commandList_->SetComputeRootDescriptorTable(1, particleInput);
    commandList_->SetComputeRootDescriptorTable(2, particleOutput);
    commandList_->Dispatch(ParticleCount / 64, 1, 1);
    D3D12_RESOURCE_BARRIER particleUavBarrier{};
    particleUavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    particleUavBarrier.UAV.pResource = particleBuffers_[particleDestination].Get();
    commandList_->ResourceBarrier(1, &particleUavBarrier);
    auto particleToRead = Transition(particleBuffers_[particleDestination].Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    commandList_->ResourceBarrier(1, &particleToRead);
    particleSource_ = particleDestination;

    if (shadowsEnabled_ && !materialGridEnabled_) {
        auto toDepthWrite = Transition(shadowMap_.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_DEPTH_WRITE);
        commandList_->ResourceBarrier(1, &toDepthWrite);
        D3D12_VIEWPORT shadowViewport{0, 0, static_cast<float>(ShadowMapSize),
            static_cast<float>(ShadowMapSize), 0, 1};
        D3D12_RECT shadowScissor{0, 0, static_cast<LONG>(ShadowMapSize), static_cast<LONG>(ShadowMapSize)};
        commandList_->RSSetViewports(1, &shadowViewport);
        commandList_->RSSetScissorRects(1, &shadowScissor);
        commandList_->SetPipelineState(shadowPipeline_.Get());
        commandList_->SetGraphicsRootSignature(shadowRootSignature_.Get());
        const D3D12_VERTEX_BUFFER_VIEW shadowViews[]{cubeVertexView_, shadowInstanceView_};
        commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList_->IASetVertexBuffers(0, 2, shadowViews);
        commandList_->IASetIndexBuffer(&cubeIndexView_);
        for (UINT cascade = 0; cascade < CascadeCount; ++cascade) {
            ShadowPassConstants constants{};
            XMStoreFloat4x4(&constants.lightViewProjection,
                XMMatrixTranspose(cascades.viewProjection[cascade]));
            memcpy(mappedShadowPass_[frameIndex_] + static_cast<size_t>(cascade) * sizeof(ShadowPassConstants),
                &constants, sizeof(constants));
            auto shadowDsv = dsvHeap_->GetCPUDescriptorHandleForHeapStart();
            shadowDsv.ptr += static_cast<SIZE_T>(1 + cascade) * dsvSize_;
            commandList_->ClearDepthStencilView(shadowDsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            commandList_->OMSetRenderTargets(0, nullptr, FALSE, &shadowDsv);
            commandList_->SetGraphicsRootConstantBufferView(0,
                shadowPassBuffers_[frameIndex_]->GetGPUVirtualAddress() +
                static_cast<UINT64>(cascade) * sizeof(ShadowPassConstants));
            commandList_->DrawIndexedInstanced(36, static_cast<UINT>(sceneObjects_.size() + 1), 0, 0, 0);
        }
        auto toShaderResource = Transition(shadowMap_.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        commandList_->ResourceBarrier(1, &toShaderResource);
        commandList_->RSSetViewports(1, &viewport);
        commandList_->RSSetScissorRects(1, &scissor);
    }

    // Geometry pass: the scene is rasterized once into albedo, normal and world-position targets.
    gBuffer_.BeginGeometryPass(commandList_.Get());
    gBuffer_.Clear(commandList_.Get());
    gBuffer_.Bind(commandList_.Get(), dsv);
    commandList_->SetGraphicsRootSignature(geometryRootSignature_.Get());
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList_->IASetVertexBuffers(0, 1, &vertexView_); commandList_->IASetIndexBuffer(&indexView_);
    XMMATRIX world = XMMatrixScaling(0.01f, 0.01f, 0.01f);
    if (showSponza_ && !materialGridEnabled_) {
    for (UINT i = 0; i < model_.submeshes.size(); ++i) {
        const auto& submesh = model_.submeshes[i];
        const auto& material = model_.materials[submesh.materialIndex];
        DrawConstants constants{};
        XMStoreFloat4x4(&constants.worldViewProjection, XMMatrixTranspose(world * view * projection));
        XMStoreFloat4x4(&constants.world, XMMatrixTranspose(world));
        constants.uvTransform = {1.0f, 1.0f, 0.0f, 0.0f};
        constants.materialColor = material.color;
        memcpy(mappedConstants_[frameIndex_] + static_cast<size_t>(i) * sizeof(DrawConstants), &constants, sizeof(constants));
        commandList_->SetGraphicsRootConstantBufferView(0,
            constantBuffers_[frameIndex_]->GetGPUVirtualAddress() + static_cast<UINT64>(i) * sizeof(DrawConstants));
        auto texture = srvHeap_->GetGPUDescriptorHandleForHeapStart();
        texture.ptr += static_cast<UINT64>(material.srvIndex) * srvSize_;
        commandList_->SetGraphicsRootDescriptorTable(1, texture);
        commandList_->DrawIndexedInstanced(submesh.indexCount, 1, submesh.firstIndex, 0, 0);
    }
    }

    BoundingFrustum viewFrustum;
    BoundingFrustum::CreateFromMatrix(viewFrustum, projection);
    BoundingFrustum worldFrustum;
    viewFrustum.Transform(worldFrustum, XMMatrixInverse(nullptr, view));
    visibleObjects_.clear();
    cullingTests_ = 0;
    if (!frustumCulling_) {
        visibleObjects_.resize(sceneObjects_.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(sceneObjects_.size()); ++i)
            visibleObjects_[i] = i;
    } else if (octreeCulling_) {
        octree_.Query(worldFrustum, sceneObjects_, visibleObjects_, cullingTests_);
    } else {
        for (uint32_t i = 0; i < static_cast<uint32_t>(sceneObjects_.size()); ++i) {
            ++cullingTests_;
            if (worldFrustum.Contains(sceneObjects_[i].Bounds()) != DISJOINT)
                visibleObjects_.push_back(i);
        }
    }

    auto* instances = reinterpret_cast<InstanceData*>(mappedInstances_[frameIndex_]);
    UINT instanceCount = 0;
    if (materialGridEnabled_) {
        constexpr UINT gridSize = 7;
        constexpr float spacing = 2.35f;
        for (UINT row = 0; row < gridSize; ++row) {
            for (UINT column = 0; column < gridSize; ++column) {
                const float metallic = static_cast<float>(row) / (gridSize - 1);
                const float roughness = 0.05f + 0.95f * static_cast<float>(column) / (gridSize - 1);
                instances[instanceCount++] = {
                    {(static_cast<float>(column) - 3.0f) * spacing,
                     8.0f + (static_cast<float>(row) - 3.0f) * spacing, 5.0f}, 0.0f,
                    {0.9f, 0.9f, 0.9f}, 0.0f, {0.82f, 0.24f, 0.07f, metallic}, roughness};
            }
        }
    } else {
        for (size_t i = 0; i < visibleObjects_.size(); ++i) {
            const SceneObject& object = sceneObjects_[visibleObjects_[i]];
            instances[instanceCount++] = {object.position, 0.0f,
                {object.scale, object.scale, object.scale}, 0.0f, object.color};
        }
        instances[instanceCount++] = {{0.0f, -0.6f, 30.0f}, 0.0f,
            {165.0f, 1.0f, 165.0f}, 0.0f, {0.32f, 0.34f, 0.38f, 0.0f}};
    }
    InstancePassConstants instancePass{};
    XMStoreFloat4x4(&instancePass.viewProjection, XMMatrixTranspose(view * projection));
    memcpy(mappedInstancePass_[frameIndex_], &instancePass, sizeof(instancePass));
    D3D12_VERTEX_BUFFER_VIEW instanceView{
        instanceBuffers_[frameIndex_]->GetGPUVirtualAddress(),
        static_cast<UINT>(instanceCount * sizeof(InstanceData)), sizeof(InstanceData)};
    const D3D12_VERTEX_BUFFER_VIEW instancingViews[]{
        materialGridEnabled_ ? sphereVertexView_ : cubeVertexView_, instanceView};
    commandList_->SetPipelineState(instancingPipeline_.Get());
    commandList_->SetGraphicsRootSignature(instancingRootSignature_.Get());
    commandList_->SetGraphicsRootConstantBufferView(0, instancePassBuffers_[frameIndex_]->GetGPUVirtualAddress());
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList_->IASetVertexBuffers(0, 2, instancingViews);
    commandList_->IASetIndexBuffer(materialGridEnabled_ ? &sphereIndexView_ : &cubeIndexView_);
    commandList_->DrawIndexedInstanced(materialGridEnabled_ ? sphereIndexCount_ : 36,
        instanceCount, 0, 0, 0);

    titleUpdateTime_ += delta;
    ++titleFrameCount_;
    if (titleUpdateTime_ >= 0.35f) {
        displayedFps_ = static_cast<float>(titleFrameCount_) / titleUpdateTime_;
        const wchar_t* mode = !frustumCulling_ ? L"OFF" : (octreeCulling_ ? L"OCTREE" : L"LINEAR");
        std::wostringstream title;
        const size_t culledCount = sceneObjects_.size() - visibleObjects_.size();
        title << L"D3D12 PBR + IBL | " << (materialGridEnabled_ ? L"MATERIAL GRID" : L"SCENE")
            << (materialGridEnabled_ ? L" (ROUGHNESS ->, METALLIC up)" : L"")
            << L" | IBL " << (iblEnabled_ ? L"ON" : L"OFF")
            << L" | VIGNETTE " << (vignetteEnabled_ ? L"ON" : L"OFF")
            << L" | OUTLINES " << (outlinesEnabled_ ? L"ON" : L"OFF")
            << L" | PARTICLES " << (particlesEnabled_ ? L"ON" : L"OFF")
            << L" (" << ParticleCount << L") | " << (shadowsEnabled_ ? L"SHADOWS ON" : L"SHADOWS OFF")
            << (visualizeCascades_ ? L" + CASCADES" : L"") << L" | " << mode << L" | submitted "
            << visibleObjects_.size() << L" / " << sceneObjects_.size()
            << L" | culled " << culledCount << L" | tests " << cullingTests_
            << L" | FPS " << static_cast<int>(displayedFps_ + 0.5f);
        SetWindowTextW(window_, title.str().c_str());
        titleUpdateTime_ = 0.0f;
        titleFrameCount_ = 0;
    }
    if (showTessellation_) {
        TessellationConstants tessellationConstants{};
        XMStoreFloat4x4(&tessellationConstants.worldViewProjection, XMMatrixTranspose(view * projection));
        XMStoreFloat4x4(&tessellationConstants.world, XMMatrixTranspose(XMMatrixIdentity()));
        tessellationConstants.eyePosition = {cameraPosition_.x, cameraPosition_.y, cameraPosition_.z, 1};
        tessellationConstants.parameters = {16.0f, 3.0f, 25.0f, 0.18f};
        tessellationConstants.options = {useNormalMap_ ? 1.0f : 0.0f,
            useDisplacement_ ? 1.0f : 0.0f, 0.0f, 0.0f};
        memcpy(mappedTessellation_[frameIndex_], &tessellationConstants, sizeof(tessellationConstants));
        commandList_->SetPipelineState(tessellationWireframe_ ?
            tessellationWireframePipeline_.Get() : tessellationPipeline_.Get());
        commandList_->SetGraphicsRootSignature(tessellationRootSignature_.Get());
        commandList_->SetGraphicsRootConstantBufferView(0,
            tessellationBuffers_[frameIndex_]->GetGPUVirtualAddress());
        auto tessellationTextures = srvHeap_->GetGPUDescriptorHandleForHeapStart();
        tessellationTextures.ptr += static_cast<UINT64>(tessellationSrvStart_) * srvSize_;
        commandList_->SetGraphicsRootDescriptorTable(1, tessellationTextures);
        commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);
        commandList_->IASetVertexBuffers(0, 1, &tessellationVertexView_);
        commandList_->IASetIndexBuffer(nullptr);
        commandList_->DrawInstanced(6, 1, 0, 0);
    }
    if (particlesEnabled_) {
        commandList_->SetPipelineState(particleRenderPipeline_.Get());
        commandList_->SetGraphicsRootSignature(particleRenderRootSignature_.Get());
        commandList_->SetGraphicsRootConstantBufferView(0,
            particleConstantBuffers_[frameIndex_]->GetGPUVirtualAddress());
        auto particleSrv = srvHeap_->GetGPUDescriptorHandleForHeapStart();
        particleSrv.ptr += static_cast<UINT64>(particleSrvStart_ + particleSource_) * srvSize_;
        commandList_->SetGraphicsRootDescriptorTable(1, particleSrv);
        commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
        commandList_->IASetVertexBuffers(0, 0, nullptr);
        commandList_->IASetIndexBuffer(nullptr);
        commandList_->DrawInstanced(ParticleCount, 1, 0, 0);
    }
    auto particleToWrite = Transition(particleBuffers_[particleSource_].Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    commandList_->ResourceBarrier(1, &particleToWrite);
    gBuffer_.EndGeometryPass(commandList_.Get());

    // Lighting pass: a full-screen triangle reads the G-buffer and evaluates all light types.
    auto toRender = Transition(backBuffers_[frameIndex_].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    commandList_->ResourceBarrier(1, &toRender);
    commandList_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    commandList_->SetPipelineState(lightingPipeline_.Get());
    commandList_->SetGraphicsRootSignature(lightingRootSignature_.Get());
    LightConstants lights{};
    lights.cameraPosition = {cameraPosition_.x, cameraPosition_.y, cameraPosition_.z, 1};
    lights.ambientColor = {0.055f, 0.065f, 0.09f, 1};
    lights.directionalDirectionIntensity = {0.3f, -0.85f, 0.25f, 0.65f};
    lights.directionalColor = {1.0f, 0.93f, 0.82f, 1};
    const std::array<XMFLOAT3, 8> positions{{
        {-6, 2.0f, -7}, {6, 2.0f, -7}, {-6, 2.0f, -1}, {6, 2.0f, -1},
        {-6, 2.0f, 5}, {6, 2.0f, 5}, {-3, 7.0f, 0}, {3, 7.0f, 0}}};
    const std::array<XMFLOAT3, 8> colors{{
        {1.0f, 0.18f, 0.08f}, {0.08f, 0.28f, 1.0f}, {1.0f, 0.55f, 0.08f}, {0.12f, 1.0f, 0.35f},
        {0.65f, 0.12f, 1.0f}, {0.05f, 0.8f, 1.0f}, {1.0f, 0.12f, 0.35f}, {0.3f, 0.45f, 1.0f}}};
    for (UINT i = 0; i < positions.size(); ++i) {
        lights.pointLights[i].positionRadius = {positions[i].x, positions[i].y, positions[i].z, 7.5f};
        lights.pointLights[i].colorIntensity = {colors[i].x, colors[i].y, colors[i].z, 3.2f};
    }
    lights.spotLights[0].positionRange = {0, 12.0f, -2.0f, 18.0f};
    lights.spotLights[0].directionCosOuter = {0, -1, 0.15f, cosf(XMConvertToRadians(28.0f))};
    lights.spotLights[0].colorIntensity = {1.0f, 0.9f, 0.68f, 5.0f};
    lights.spotLights[0].parameters = {cosf(XMConvertToRadians(18.0f)), 0, 0, 0};
    const UINT postProcessFlags = (vignetteEnabled_ ? 1u : 0u) | (outlinesEnabled_ ? 2u : 0u) |
        (iblEnabled_ ? 4u : 0u);
    lights.lightCountsAndMode = {static_cast<UINT>(positions.size()), 1, displayMode_, postProcessFlags};
    for (UINT cascade = 0; cascade < CascadeCount; ++cascade)
        XMStoreFloat4x4(&lights.shadowViewProjection[cascade],
            XMMatrixTranspose(cascades.viewProjection[cascade]));
    lights.cascadeSplits = {cascades.splitDistances[0], cascades.splitDistances[1],
        cascades.splitDistances[2], cascades.splitDistances[3]};
    lights.shadowParameters = {(shadowsEnabled_ && !materialGridEnabled_) ? 1.0f : 0.0f,
        visualizeCascades_ ? 1.0f : 0.0f, 0.0012f, 1.0f / ShadowMapSize};
    XMStoreFloat4(&lights.cameraForward, XMVectorSetW(XMVector3Normalize(lookDirection), 0.0f));
    memcpy(mappedLights_[frameIndex_], &lights, sizeof(lights));
    commandList_->SetGraphicsRootConstantBufferView(0, lightBuffers_[frameIndex_]->GetGPUVirtualAddress());
    commandList_->SetGraphicsRootDescriptorTable(1,
        gBuffer_.FirstSrv(srvHeap_->GetGPUDescriptorHandleForHeapStart(), srvSize_));
    auto shadowSrv = srvHeap_->GetGPUDescriptorHandleForHeapStart();
    shadowSrv.ptr += static_cast<UINT64>(shadowSrvIndex_) * srvSize_;
    commandList_->SetGraphicsRootDescriptorTable(2, shadowSrv);
    auto iblSrv = srvHeap_->GetGPUDescriptorHandleForHeapStart();
    iblSrv.ptr += static_cast<UINT64>(iblSrvStart_) * srvSize_;
    commandList_->SetGraphicsRootDescriptorTable(3, iblSrv);
    commandList_->IASetVertexBuffers(0, 0, nullptr);
    commandList_->IASetIndexBuffer(nullptr);
    commandList_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    commandList_->DrawInstanced(3, 1, 0, 0);
    auto toPresent = Transition(backBuffers_[frameIndex_].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    commandList_->ResourceBarrier(1, &toPresent);
    Check(commandList_->Close(), "Cannot close frame command list");
    ID3D12CommandList* lists[] = {commandList_.Get()}; queue_->ExecuteCommandLists(1, lists);
    Check(swapChain_->Present(1, 0), "Cannot present frame");
    MoveToNextFrame();
}

void RenderingSystem::Flush()
{
    const UINT64 value = ++fenceValues_[frameIndex_];
    Check(queue_->Signal(fence_.Get(), value), "Cannot signal fence");
    if (fence_->GetCompletedValue() < value) {
        Check(fence_->SetEventOnCompletion(value, fenceEvent_), "Cannot set fence event");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
}

void RenderingSystem::MoveToNextFrame()
{
    const UINT64 currentValue = ++fenceValues_[frameIndex_];
    Check(queue_->Signal(fence_.Get(), currentValue), "Cannot signal frame fence");
    frameIndex_ = swapChain_->GetCurrentBackBufferIndex();
    if (fence_->GetCompletedValue() < fenceValues_[frameIndex_]) {
        Check(fence_->SetEventOnCompletion(fenceValues_[frameIndex_], fenceEvent_), "Cannot wait for frame fence");
        WaitForSingleObject(fenceEvent_, INFINITE);
    }
    fenceValues_[frameIndex_] = currentValue;
}

int RenderingSystem::Run(HINSTANCE instance)
{
    Check(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "Cannot initialize COM");
    try {
        CreateWindowHandle(instance);
        CreateDeviceObjects();
        CreatePipeline();
        LoadScene();
        previousTime_ = std::chrono::steady_clock::now();
        MSG message{};
        while (message.message != WM_QUIT) {
            if (PeekMessage(&message, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&message); DispatchMessage(&message);
            } else Render();
        }
        Flush();
        for (auto& buffer : constantBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        for (auto& buffer : lightBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        for (auto& buffer : tessellationBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        for (auto& buffer : instanceBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        for (auto& buffer : instancePassBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        for (auto& buffer : shadowPassBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        for (auto& buffer : particleConstantBuffers_) if (buffer) buffer->Unmap(0, nullptr);
        CloseHandle(fenceEvent_);
        CoUninitialize();
        return static_cast<int>(message.wParam);
    } catch (const std::exception& e) {
        MessageBoxA(window_, e.what(), "PbrIblHomework error", MB_ICONERROR);
        if (fenceEvent_) CloseHandle(fenceEvent_);
        CoUninitialize();
        return 1;
    }
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int)
{
    RenderingSystem app;
    return app.Run(instance);
}
