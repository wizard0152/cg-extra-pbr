#include "GBuffer.h"
#include <stdexcept>

using Microsoft::WRL::ComPtr;

static void GBufferCheck(HRESULT hr, const char* message)
{
    if (FAILED(hr)) throw std::runtime_error(message);
}

static D3D12_HEAP_PROPERTIES DefaultHeap()
{
    D3D12_HEAP_PROPERTIES result{};
    result.Type = D3D12_HEAP_TYPE_DEFAULT;
    result.CreationNodeMask = result.VisibleNodeMask = 1;
    return result;
}

void GBuffer::Create(ID3D12Device* device, UINT width, UINT height,
    D3D12_CPU_DESCRIPTOR_HANDLE srvHeapStart, UINT srvDescriptorSize, UINT firstSrvIndex)
{
    firstSrvIndex_ = firstSrvIndex;
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heapDesc.NumDescriptors = TargetCount;
    GBufferCheck(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&rtvHeap_)), "Cannot create G-buffer RTV heap");
    const UINT rtvSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    auto heap = DefaultHeap();
    for (UINT i = 0; i < TargetCount; ++i) {
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = formats_[i];
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = formats_[i];
        clear.Color[3] = i == 2 ? 0.0f : 1.0f;
        GBufferCheck(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &clear, IID_PPV_ARGS(&targets_[i])), "Cannot create G-buffer texture");

        rtvs_[i] = rtvHeap_->GetCPUDescriptorHandleForHeapStart();
        rtvs_[i].ptr += static_cast<SIZE_T>(i) * rtvSize;
        device->CreateRenderTargetView(targets_[i].Get(), nullptr, rtvs_[i]);

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Format = formats_[i];
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;
        auto srvHandle = srvHeapStart;
        srvHandle.ptr += static_cast<SIZE_T>(firstSrvIndex + i) * srvDescriptorSize;
        device->CreateShaderResourceView(targets_[i].Get(), &srv, srvHandle);
    }
}

void GBuffer::BeginGeometryPass(ID3D12GraphicsCommandList* commandList)
{
    std::array<D3D12_RESOURCE_BARRIER, TargetCount> barriers{};
    for (UINT i = 0; i < TargetCount; ++i) {
        barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[i].Transition.pResource = targets_[i].Get();
        barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }
    commandList->ResourceBarrier(TargetCount, barriers.data());
}

void GBuffer::EndGeometryPass(ID3D12GraphicsCommandList* commandList)
{
    std::array<D3D12_RESOURCE_BARRIER, TargetCount> barriers{};
    for (UINT i = 0; i < TargetCount; ++i) {
        barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[i].Transition.pResource = targets_[i].Get();
        barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    }
    commandList->ResourceBarrier(TargetCount, barriers.data());
}

void GBuffer::Clear(ID3D12GraphicsCommandList* commandList)
{
    const float albedo[4] = {0, 0, 0, 1};
    const float empty[4] = {0, 0, 0, 0};
    commandList->ClearRenderTargetView(rtvs_[0], albedo, 0, nullptr);
    commandList->ClearRenderTargetView(rtvs_[1], empty, 0, nullptr);
    commandList->ClearRenderTargetView(rtvs_[2], empty, 0, nullptr);
}

void GBuffer::Bind(ID3D12GraphicsCommandList* commandList, D3D12_CPU_DESCRIPTOR_HANDLE depthStencil)
{
    commandList->OMSetRenderTargets(TargetCount, rtvs_.data(), FALSE, &depthStencil);
}

D3D12_GPU_DESCRIPTOR_HANDLE GBuffer::FirstSrv(
    D3D12_GPU_DESCRIPTOR_HANDLE srvHeapStart, UINT srvDescriptorSize) const
{
    srvHeapStart.ptr += static_cast<UINT64>(firstSrvIndex_) * srvDescriptorSize;
    return srvHeapStart;
}

