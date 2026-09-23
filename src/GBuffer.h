#pragma once

#include <array>
#include <wrl.h>
#include <d3d12.h>

class GBuffer
{
public:
    static constexpr UINT TargetCount = 3;

    void Create(ID3D12Device* device, UINT width, UINT height,
        D3D12_CPU_DESCRIPTOR_HANDLE srvHeapStart, UINT srvDescriptorSize, UINT firstSrvIndex);
    void BeginGeometryPass(ID3D12GraphicsCommandList* commandList);
    void EndGeometryPass(ID3D12GraphicsCommandList* commandList);
    void Clear(ID3D12GraphicsCommandList* commandList);
    void Bind(ID3D12GraphicsCommandList* commandList, D3D12_CPU_DESCRIPTOR_HANDLE depthStencil);

    D3D12_GPU_DESCRIPTOR_HANDLE FirstSrv(
        D3D12_GPU_DESCRIPTOR_HANDLE srvHeapStart, UINT srvDescriptorSize) const;
    const std::array<DXGI_FORMAT, TargetCount>& Formats() const { return formats_; }

private:
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, TargetCount> targets_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtvHeap_;
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, TargetCount> rtvs_{};
    std::array<DXGI_FORMAT, TargetCount> formats_{
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R16G16B16A16_FLOAT};
    UINT firstSrvIndex_ = 0;
};

