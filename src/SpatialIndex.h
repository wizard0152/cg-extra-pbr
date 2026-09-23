#pragma once

#include <DirectXCollision.h>
#include <DirectXMath.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct SceneObject
{
    DirectX::XMFLOAT3 position{};
    float scale = 1.0f;
    DirectX::XMFLOAT4 color{1, 1, 1, 1};

    DirectX::BoundingBox Bounds() const
    {
        const float half = scale * 0.5f;
        return {position, {half, half, half}};
    }
};

class Octree
{
public:
    void Build(const std::vector<SceneObject>& objects, const DirectX::BoundingBox& bounds);
    void Query(const DirectX::BoundingFrustum& frustum, const std::vector<SceneObject>& objects,
        std::vector<uint32_t>& visibleObjects, uint32_t& boundingVolumeTests) const;

private:
    struct Node
    {
        DirectX::BoundingBox bounds{};
        std::vector<uint32_t> objects;
        std::array<std::unique_ptr<Node>, 8> children;
    };

    std::unique_ptr<Node> root_;

    static void Split(Node& node, const std::vector<SceneObject>& objects, uint32_t depth);
    static void QueryNode(const Node& node, const DirectX::BoundingFrustum& frustum,
        const std::vector<SceneObject>& objects, std::vector<uint32_t>& visibleObjects,
        uint32_t& boundingVolumeTests);
    static void AppendSubtree(const Node& node, std::vector<uint32_t>& visibleObjects);
};
