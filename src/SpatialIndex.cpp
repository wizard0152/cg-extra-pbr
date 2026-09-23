#include "SpatialIndex.h"

#include <cmath>

using namespace DirectX;

namespace
{
constexpr uint32_t MaxObjectsPerNode = 32;
constexpr uint32_t MaxDepth = 6;

BoundingBox ChildBounds(const BoundingBox& parent, uint32_t childIndex)
{
    const XMFLOAT3 childExtents{
        parent.Extents.x * 0.5f,
        parent.Extents.y * 0.5f,
        parent.Extents.z * 0.5f};
    const XMFLOAT3 childCenter{
        parent.Center.x + ((childIndex & 1) ? childExtents.x : -childExtents.x),
        parent.Center.y + ((childIndex & 2) ? childExtents.y : -childExtents.y),
        parent.Center.z + ((childIndex & 4) ? childExtents.z : -childExtents.z)};
    return {childCenter, childExtents};
}

bool FitsCompletely(const BoundingBox& object, const BoundingBox& container)
{
    return std::abs(object.Center.x - container.Center.x) + object.Extents.x <= container.Extents.x &&
        std::abs(object.Center.y - container.Center.y) + object.Extents.y <= container.Extents.y &&
        std::abs(object.Center.z - container.Center.z) + object.Extents.z <= container.Extents.z;
}
}

void Octree::Build(const std::vector<SceneObject>& objects, const BoundingBox& bounds)
{
    root_ = std::make_unique<Node>();
    root_->bounds = bounds;
    root_->objects.resize(objects.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(objects.size()); ++i)
        root_->objects[i] = i;
    Split(*root_, objects, 0);
}

void Octree::Split(Node& node, const std::vector<SceneObject>& objects, uint32_t depth)
{
    if (depth >= MaxDepth || node.objects.size() <= MaxObjectsPerNode)
        return;

    std::array<std::vector<uint32_t>, 8> childObjects;
    std::vector<uint32_t> remaining;
    remaining.reserve(node.objects.size());

    for (uint32_t objectIndex : node.objects) {
        const BoundingBox objectBounds = objects[objectIndex].Bounds();
        bool assigned = false;
        for (uint32_t childIndex = 0; childIndex < 8; ++childIndex) {
            const BoundingBox childBounds = ChildBounds(node.bounds, childIndex);
            if (FitsCompletely(objectBounds, childBounds)) {
                childObjects[childIndex].push_back(objectIndex);
                assigned = true;
                break;
            }
        }
        if (!assigned)
            remaining.push_back(objectIndex);
    }
    node.objects = std::move(remaining);

    for (uint32_t childIndex = 0; childIndex < 8; ++childIndex) {
        if (childObjects[childIndex].empty())
            continue;
        node.children[childIndex] = std::make_unique<Node>();
        node.children[childIndex]->bounds = ChildBounds(node.bounds, childIndex);
        node.children[childIndex]->objects = std::move(childObjects[childIndex]);
        Split(*node.children[childIndex], objects, depth + 1);
    }
}

void Octree::Query(const BoundingFrustum& frustum, const std::vector<SceneObject>& objects,
    std::vector<uint32_t>& visibleObjects, uint32_t& boundingVolumeTests) const
{
    visibleObjects.clear();
    boundingVolumeTests = 0;
    if (root_)
        QueryNode(*root_, frustum, objects, visibleObjects, boundingVolumeTests);
}

void Octree::QueryNode(const Node& node, const BoundingFrustum& frustum,
    const std::vector<SceneObject>& objects, std::vector<uint32_t>& visibleObjects,
    uint32_t& boundingVolumeTests)
{
    ++boundingVolumeTests;
    const ContainmentType nodeContainment = frustum.Contains(node.bounds);
    if (nodeContainment == DISJOINT)
        return;
    if (nodeContainment == CONTAINS) {
        AppendSubtree(node, visibleObjects);
        return;
    }

    for (uint32_t objectIndex : node.objects) {
        ++boundingVolumeTests;
        if (frustum.Contains(objects[objectIndex].Bounds()) != DISJOINT)
            visibleObjects.push_back(objectIndex);
    }
    for (const auto& child : node.children)
        if (child)
            QueryNode(*child, frustum, objects, visibleObjects, boundingVolumeTests);
}

void Octree::AppendSubtree(const Node& node, std::vector<uint32_t>& visibleObjects)
{
    visibleObjects.insert(visibleObjects.end(), node.objects.begin(), node.objects.end());
    for (const auto& child : node.children)
        if (child)
            AppendSubtree(*child, visibleObjects);
}
