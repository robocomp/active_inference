#pragma once

#include <Eigen/Core>

#include <string>

struct GraphObjectBox
{
    Eigen::Vector3f min = Eigen::Vector3f::Zero();
    Eigen::Vector3f max = Eigen::Vector3f::Zero();
    std::string category;
};