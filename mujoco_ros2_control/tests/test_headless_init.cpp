// Copyright (C) 2026 ros2_control Development Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Julia Jia

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#include <hardware_interface/version.h>
#include <hardware_interface/hardware_info.hpp>
#include <mujoco/mujoco.h>
#include <mujoco_ros2_control/mujoco_system_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#define ROS_DISTRO_HUMBLE (HARDWARE_INTERFACE_VERSION_MAJOR < 3)

class HeadlessInitTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok())
    {
      rclcpp::init(0, nullptr);
    }

    // Create a simple MuJoCo model
    create_test_model();

    // Initialize hardware interface
    hardware_info_ = create_hardware_info();
    interface_ = std::make_shared<mujoco_ros2_control::MujocoSystemInterface>();
  }

  void TearDown() override
  {
    // Deactivate interface before destroying to stop threads cleanly
    if (interface_)
    {
      rclcpp_lifecycle::State inactive_state(0, "inactive");
      interface_->on_deactivate(inactive_state);
      interface_.reset();
    }

    // Clean up ROS
    rclcpp::shutdown();

    // Clean up test file
    if (std::filesystem::exists(test_model_path_))
    {
      std::filesystem::remove(test_model_path_);
    }
  }

  void create_test_model()
  {
    // Create a simple MuJoCo XML with two bodies and a floor
    test_model_path_ = "/tmp/test_headless_init_model.xml";
    std::ofstream file(test_model_path_);
    file << R"(<?xml version="1.0"?>
<mujoco model="test_headless_init">
  <option timestep="0.002"/>

  <size nconmax="100"/>

  <worldbody>
    <!-- Floor as infinite plane directly in worldbody (static by default) -->
    <geom name="floor_geom" type="plane" size="0 0 1"
          contype="1" conaffinity="1"/>

    <body name="box1" pos="0 0 0.1">
      <freejoint name="box1_joint"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="box1_geom" type="box" size="0.05 0.05 0.05"
            contype="1" conaffinity="1" friction="0.6"/>
    </body>

    <body name="box2" pos="0.2 0 0.1">
      <freejoint/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom name="box2_geom" type="box" size="0.05 0.05 0.05"
            contype="1" conaffinity="1" friction="0.6"/>
    </body>
  </worldbody>
</mujoco>
)";
    file.close();
  }

  hardware_interface::HardwareInfo create_hardware_info()
  {
    hardware_interface::HardwareInfo info;
    info.name = "test_mujoco";
    info.type = "system";
    info.hardware_parameters["mujoco_model"] = test_model_path_;
    info.hardware_parameters["meshdir"] = "";
    info.hardware_parameters["headless"] = "true";           // Enable headless mode for CI compatibility
    info.hardware_parameters["disable_rendering"] = "true";  // Disable cameras/lidar to avoid OpenGL issues in tests

    return info;
  }

  hardware_interface::CallbackReturn init_interface()
  {
#if ROS_DISTRO_HUMBLE
    return interface_->on_init(hardware_info_);
#else
    hardware_interface::HardwareComponentInterfaceParams params;
    params.hardware_info = hardware_info_;
    return interface_->on_init(params);
#endif
  }

  void wait_for_simulation(mjModel*& model, mjData*& data, std::chrono::seconds timeout = std::chrono::seconds(1))
  {
    model = nullptr;
    data = nullptr;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
    {
      interface_->get_model(model);
      interface_->get_data(data);
      if (model != nullptr && data != nullptr)
      {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  std::string test_model_path_;
  hardware_interface::HardwareInfo hardware_info_;
  std::shared_ptr<mujoco_ros2_control::MujocoSystemInterface> interface_;
};

TEST_F(HeadlessInitTest, HeadlessInitialization)
{
  ASSERT_EQ(init_interface(), hardware_interface::CallbackReturn::SUCCESS);

  mjModel* test_model = nullptr;
  mjData* test_data = nullptr;
  wait_for_simulation(test_model, test_data);
  ASSERT_NE(test_model, nullptr) << "Model failed to initialize within timeout";
  ASSERT_NE(test_data, nullptr) << "Model failed to initialize within timeout";

  auto state_interfaces = interface_->export_state_interfaces();
  EXPECT_GE(state_interfaces.size(), 0);
}

// Reproduces ros2_control_node crash on headless launch when MJCF has no cameras:
// on_activate() starts MujocoCameras offscreen GLFW/OpenGL even with cameras_.empty().
// Without the cameras_.empty() guard in mujoco_cameras.cpp, this often aborts (gladLoadGL).
// Matches CI/headless pipelines that only exercised on_init before (test_headless_init).
TEST_F(HeadlessInitTest, HeadlessActivateWithoutCameras)
{
  ASSERT_EQ(init_interface(), hardware_interface::CallbackReturn::SUCCESS);

  mjModel* test_model = nullptr;
  mjData* test_data = nullptr;
  wait_for_simulation(test_model, test_data);
  ASSERT_NE(test_model, nullptr);
  ASSERT_NE(test_data, nullptr);
  ASSERT_EQ(test_model->ncam, 0) << "Fixture must be camera-free to reproduce empty-camera init path";

  const rclcpp_lifecycle::State inactive_state(0, "inactive");
  ASSERT_EQ(interface_->on_activate(inactive_state), hardware_interface::CallbackReturn::SUCCESS);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  ASSERT_EQ(interface_->read(rclcpp::Time(0), rclcpp::Duration::from_seconds(0.02)),
            hardware_interface::return_type::OK);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  return result;
}
