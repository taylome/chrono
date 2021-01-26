// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2021 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Jason Zhou
// =============================================================================
//
// NASA VIPER Lunar Rover Model Class.
// This class contains model for NASA's VIPER lunar rover for NASA's 2024 Moon
// exploration mission.
//
// =============================================================================

#include <cmath>

#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChColorAsset.h"
#include "chrono/assets/ChCylinderShape.h"
#include "chrono/assets/ChSphereShape.h"
#include "chrono/assets/ChTexture.h"
#include "chrono/assets/ChTriangleMeshShape.h"

#include "chrono/motion_functions/ChFunction_Setpoint.h"

#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/physics/ChLinkMotorRotationSpeed.h"
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChSystemSMC.h"

#include "chrono_models/robot/viper/Viper.h"

namespace chrono {
namespace viper {

// =============================================================================
// Create default contact material for the rover
std::shared_ptr<ChMaterialSurface> DefaultContactMaterial(ChContactMethod contact_method) {
    float mu = 0.4f;   // coefficient of friction
    float cr = 0.0f;   // coefficient of restitution
    float Y = 2e7f;    // Young's modulus
    float nu = 0.3f;   // Poisson ratio
    float kn = 2e5f;   // normal stiffness
    float gn = 40.0f;  // normal viscous damping
    float kt = 2e5f;   // tangential stiffness
    float gt = 20.0f;  // tangential viscous damping

    switch (contact_method) {
        case ChContactMethod::NSC: {
            auto matNSC = chrono_types::make_shared<ChMaterialSurfaceNSC>();
            matNSC->SetFriction(mu);
            matNSC->SetRestitution(cr);
            return matNSC;
        }
        case ChContactMethod::SMC: {
            auto matSMC = chrono_types::make_shared<ChMaterialSurfaceSMC>();
            matSMC->SetFriction(mu);
            matSMC->SetRestitution(cr);
            matSMC->SetYoungModulus(Y);
            matSMC->SetPoissonRatio(nu);
            matSMC->SetKn(kn);
            matSMC->SetGn(gn);
            matSMC->SetKt(kt);
            matSMC->SetGt(gt);
            return matSMC;
        }
        default:
            return std::shared_ptr<ChMaterialSurface>();
    }
}

// Add a revolute joint between body_1 and body_2
// rel_joint_pos and rel_joint_rot are the position and the rotation of the revolute point
void AddRevoluteJoint(std::shared_ptr<ChBodyAuxRef> body_1,
                      std::shared_ptr<ChBodyAuxRef> body_2,
                      std::shared_ptr<ChBodyAuxRef> chassis,
                      ChSystem* system,
                      const ChVector<>& rel_joint_pos,
                      const ChQuaternion<>& rel_joint_rot) {
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(rel_joint_pos, rel_joint_rot);            // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                            // global -> child

    auto revo = chrono_types::make_shared<ChLinkLockRevolute>();
    revo->Initialize(body_1, body_2, ChCoordsys<>(X_GC.GetCoord().pos, X_GC.GetCoord().rot));
    system->AddLink(revo);
}

// Add a rotational speed controlled motor between body 'steer' and body 'wheel'
// rel_joint_pos and rel_joint_rot are the position and the rotation of the motor
std::shared_ptr<ChLinkMotorRotationSpeed> AddMotor(std::shared_ptr<ChBodyAuxRef> steer,
                                                   std::shared_ptr<ChBodyAuxRef> wheel,
                                                   std::shared_ptr<ChBodyAuxRef> chassis,
                                                   ChSystem* system,
                                                   const ChVector<>& rel_joint_pos,
                                                   const ChQuaternion<>& rel_joint_rot,
                                                   std::shared_ptr<ChFunction_Const> speed_func) {
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(rel_joint_pos, rel_joint_rot);            // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                            // global -> child

    auto motor_angle = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    motor_angle->Initialize(steer, wheel, ChFrame<>(X_GC.GetCoord().pos, X_GC.GetCoord().rot));
    system->AddLink(motor_angle);
    motor_angle->SetSpeedFunction(speed_func);
    return motor_angle;
}

// Add a spring between pos_1 and pos_2
// the default length of the spring is auto-adjusted
std::shared_ptr<ChLinkTSDA> AddSuspensionSpring(std::shared_ptr<ChBodyAuxRef> chassis,
                                                std::shared_ptr<ChBodyAuxRef> steer,
                                                ChSystem* system,
                                                const ChVector<>& pos_1,
                                                const ChVector<>& pos_2) {
    ChQuaternion<> ori = ChQuaternion<>(1, 0, 0, 0);
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();  // global -> parent

    ChFrame<> X_PC_1(pos_1, ori);      // parent -> child
    ChFrame<> X_GC_1 = X_GP * X_PC_1;  // global -> child

    ChFrame<> X_PC_2(pos_2, ori);      // parent -> child
    ChFrame<> X_GC_2 = X_GP * X_PC_2;  // global -> child

    std::shared_ptr<ChLinkTSDA> spring;
    spring = chrono_types::make_shared<ChLinkTSDA>();
    spring->Initialize(chassis, steer, false, X_GC_1.GetCoord().pos, X_GC_2.GetCoord().pos, true, 0.0);
    spring->SetSpringCoefficient(800000.0);
    spring->SetDampingCoefficient(10000.0);
    system->AddLink(spring);
    return spring;
}

// ===============================================================================
// Base class for all Viper Part
Viper_Part::Viper_Part(const std::string& name,
                       bool fixed,
                       std::shared_ptr<ChMaterialSurface> mat,
                       ChSystem* system,
                       const ChVector<>& body_pos,
                       const ChQuaternion<>& body_rot,
                       std::shared_ptr<ChBodyAuxRef> chassis_body,
                       bool collide) {
    m_body = std::shared_ptr<ChBodyAuxRef>(system->NewBodyAuxRef());
    m_body->SetNameString(name + "_body");
    m_chassis = chassis_body;
    m_mat = mat;
    m_pos = body_pos;
    m_rot = body_rot;
    m_system = system;
    m_collide = collide;
    m_fixed = fixed;
}

// Create Visulization assets -> Finer mesh
void Viper_Part::AddVisualizationAssets() {
    // Read mesh from the obj folder
    std::string vis_mesh_file = "viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), true, false);
    trimesh->Transform(m_offset, ChMatrix33<>(1));
    auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
    trimesh_shape->SetMesh(trimesh);
    trimesh_shape->SetName(m_mesh_name);
    trimesh_shape->SetStatic(true);
    m_body->AddAsset(trimesh_shape);
    return;
}

void Viper_Part::SetCollide(bool state) {
    m_collide = state;
}

// Add collision assets -> Rougher mesh
void Viper_Part::AddCollisionShapes() {
    // read mesh from the col folder
    std::string vis_mesh_file = "viper/col/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), true, false);
    trimesh->Transform(m_offset, ChMatrix33<>(1));
    auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
    trimesh_shape->SetMesh(trimesh);
    trimesh_shape->SetName(m_mesh_name);
    trimesh_shape->SetStatic(true);

    m_body->GetCollisionModel()->ClearModel();
    m_body->GetCollisionModel()->AddTriangleMesh(m_mat, trimesh, false, false, VNULL, ChMatrix33<>(1), 0.005);
    m_body->GetCollisionModel()->BuildModel();
    m_body->SetCollide(m_collide);
}

// =============================================================================
// Rover Chassis
Viper_Chassis::Viper_Chassis(const std::string& name,
                             bool fixed,
                             std::shared_ptr<ChMaterialSurface> mat,
                             ChSystem* system,
                             const ChVector<>& body_pos,
                             const ChQuaternion<>& body_rot,
                             bool collide)
    : Viper_Part(name, fixed, mat, system, body_pos, body_rot, NULL, collide) {
    m_mesh_name = "viper_chassis";
    m_offset = ChVector<>(0, 0, 0);
    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

void Viper_Chassis::Initialize() {
    std::string vis_mesh_file = "viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), false, false);
    trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(1));  // scale to a different size
    trimesh->RepairDuplicateVertexes(1e-9);                    // if meshes are not watertight

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    trimesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

    m_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    m_body->SetMass(mmass * m_density);
    m_body->SetInertiaXX(m_density * principal_I);
    m_body->SetFrame_REF_to_abs(ChFrame<>(m_pos, m_rot));
    m_body->SetBodyFixed(m_fixed);

    AddCollisionShapes();
    AddVisualizationAssets();

    m_system->Add(m_body);
}

void Viper_Chassis::SetCollide(bool state) {
    m_collide = state;
    m_body->SetCollide(state);
}

void Viper_Chassis::Translate(const ChVector<>& shift) {
    m_body->SetPos(m_body->GetPos() + shift);
}

// ==========================================================
// Viper Wheel
Viper_Wheel::Viper_Wheel(const std::string& name,
                         bool fixed,
                         std::shared_ptr<ChMaterialSurface> mat,
                         ChSystem* system,
                         const ChVector<>& body_pos,
                         const ChQuaternion<>& body_rot,
                         std::shared_ptr<ChBodyAuxRef> chassis,
                         bool collide)
    : Viper_Part(name, fixed, mat, system, body_pos, body_rot, chassis, collide) {
    m_mesh_name = "viper_wheel";
    m_offset = ChVector<>(0, 0, 0);
    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

void Viper_Wheel::Initialize() {
    std::string vis_mesh_file = "viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), false, false);
    trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(1));  // scale to a different size
    trimesh->RepairDuplicateVertexes(1e-9);                    // if meshes are not watertight

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    trimesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

    m_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    m_body->SetMass(mmass * m_density);
    m_body->SetInertiaXX(m_density * principal_I);

    // set relative position to chassis
    const ChFrame<>& X_GP = m_chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(m_pos, m_rot);                              // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                              // global -> child
    m_body->SetFrame_REF_to_abs(X_GC);
    m_body->SetBodyFixed(m_fixed);

    AddCollisionShapes();
    AddVisualizationAssets();

    m_system->Add(m_body);
}

void Viper_Wheel::SetCollide(bool state) {
    m_collide = state;
    m_body->SetCollide(state);
}

void Viper_Wheel::Translate(const ChVector<>& shift) {
    m_body->SetPos(m_body->GetPos() + shift);
}

// ==========================================================
// Viper Upper Suspension Arm
Viper_Up_Arm::Viper_Up_Arm(const std::string& name,
                           bool fixed,
                           std::shared_ptr<ChMaterialSurface> mat,
                           ChSystem* system,
                           const ChVector<>& body_pos,
                           const ChQuaternion<>& body_rot,
                           std::shared_ptr<ChBodyAuxRef> chassis,
                           bool collide,
                           const int& side)
    : Viper_Part(name, fixed, mat, system, body_pos, body_rot, chassis, collide) {
    if (side == 0) {
        m_mesh_name = "viper_L_up_sus";
    } else if (side == 1) {
        m_mesh_name = "viper_R_up_sus";
    }

    m_offset = ChVector<>(0, 0, 0);
    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

void Viper_Up_Arm::Initialize() {
    std::string vis_mesh_file = "viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), false, false);
    trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(1));  // scale to a different size
    trimesh->RepairDuplicateVertexes(1e-9);                    // if meshes are not watertight

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    trimesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

    m_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    m_body->SetMass(mmass * m_density);
    m_body->SetInertiaXX(m_density * principal_I);

    // set relative position to chassis
    const ChFrame<>& X_GP = m_chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(m_pos, m_rot);                              // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                              // global -> child
    m_body->SetFrame_REF_to_abs(X_GC);
    m_body->SetBodyFixed(m_fixed);

    AddCollisionShapes();
    AddVisualizationAssets();

    m_system->Add(m_body);
}

void Viper_Up_Arm::SetCollide(bool state) {
    m_collide = state;
    m_body->SetCollide(state);
}

void Viper_Up_Arm::Translate(const ChVector<>& shift) {
    m_body->SetPos(m_body->GetPos() + shift);
}

// ==========================================================
// Viper Bottom Suspension Arm
Viper_Bottom_Arm::Viper_Bottom_Arm(const std::string& name,
                           bool fixed,
                           std::shared_ptr<ChMaterialSurface> mat,
                           ChSystem* system,
                           const ChVector<>& body_pos,
                           const ChQuaternion<>& body_rot,
                           std::shared_ptr<ChBodyAuxRef> chassis,
                           bool collide,
                           const int& side)
    : Viper_Part(name, fixed, mat, system, body_pos, body_rot, chassis, collide) {
    if (side == 0) {
        m_mesh_name = "viper_L_bt_sus";
    } else if (side == 1) {
        m_mesh_name = "viper_R_bt_sus";
    }

    m_offset = ChVector<>(0, 0, 0);
    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

void Viper_Bottom_Arm::Initialize() {
    std::string vis_mesh_file = "viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), false, false);
    trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(1));  // scale to a different size
    trimesh->RepairDuplicateVertexes(1e-9);                    // if meshes are not watertight

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    trimesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

    m_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    m_body->SetMass(mmass * m_density);
    m_body->SetInertiaXX(m_density * principal_I);

    // set relative position to chassis
    const ChFrame<>& X_GP = m_chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(m_pos, m_rot);                              // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                              // global -> child
    m_body->SetFrame_REF_to_abs(X_GC);
    m_body->SetBodyFixed(m_fixed);

    AddCollisionShapes();
    AddVisualizationAssets();

    m_system->Add(m_body);
}

void Viper_Bottom_Arm::SetCollide(bool state) {
    m_collide = state;
    m_body->SetCollide(state);
}

void Viper_Bottom_Arm::Translate(const ChVector<>& shift) {
    m_body->SetPos(m_body->GetPos() + shift);
}

// ==========================================================
// Viper Steering Rod
Viper_Steer::Viper_Steer(const std::string& name,
                         bool fixed,
                         std::shared_ptr<ChMaterialSurface> mat,
                         ChSystem* system,
                         const ChVector<>& body_pos,
                         const ChQuaternion<>& body_rot,
                         std::shared_ptr<ChBodyAuxRef> chassis,
                         bool collide,
                         const int& side)
    : Viper_Part(name, fixed, mat, system, body_pos, body_rot, chassis, collide) {
    if (side == 0) {
        m_mesh_name = "viper_L_steer";
    } else if (side == 1) {
        m_mesh_name = "viper_R_steer";
    }

    m_offset = ChVector<>(0, 0, 0);
    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

void Viper_Steer::Initialize() {
    std::string vis_mesh_file = "viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), false, false);
    trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(1));  // scale to a different size
    trimesh->RepairDuplicateVertexes(1e-9);                    // if meshes are not watertight

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    trimesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

    m_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    m_body->SetMass(mmass * m_density);
    m_body->SetInertiaXX(m_density * principal_I);

    // set relative position to chassis
    const ChFrame<>& X_GP = m_chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(m_pos, m_rot);                              // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                              // global -> child
    m_body->SetFrame_REF_to_abs(X_GC);
    m_body->SetBodyFixed(m_fixed);

    AddCollisionShapes();
    AddVisualizationAssets();

    m_system->Add(m_body);
}

void Viper_Steer::SetCollide(bool state) {
    m_collide = state;
    m_body->SetCollide(state);
}

void Viper_Steer::Translate(const ChVector<>& shift) {
    m_body->SetPos(m_body->GetPos() + shift);
}

// ==========================================================
// Rover Class for the entire rover model
ViperRover::ViperRover(ChSystem* system, 
                    const ChVector<>& rover_pos, 
                    const ChQuaternion<>& rover_rot, 
                    std::shared_ptr<ChMaterialSurface> wheel_mat)
    : m_system(system), m_rover_pos(rover_pos), m_rover_rot(rover_rot), m_wheel_material(wheel_mat), m_custom_wheel_mat(true){
    Create();
}

ViperRover::ViperRover(ChSystem* system, 
                    const ChVector<>& rover_pos, 
                    const ChQuaternion<>& rover_rot)
    : m_system(system), m_rover_pos(rover_pos), m_rover_rot(rover_rot), m_custom_wheel_mat(false){
    Create();
}

ViperRover::~ViperRover() {}

void ViperRover::Create() {
    auto contact_method = m_system->GetContactMethod();

    // Set default collision model envelope commensurate with model dimensions.
    // Note that an SMC system automatically sets envelope to 0.
    if (contact_method == ChContactMethod::NSC) {
        collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.01);
        collision::ChCollisionModel::SetDefaultSuggestedMargin(0.005);
    }

    if(m_custom_wheel_mat == true){
        m_chassis_material = DefaultContactMaterial(contact_method);
        m_suspension_material = DefaultContactMaterial(contact_method);
        m_steer_material = DefaultContactMaterial(contact_method);
    }else{
        // Create the contact materials (all with default properties)
        m_chassis_material = DefaultContactMaterial(contact_method);
        m_wheel_material = DefaultContactMaterial(contact_method);
        m_suspension_material = DefaultContactMaterial(contact_method);
        m_steer_material = DefaultContactMaterial(contact_method);
    }


    // initialize rover chassis
    ChQuaternion<> body_rot;
    m_chassis = chrono_types::make_shared<Viper_Chassis>("chassis", false, m_chassis_material, m_system, m_rover_pos,
                                                         m_rover_rot, false);

    // initilize rover wheels
    double w_lx = 0.5618 + 0.08;
    double w_ly = 0.2067 + 0.32 + 0.0831;
    double w_lz = 0.0;
    ChVector<> wheel_rel_pos_lf = ChVector<>(w_lx, w_ly, w_lz);
    ChVector<> wheel_rel_pos_rf = ChVector<>(w_lx, -w_ly, w_lz);
    ChVector<> wheel_rel_pos_lb = ChVector<>(-w_lx, w_ly, w_lz);
    ChVector<> wheel_rel_pos_rb = ChVector<>(-w_lx, -w_ly, w_lz);
    {
        ChVector<> wheel_pos;
        ChQuaternion<> wheel_rot_ori = ChQuaternion<>(1, 0, 0, 0);
        ChQuaternion<> wheel_rot_ori_pi;
        wheel_rot_ori_pi = Q_from_Euler123(ChVector<double>(0, 0, CH_C_PI));

        for (int i = 0; i < 4; i++) {
            if (i == 0) {
                wheel_pos = wheel_rel_pos_lf;
            }
            if (i == 1) {
                wheel_pos = wheel_rel_pos_rf;
            }
            if (i == 2) {
                wheel_pos = wheel_rel_pos_lb;
            }
            if (i == 3) {
                wheel_pos = wheel_rel_pos_rb;
            }
            std::shared_ptr<Viper_Wheel> m_wheel;
            if (i == 1 || i == 3) {
                m_wheel = chrono_types::make_shared<Viper_Wheel>("wheel", false, m_wheel_material, m_system, wheel_pos,
                                                                 wheel_rot_ori, m_chassis->GetBody(), true);
            } else {
                m_wheel = chrono_types::make_shared<Viper_Wheel>("wheel", false, m_wheel_material, m_system, wheel_pos,
                                                                 wheel_rot_ori_pi, m_chassis->GetBody(), true);
            }
            m_wheels.push_back(m_wheel);
        }
    }

    // initialize rover up and bottom suspensions
    double cr_lx = 0.5618 + 0.08;
    double cr_ly = 0.2067;  // + 0.32/2;
    double cr_lz = 0.0525;
    ChVector<> cr_rel_pos_lf_lower = ChVector<>(cr_lx, cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_rf_lower = ChVector<>(cr_lx, -cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_lb_lower = ChVector<>(-cr_lx, cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_rb_lower = ChVector<>(-cr_lx, -cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_lf_upper = ChVector<>(cr_lx, cr_ly, cr_lz);
    ChVector<> cr_rel_pos_rf_upper = ChVector<>(cr_lx, -cr_ly, cr_lz);
    ChVector<> cr_rel_pos_lb_upper = ChVector<>(-cr_lx, cr_ly, cr_lz);
    ChVector<> cr_rel_pos_rb_upper = ChVector<>(-cr_lx, -cr_ly, cr_lz);
    {
        ChVector<> sus_rel_pos;
        ChQuaternion<> sus_rel_rot = ChQuaternion<>(1, 0, 0, 0);
        for (int i = 0; i < 8; i++) {
            if (i == 0) {
                sus_rel_pos = cr_rel_pos_lf_lower;
            }
            if (i == 1) {
                sus_rel_pos = cr_rel_pos_rf_lower;
            }
            if (i == 2) {
                sus_rel_pos = cr_rel_pos_lb_lower;
            }
            if (i == 3) {
                sus_rel_pos = cr_rel_pos_rb_lower;
            }
            if (i == 4) {
                sus_rel_pos = cr_rel_pos_lf_upper;
            }
            if (i == 5) {
                sus_rel_pos = cr_rel_pos_rf_upper;
            }
            if (i == 6) {
                sus_rel_pos = cr_rel_pos_lb_upper;
            }
            if (i == 7) {
                sus_rel_pos = cr_rel_pos_rb_upper;
            }
            int side = i % 2;
            if (i < 4) {
                std::shared_ptr<Viper_Bottom_Arm> m_bt_sus = chrono_types::make_shared<Viper_Bottom_Arm>(
                    "bt_sus", false, m_suspension_material, m_system, sus_rel_pos, sus_rel_rot, m_chassis->GetBody(),
                    false, side);
                m_bts_suss.push_back(m_bt_sus);
            } else {
                std::shared_ptr<Viper_Up_Arm> m_up_sus = chrono_types::make_shared<Viper_Up_Arm>(
                    "up_sus", false, m_suspension_material, m_system, sus_rel_pos, sus_rel_rot, m_chassis->GetBody(),
                    false, side);
                m_up_suss.push_back(m_up_sus);
            }
        }
    }

    // initialize steering rod
    double sr_lx = 0.5618 + 0.08;
    double sr_ly = 0.2067 + 0.32 + 0.0831;
    double sr_lz = 0.0;
    double sr_ly_joint = 0.2067 + 0.32;
    ChVector<> sr_rel_pos_lf = ChVector<>(sr_lx, sr_ly, sr_lz);
    ChVector<> sr_rel_pos_rf = ChVector<>(sr_lx, -sr_ly, sr_lz);
    ChVector<> sr_rel_pos_lb = ChVector<>(-sr_lx, sr_ly, sr_lz);
    ChVector<> sr_rel_pos_rb = ChVector<>(-sr_lx, -sr_ly, sr_lz);

    ChVector<> sr_rel_pos_lf_lower = ChVector<>(sr_lx, sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_rf_lower = ChVector<>(sr_lx, -sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_lb_lower = ChVector<>(-sr_lx, sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_rb_lower = ChVector<>(-sr_lx, -sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_lf_upper = ChVector<>(sr_lx, sr_ly_joint, cr_lz);
    ChVector<> sr_rel_pos_rf_upper = ChVector<>(sr_lx, -sr_ly_joint, cr_lz);
    ChVector<> sr_rel_pos_lb_upper = ChVector<>(-sr_lx, sr_ly_joint, cr_lz);
    ChVector<> sr_rel_pos_rb_upper = ChVector<>(-sr_lx, -sr_ly_joint, cr_lz);
    {
        ChVector<> steer_rel_pos;
        ChQuaternion<> steer_rel_rot = ChQuaternion<>(1, 0, 0, 0);
        for (int i = 0; i < 4; i++) {
            if (i == 0) {
                steer_rel_pos = sr_rel_pos_lf;
            }
            if (i == 1) {
                steer_rel_pos = sr_rel_pos_rf;
            }
            if (i == 2) {
                steer_rel_pos = sr_rel_pos_lb;
            }
            if (i == 3) {
                steer_rel_pos = sr_rel_pos_rb;
            }
            int side = i % 2;
            std::shared_ptr<Viper_Steer> m_steer =
                chrono_types::make_shared<Viper_Steer>("steering", false, m_steer_material, m_system, steer_rel_pos,
                                                       steer_rel_rot, m_chassis->GetBody(), false, side);
            m_steers.push_back(m_steer);
        }
    }
}

void ViperRover::Initialize() {
    m_chassis->Initialize();
    for (int i = 0; i < 4; i++) {
        m_wheels[i]->Initialize();
        m_up_suss[i]->Initialize();
        m_bts_suss[i]->Initialize();
        m_steers[i]->Initialize();
    }

    // add all constraints to the system
    // redefine pos data for constraints
    double sr_lx = 0.5618 + 0.08;
    double sr_ly = 0.2067 + 0.32 + 0.0831;
    double sr_lz = 0.0;
    double sr_ly_joint = 0.2067 + 0.32;

    double cr_lx = 0.5618 + 0.08;
    double cr_ly = 0.2067;  // + 0.32/2;
    double cr_lz = 0.0525;

    double w_lx = 0.5618 + 0.08;
    double w_ly = 0.2067 + 0.32 + 0.0831;
    double w_lz = 0.0;

    ChVector<> wheel_rel_pos_lf = ChVector<>(w_lx, w_ly, w_lz);
    ChVector<> wheel_rel_pos_rf = ChVector<>(w_lx, -w_ly, w_lz);
    ChVector<> wheel_rel_pos_lb = ChVector<>(-w_lx, w_ly, w_lz);
    ChVector<> wheel_rel_pos_rb = ChVector<>(-w_lx, -w_ly, w_lz);

    ChVector<> sr_rel_pos_lf = ChVector<>(sr_lx, sr_ly, sr_lz);
    ChVector<> sr_rel_pos_rf = ChVector<>(sr_lx, -sr_ly, sr_lz);
    ChVector<> sr_rel_pos_lb = ChVector<>(-sr_lx, sr_ly, sr_lz);
    ChVector<> sr_rel_pos_rb = ChVector<>(-sr_lx, -sr_ly, sr_lz);

    ChVector<> sr_rel_pos_lf_lower = ChVector<>(sr_lx, sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_rf_lower = ChVector<>(sr_lx, -sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_lb_lower = ChVector<>(-sr_lx, sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_rb_lower = ChVector<>(-sr_lx, -sr_ly_joint, -cr_lz);
    ChVector<> sr_rel_pos_lf_upper = ChVector<>(sr_lx, sr_ly_joint, cr_lz);
    ChVector<> sr_rel_pos_rf_upper = ChVector<>(sr_lx, -sr_ly_joint, cr_lz);
    ChVector<> sr_rel_pos_lb_upper = ChVector<>(-sr_lx, sr_ly_joint, cr_lz);
    ChVector<> sr_rel_pos_rb_upper = ChVector<>(-sr_lx, -sr_ly_joint, cr_lz);

    ChVector<> cr_rel_pos_lf_lower = ChVector<>(cr_lx, cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_rf_lower = ChVector<>(cr_lx, -cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_lb_lower = ChVector<>(-cr_lx, cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_rb_lower = ChVector<>(-cr_lx, -cr_ly, -cr_lz);
    ChVector<> cr_rel_pos_lf_upper = ChVector<>(cr_lx, cr_ly, cr_lz);
    ChVector<> cr_rel_pos_rf_upper = ChVector<>(cr_lx, -cr_ly, cr_lz);
    ChVector<> cr_rel_pos_lb_upper = ChVector<>(-cr_lx, cr_ly, cr_lz);
    ChVector<> cr_rel_pos_rb_upper = ChVector<>(-cr_lx, -cr_ly, cr_lz);

    {
        for (int i = 0; i < 4; i++) {
            // pick up relative position of wheel and conecting rod to the body
            ChVector<> CR_Rel_pos_Lower;
            ChVector<> CR_Rel_pos_Upper;
            ChVector<> SR_Rel_pos_Lower;
            ChVector<> SR_Rel_pos_Upper;
            ChVector<> Wheel_Rel_pos;
            if (i == 0) {
                Wheel_Rel_pos = wheel_rel_pos_lf;
                CR_Rel_pos_Lower = cr_rel_pos_lf_lower;
                CR_Rel_pos_Upper = cr_rel_pos_lf_upper;
                SR_Rel_pos_Lower = sr_rel_pos_lf_lower;
                SR_Rel_pos_Upper = sr_rel_pos_lf_upper;
            }
            if (i == 1) {
                Wheel_Rel_pos = wheel_rel_pos_rf;
                CR_Rel_pos_Lower = cr_rel_pos_rf_lower;
                CR_Rel_pos_Upper = cr_rel_pos_rf_upper;
                SR_Rel_pos_Lower = sr_rel_pos_rf_lower;
                SR_Rel_pos_Upper = sr_rel_pos_rf_upper;
            }
            if (i == 2) {
                Wheel_Rel_pos = wheel_rel_pos_lb;
                CR_Rel_pos_Lower = cr_rel_pos_lb_lower;
                CR_Rel_pos_Upper = cr_rel_pos_lb_upper;
                SR_Rel_pos_Lower = sr_rel_pos_lb_lower;
                SR_Rel_pos_Upper = sr_rel_pos_lb_upper;
            }
            if (i == 3) {
                Wheel_Rel_pos = wheel_rel_pos_rb;
                CR_Rel_pos_Lower = cr_rel_pos_rb_lower;
                CR_Rel_pos_Upper = cr_rel_pos_rb_upper;
                SR_Rel_pos_Lower = sr_rel_pos_rb_lower;
                SR_Rel_pos_Upper = sr_rel_pos_rb_upper;
            }

            ChQuaternion<> z2x;
            z2x.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(0, 1, 0));

            AddRevoluteJoint(m_chassis->GetBody(), m_bts_suss[i]->GetBody(), m_chassis->GetBody(), m_system,
                             CR_Rel_pos_Lower, z2x);
            AddRevoluteJoint(m_chassis->GetBody(), m_up_suss[i]->GetBody(), m_chassis->GetBody(), m_system,
                             CR_Rel_pos_Upper, z2x);
            AddRevoluteJoint(m_bts_suss[i]->GetBody(), m_steers[i]->GetBody(), m_chassis->GetBody(), m_system,
                             SR_Rel_pos_Lower, z2x);
            AddRevoluteJoint(m_up_suss[i]->GetBody(), m_steers[i]->GetBody(), m_chassis->GetBody(), m_system,
                             SR_Rel_pos_Upper, z2x);

            ChQuaternion<> z2y;
            z2y.Q_from_AngAxis(CH_C_PI / 2, ChVector<>(1, 0, 0));

            // initialize and store the const speed function
            auto const_speed_function = chrono_types::make_shared<ChFunction_Const>(CH_C_PI);  // speed w=3.145 rad/sec

            m_motors_func.push_back(const_speed_function);

            m_motors.push_back(AddMotor(m_steers[i]->GetBody(), m_wheels[i]->GetBody(), m_chassis->GetBody(), m_system,
                                        Wheel_Rel_pos, z2y, const_speed_function));

            // after motors are set, get speed function and store then in private fields
            m_sus_springs.push_back(AddSuspensionSpring(m_chassis->GetBody(), m_steers[i]->GetBody(), m_system,
                                                        CR_Rel_pos_Upper, SR_Rel_pos_Lower));
        }
    }
}

void ViperRover::SetMotorSpeed(double rad_speed, WheelID id) {
    m_motors_func[id]->Set_yconst(rad_speed);
}

ChVector<> ViperRover::GetWheelSpeed(WheelID id) {
    return m_wheels[id]->GetBody()->GetPos_dt();
}

ChQuaternion<> ViperRover::GetWheelAngVel(WheelID id) {
    return m_wheels[id]->GetBody()->GetRot_dt();
}

ChVector<> ViperRover::GetWheelContactForce(WheelID id) {
    return m_wheels[id]->GetBody()->GetContactForce();
}

ChVector<> ViperRover::GetWheelContactTorque(WheelID id) {
    return m_wheels[id]->GetBody()->GetContactTorque();
}

ChVector<> ViperRover::GetWheelAppliedForce(WheelID id) {
    return m_wheels[id]->GetBody()->GetAppliedForce();
}

ChVector<> ViperRover::GetWheelAppliedTorque(WheelID id) {
    return m_wheels[id]->GetBody()->GetAppliedTorque();
}

std::shared_ptr<ChBodyAuxRef> ViperRover::GetWheelBody(WheelID id) {
    return m_wheels[id]->GetBody();
}

std::shared_ptr<ChBodyAuxRef> ViperRover::GetChassisBody() {
    return m_chassis->GetBody();
}

double ViperRover::GetRoverMass() {
    double tot_mass = 0.0;
    for (int i = 0; i < 4; i++) {
        tot_mass = tot_mass + m_wheels[i]->GetBody()->GetMass();
        tot_mass = tot_mass + m_up_suss[i]->GetBody()->GetMass();
        tot_mass = tot_mass + m_bts_suss[i]->GetBody()->GetMass();
        tot_mass = tot_mass + m_steers[i]->GetBody()->GetMass();
    }
    tot_mass = tot_mass + m_chassis->GetBody()->GetMass();
    return tot_mass;
}

double ViperRover::GetWheelMass() {
    return m_wheels[0]->GetBody()->GetMass();
}

}  // namespace viper
}  // namespace chrono
