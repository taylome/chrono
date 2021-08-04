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
//
// RADU TODO:
// - Forces and torques are reported relative to the part's centroidal frame.
//   Likely confusing for a user since all bodies are ChBodyAuxRef!
//
// =============================================================================

#include <cmath>

#include "chrono/assets/ChBoxShape.h"
#include "chrono/physics/ChBodyEasy.h"
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

#include "chrono/physics/ChInertiaUtils.h"

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
void AddRevoluteJoint(std::shared_ptr<ChBody> body_1,
                      std::shared_ptr<ChBody> body_2,
                      std::shared_ptr<ChBodyAuxRef> chassis,
                      ChSystem* system,
                      const ChVector<>& rel_joint_pos,
                      const ChQuaternion<>& rel_joint_rot) {
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(rel_joint_pos, rel_joint_rot);            // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                            // global -> child

    auto revo = chrono_types::make_shared<ChLinkLockRevolute>();
    revo->Initialize(body_1, body_2, ChCoordsys<>(X_GC.GetPos(), X_GC.GetRot()));
    system->AddLink(revo);
}

// Add a rotational speed controlled motor between body 'steer' and body 'wheel'
// rel_joint_pos and rel_joint_rot are the position and the rotation of the motor
std::shared_ptr<ChLinkMotorRotationSpeed> AddMotor(std::shared_ptr<ChBody> steer,
                                                   std::shared_ptr<ChBody> wheel,
                                                   std::shared_ptr<ChBodyAuxRef> chassis,
                                                   ChSystem* system,
                                                   const ChVector<>& rel_joint_pos,
                                                   const ChQuaternion<>& rel_joint_rot,
                                                   std::shared_ptr<ChFunction_Const> speed_func) {
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_PC(rel_joint_pos, rel_joint_rot);            // parent -> child
    ChFrame<> X_GC = X_GP * X_PC;                            // global -> child

    auto motor_angle = chrono_types::make_shared<ChLinkMotorRotationSpeed>();
    motor_angle->Initialize(steer, wheel, X_GC);
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
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();
    auto p1 = X_GP.TransformPointLocalToParent(pos_1);
    auto p2 = X_GP.TransformPointLocalToParent(pos_2);

    std::shared_ptr<ChLinkTSDA> spring;
    spring = chrono_types::make_shared<ChLinkTSDA>();
    spring->Initialize(chassis, steer, false, p1, p2, true, 0.0);
    spring->SetSpringCoefficient(800000.0);
    spring->SetDampingCoefficient(10000.0);
    system->AddLink(spring);
    return spring;
}

// ===============================================================================
// Base class for all Viper Part
ViperPart::ViperPart(const std::string& name,
                     const ChFrame<>& rel_pos,
                     std::shared_ptr<ChMaterialSurface> mat,
                     bool collide)
    : m_name(name), m_pos(rel_pos), m_density(200), m_mat(mat), m_collide(collide), m_visualize(true) {}

void ViperPart::Construct(ChSystem* system) {
    m_body = std::shared_ptr<ChBodyAuxRef>(system->NewBodyAuxRef());
    m_body->SetNameString(m_name + "_body");

    // Load geometry mesh
    std::string vis_mesh_file = "robot/viper/obj/" + m_mesh_name + ".obj";
    auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
    trimesh->LoadWavefrontMesh(GetChronoDataFile(vis_mesh_file), false, false);
    trimesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(1));  // scale to a different size
    ////trimesh->RepairDuplicateVertexes(1e-9);                    // if meshes are not watertight

    // Calculate and set intertia properties
    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    trimesh->ComputeMassProperties(true, mmass, mcog, minertia);

    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);

    m_body->SetMass(mmass * m_density);
    m_body->SetInertiaXX(m_density * principal_I);
    m_body->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Add visualization shape
    if (m_visualize) {
        auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
        trimesh_shape->SetMesh(trimesh);
        trimesh_shape->SetName(m_mesh_name);
        trimesh_shape->SetStatic(true);
        m_body->AddAsset(trimesh_shape);
    }

    // Add collision shape
    if (m_collide) {
        std::string col_mesh_file = "robot/viper/col/" + m_mesh_name + ".obj";
        auto trimesh_c = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
        trimesh_c->LoadWavefrontMesh(GetChronoDataFile(col_mesh_file), true, false);

        m_body->GetCollisionModel()->ClearModel();
        m_body->GetCollisionModel()->AddTriangleMesh(m_mat, trimesh_c, false, false, VNULL, ChMatrix33<>(1), 0.005);
        m_body->GetCollisionModel()->BuildModel();
        m_body->SetCollide(m_collide);
    }

    system->AddBody(m_body);
}

void ViperPart::SetVisualize(bool state) {
    m_visualize = state;
}

void ViperPart::SetCollide(bool state) {
    m_collide = state;
}

void ViperPart::Initialize(std::shared_ptr<ChBodyAuxRef> chassis) {
    Construct(chassis->GetSystem());

    // Set absolute position
    const ChFrame<>& X_GP = chassis->GetFrame_REF_to_abs();  // global -> parent
    ChFrame<> X_GC = X_GP * m_pos;                           // global -> child
    m_body->SetFrame_REF_to_abs(X_GC);
}

// =============================================================================
// Rover Chassis
ViperChassis::ViperChassis(const std::string& name, std::shared_ptr<ChMaterialSurface> mat)
    : ViperPart(name, ChFrame<>(VNULL, QUNIT), mat, false) {
    m_mesh_name = "viper_chassis";
    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

void ViperChassis::Initialize(ChSystem* system, const ChFrame<>& pos) {
    Construct(system);

    m_body->SetFrame_REF_to_abs(pos);
}

// ==========================================================
// Viper Wheel
ViperWheel::ViperWheel(const std::string& name,
                       const ChFrame<>& rel_pos,
                       std::shared_ptr<ChMaterialSurface> mat,
                       WheelType wheel_type)
    : ViperPart(name, rel_pos, mat, true) {
    switch (wheel_type) {
        case WheelType::RealWheel:
            m_mesh_name = "viper_wheel";
            break;
        case WheelType::SimpleWheel:
            m_mesh_name = "viper_simplewheel";
            break;
        case WheelType::CylWheel:
            m_mesh_name = "viper_cylwheel";
            break;
    }

    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

// ==========================================================
// Viper Upper Suspension Arm
ViperUpperArm::ViperUpperArm(const std::string& name,
                             const ChFrame<>& rel_pos,
                             std::shared_ptr<ChMaterialSurface> mat,
                             const int& side)
    : ViperPart(name, rel_pos, mat, false) {
    if (side == 0) {
        m_mesh_name = "viper_L_up_sus";
    } else if (side == 1) {
        m_mesh_name = "viper_R_up_sus";
    }

    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

// ==========================================================
// Viper Bottom Suspension Arm
ViperLowerArm::ViperLowerArm(const std::string& name,
                             const ChFrame<>& rel_pos,
                             std::shared_ptr<ChMaterialSurface> mat,
                             const int& side)
    : ViperPart(name, rel_pos, mat, false) {
    if (side == 0) {
        m_mesh_name = "viper_L_bt_sus";
    } else if (side == 1) {
        m_mesh_name = "viper_R_bt_sus";
    }

    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

// ==========================================================
// Viper Steering Rod
ViperUpright::ViperUpright(const std::string& name,
                           const ChFrame<>& rel_pos,
                           std::shared_ptr<ChMaterialSurface> mat,
                           const int& side)
    : ViperPart(name, rel_pos, mat, false) {
    if (side == 0) {
        m_mesh_name = "viper_L_steer";
    } else if (side == 1) {
        m_mesh_name = "viper_R_steer";
    }

    m_color = ChColor(0.4f, 0.4f, 0.7f);
    m_density = 200;
}

// ==========================================================
// Rover Class for the entire rover model
Viper::Viper(ChSystem* system, WheelType wheel_type)
    : m_system(system), m_turn_state(TurnSig::HOLD), m_chassis_fixed(false), m_dc_motor_control(false) {
    // Set default collision model envelope commensurate with model dimensions.
    // Note that an SMC system automatically sets envelope to 0.
    auto contact_method = m_system->GetContactMethod();
    if (contact_method == ChContactMethod::NSC) {
        collision::ChCollisionModel::SetDefaultSuggestedEnvelope(0.01);
        collision::ChCollisionModel::SetDefaultSuggestedMargin(0.005);
    }

    // Create the contact materials
    m_default_material = DefaultContactMaterial(contact_method);
    m_wheel_material = DefaultContactMaterial(contact_method);

    m_stall_torque = {300, 300, 300, 300};
    m_no_load_speed = {CH_C_PI, CH_C_PI, CH_C_PI, CH_C_PI};

    Create(wheel_type);
}

Viper::~Viper() {}

void Viper::Create(WheelType wheel_type) {
    // create rover chassis
    m_chassis = chrono_types::make_shared<ViperChassis>("chassis", m_default_material);

    // initilize rover wheels
    double wx = 0.5618 + 0.08;
    double wy = 0.2067 + 0.32 + 0.0831;
    double wz = 0.0;

    m_wheels[LF] = chrono_types::make_shared<ViperWheel>(
        "wheelLF", ChFrame<>(ChVector<>(+wx, +wy, wz), Q_from_AngZ(CH_C_PI)), m_wheel_material, wheel_type);
    m_wheels[RF] = chrono_types::make_shared<ViperWheel>("wheelRF", ChFrame<>(ChVector<>(+wx, -wy, wz), QUNIT),
                                                         m_wheel_material, wheel_type);
    m_wheels[LB] = chrono_types::make_shared<ViperWheel>(
        "wheelLB", ChFrame<>(ChVector<>(-wx, +wy, wz), Q_from_AngZ(CH_C_PI)), m_wheel_material, wheel_type);
    m_wheels[RB] = chrono_types::make_shared<ViperWheel>("wheelRB", ChFrame<>(ChVector<>(-wx, -wy, wz), QUNIT),
                                                         m_wheel_material, wheel_type);

    // create rover up and bottom suspensions
    double cr_lx = 0.5618 + 0.08;
    double cr_ly = 0.2067;  // + 0.32/2;
    double cr_lz = 0.0525;

    ChVector<> cr_rel_pos_lower[] = {
        ChVector<>(+cr_lx, +cr_ly, -cr_lz),  // LF
        ChVector<>(+cr_lx, -cr_ly, -cr_lz),  // RF
        ChVector<>(-cr_lx, +cr_ly, -cr_lz),  // LB
        ChVector<>(-cr_lx, -cr_ly, -cr_lz)   // RB
    };

    ChVector<> cr_rel_pos_upper[] = {
        ChVector<>(+cr_lx, +cr_ly, cr_lz),  // LF
        ChVector<>(+cr_lx, -cr_ly, cr_lz),  // RF
        ChVector<>(-cr_lx, +cr_ly, cr_lz),  // LB
        ChVector<>(-cr_lx, -cr_ly, cr_lz)   // RB
    };

    for (int i = 0; i < 4; i++) {
        m_lower_arms[i] = chrono_types::make_shared<ViperLowerArm>("bt_sus", ChFrame<>(cr_rel_pos_lower[i], QUNIT),
                                                                   m_default_material, i % 2);
        m_upper_arms[i] = chrono_types::make_shared<ViperUpperArm>("up_sus", ChFrame<>(cr_rel_pos_upper[i], QUNIT),
                                                                   m_default_material, i % 2);
    }

    // create steering rod
    double sr_lx = 0.5618 + 0.08;
    double sr_ly = 0.2067 + 0.32 + 0.0831;
    double sr_lz = 0.0;
    ChVector<> sr_rel_pos[] = {
        ChVector<>(+sr_lx, +sr_ly, -sr_lz),  // LF
        ChVector<>(+sr_lx, -sr_ly, -sr_lz),  // RF
        ChVector<>(-sr_lx, +sr_ly, -sr_lz),  // LB
        ChVector<>(-sr_lx, -sr_ly, -sr_lz)   // RB
    };

    for (int i = 0; i < 4; i++) {
        m_uprights[i] = chrono_types::make_shared<ViperUpright>("steering", ChFrame<>(sr_rel_pos[i], QUNIT),
                                                                m_default_material, i % 2);
    }

    // DC Motor Test
    for (int i = 0; i < 4; i++) {
        m_power_shafts[i] = chrono_types::make_shared<ChShaft>();
        m_driven_shafts[i] = chrono_types::make_shared<ChShaft>();
        m_shaft_gears[i] = chrono_types::make_shared<ChShaftsGear>();
    }
}

void Viper::Initialize(const ChFrame<>& pos) {
    m_chassis->Initialize(m_system, pos);
    m_chassis->GetBody()->SetBodyFixed(m_chassis_fixed);

    for (int i = 0; i < 4; i++) {
        m_wheels[i]->Initialize(m_chassis->GetBody());
        m_upper_arms[i]->Initialize(m_chassis->GetBody());
        m_lower_arms[i]->Initialize(m_chassis->GetBody());
        m_uprights[i]->Initialize(m_chassis->GetBody());
    }

    // add all constraints to the system
    // redefine pos data for constraints
    double sr_lx = 0.5618 + 0.08;
    // double sr_ly = 0.2067 + 0.32 + 0.0831;
    // double sr_lz = 0.0;
    double sr_ly_joint = 0.2067 + 0.32;

    double cr_lx = 0.5618 + 0.08;
    double cr_ly = 0.2067;  // + 0.32/2;
    double cr_lz = 0.0525;

    double w_lx = 0.5618 + 0.08;
    double w_ly = 0.2067 + 0.32 + 0.0831;
    double w_lz = 0.0;

    ChVector<> wheel_rel_pos[] = {
        ChVector<>(+w_lx, +w_ly, w_lz),  // LF
        ChVector<>(+w_lx, -w_ly, w_lz),  // RF
        ChVector<>(-w_lx, +w_ly, w_lz),  // LB
        ChVector<>(-w_lx, -w_ly, w_lz)   // RB
    };

    ChVector<> sr_rel_pos_lower[] = {
        ChVector<>(+sr_lx, +sr_ly_joint, -cr_lz),  // LF
        ChVector<>(+sr_lx, -sr_ly_joint, -cr_lz),  // RF
        ChVector<>(-sr_lx, +sr_ly_joint, -cr_lz),  // LB
        ChVector<>(-sr_lx, -sr_ly_joint, -cr_lz)   // RB
    };

    ChVector<> sr_rel_pos_upper[] = {
        ChVector<>(+sr_lx, +sr_ly_joint, cr_lz),  // LF
        ChVector<>(+sr_lx, -sr_ly_joint, cr_lz),  // RF
        ChVector<>(-sr_lx, +sr_ly_joint, cr_lz),  // LB
        ChVector<>(-sr_lx, -sr_ly_joint, cr_lz)   // RB
    };

    ChVector<> cr_rel_pos_lower[] = {
        ChVector<>(+cr_lx, +cr_ly, -cr_lz),  // LF
        ChVector<>(+cr_lx, -cr_ly, -cr_lz),  // RF
        ChVector<>(-cr_lx, +cr_ly, -cr_lz),  // LB
        ChVector<>(-cr_lx, -cr_ly, -cr_lz)   // RB
    };

    ChVector<> cr_rel_pos_upper[] = {
        ChVector<>(+cr_lx, +cr_ly, cr_lz),  // LF
        ChVector<>(+cr_lx, -cr_ly, cr_lz),  // RF
        ChVector<>(-cr_lx, +cr_ly, cr_lz),  // LB
        ChVector<>(-cr_lx, -cr_ly, cr_lz)   // RB
    };

    ChQuaternion<> z2x;
    z2x.Q_from_AngAxis(-CH_C_PI / 2, ChVector<>(0, 1, 0));

    for (int i = 0; i < 4; i++) {
        AddRevoluteJoint(m_lower_arms[i]->GetBody(), m_uprights[i]->GetBody(), m_chassis->GetBody(), m_system,
                         sr_rel_pos_lower[i], z2x);
        AddRevoluteJoint(m_upper_arms[i]->GetBody(), m_uprights[i]->GetBody(), m_chassis->GetBody(), m_system,
                         sr_rel_pos_upper[i], z2x);

        // Add lifting motors at the connecting points between upper_suspension&chassis and bottom_suspension&chassis
        // create lifting motors speed control functions
        m_lift_motor_funcs[i] = chrono_types::make_shared<ChFunction_Const>(0.0);

        m_lift_motors[2 * i] = AddMotor(m_chassis->GetBody(), m_lower_arms[i]->GetBody(), m_chassis->GetBody(),
                                        m_system, cr_rel_pos_lower[i], z2x, m_lift_motor_funcs[i]);
        m_lift_motors[2 * i + 1] = AddMotor(m_chassis->GetBody(), m_upper_arms[i]->GetBody(), m_chassis->GetBody(),
                                            m_system, cr_rel_pos_upper[i], z2x, m_lift_motor_funcs[i]);

        auto steer_rod = chrono_types::make_shared<ChBodyEasyBox>(0.1, 0.1, 0.1, 1000, true, false);

        const ChFrame<>& X_GP = m_chassis->GetBody()->GetFrame_REF_to_abs();
        ChFrame<> X_PC(wheel_rel_pos[i], ChQuaternion<>(1, 0, 0, 0));
        ChFrame<> X_GC = X_GP * X_PC;

        steer_rod->SetPos(X_GC.GetCoord().pos);
        steer_rod->SetBodyFixed(false);
        m_system->Add(steer_rod);

        ChQuaternion<> z2y;
        z2y.Q_from_AngAxis(CH_C_PI / 2, ChVector<>(1, 0, 0));

        // temporarily disable motor link
        if (m_dc_motor_control) {
            AddRevoluteJoint(steer_rod, m_wheels[i]->GetBody(), m_chassis->GetBody(), m_system, wheel_rel_pos[i], z2y);
        } else {
            m_drive_motor_funcs[i] = chrono_types::make_shared<ChFunction_Const>(CH_C_PI);
            m_drive_motors[i] = AddMotor(steer_rod, m_wheels[i]->GetBody(), m_chassis->GetBody(), m_system,
                                         wheel_rel_pos[i], z2y, m_drive_motor_funcs[i]);
        }

        m_steer_motor_funcs[i] = chrono_types::make_shared<ChFunction_Const>(0.0);
        m_steer_motors[i] = AddMotor(steer_rod, m_uprights[i]->GetBody(), m_chassis->GetBody(), m_system,
                                     wheel_rel_pos[i], ChQuaternion<>(1, 0, 0, 0), m_steer_motor_funcs[i]);

        m_springs[i] = AddSuspensionSpring(m_chassis->GetBody(), m_uprights[i]->GetBody(), m_system,
                                           cr_rel_pos_upper[i], sr_rel_pos_lower[i]);
    }

    if (m_dc_motor_control) {
        // DC Motor Test
        double J1 = 10;   // inertia of first shaft
        double J2 = 100;  // inertia of second shaft
        double r = -1;    // gear transmission ratio
        double T = 300;   // torque applied to first shaft
        for (int i = 0; i < 4; i++) {
            m_power_shafts[i]->SetInertia(J1);
            if (i == 0 || i == 2) {
                m_power_shafts[i]->SetAppliedTorque(-T);
            } else {
                m_power_shafts[i]->SetAppliedTorque(T);
            }

            m_system->Add(m_power_shafts[i]);

            m_driven_shafts[i]->SetInertia(J2);
            m_system->Add(m_driven_shafts[i]);

            m_shaft_gears[i]->Initialize(m_power_shafts[i], m_driven_shafts[i]);
            m_shaft_gears[i]->SetTransmissionRatio(r);
            m_system->Add(m_shaft_gears[i]);

            auto shaftbody_connection = chrono_types::make_shared<ChShaftsBody>();
            shaftbody_connection->Initialize(m_driven_shafts[i], m_wheels[i]->GetBody(), ChVector<>(0, 0, 1));
            m_system->Add(shaftbody_connection);
        }
    }
}

void Viper::SetDCControl(bool dc_control) {
    m_dc_motor_control = dc_control;
}

void Viper::SetMotorSpeed(double rad_speed, WheelID id) {
    if (!m_dc_motor_control)
        m_drive_motor_funcs[id]->Set_yconst(rad_speed);
}

void Viper::SetLiftMotorSpeed(double rad_speed, WheelID id) {
    m_lift_motor_funcs[id]->Set_yconst(rad_speed);
}

void Viper::SetMotorNoLoadSpeed(double rad_speed, WheelID id) {
    if (m_dc_motor_control)
        m_no_load_speed[id] = rad_speed;
}

void Viper::SetMotorStallTorque(double torque, WheelID id) {
    m_stall_torque[id] = torque;
}

void Viper::SetWheelContactMaterial(std::shared_ptr<ChMaterialSurface> mat) {
    for (auto& wheel : m_wheels)
        wheel->m_mat = mat;
}

void Viper::SetChassisFixed(bool fixed) {
    m_chassis_fixed = fixed;
}

void Viper::SetChassisVisualization(bool state) {
    m_chassis->SetVisualize(state);
}

void Viper::SetWheelVisualization(bool state) {
    for (auto& wheel : m_wheels)
        wheel->SetVisualize(state);
}

void Viper::SetSuspensionVisualization(bool state) {
    for (auto& p : m_lower_arms)
        p->SetVisualize(state);
    for (auto& p : m_upper_arms)
        p->SetVisualize(state);
    for (auto& p : m_uprights)
        p->SetVisualize(state);
}

ChVector<> Viper::GetWheelContactForce(WheelID id) const {
    return m_wheels[id]->GetBody()->GetContactForce();
}

ChVector<> Viper::GetWheelContactTorque(WheelID id) const {
    return m_wheels[id]->GetBody()->GetContactTorque();
}

ChVector<> Viper::GetWheelAppliedForce(WheelID id) const {
    return m_wheels[id]->GetBody()->GetAppliedForce();
}

ChVector<> Viper::GetWheelAppliedTorque(WheelID id) const {
    return m_wheels[id]->GetBody()->GetAppliedTorque();
}

double Viper::GetWheelTracTorque(WheelID id) const {
    if (m_dc_motor_control)
        return 0;

    return m_drive_motors[id]->GetMotorTorque();
}

double Viper::GetRoverMass() const {
    double tot_mass = 0.0;
    for (int i = 0; i < 4; i++) {
        tot_mass = tot_mass + m_wheels[i]->GetBody()->GetMass();
        tot_mass = tot_mass + m_upper_arms[i]->GetBody()->GetMass();
        tot_mass = tot_mass + m_lower_arms[i]->GetBody()->GetMass();
        tot_mass = tot_mass + m_uprights[i]->GetBody()->GetMass();
    }
    tot_mass = tot_mass + m_chassis->GetBody()->GetMass();
    return tot_mass;
}

double Viper::GetWheelMass() const {
    return m_wheels[0]->GetBody()->GetMass();
}

std::shared_ptr<ChFunction_Const> Viper::GetMainMotorFunc(WheelID id) {
    return m_drive_motor_funcs[id];
}

std::shared_ptr<ChFunction_Const> Viper::GetSteerMotorFunc(WheelID id) {
    return m_steer_motor_funcs[id];
}

std::shared_ptr<ChLinkMotorRotationSpeed> Viper::GetMainMotorLink(WheelID id) {
    return m_drive_motors[id];
}

std::shared_ptr<ChLinkMotorRotationSpeed> Viper::GetSteerMotorLink(WheelID id) {
    return m_steer_motors[id];
}

void Viper::SetTurn(TurnSig id, double turn_speed) {
    // maximum valid turn_speed input is 4 * CH_C_PI
    std::cout << "turn_speed:" << turn_speed << std::endl;
    if (std::abs(turn_speed) > 4 * CH_C_PI) {
        std::cout << "FATAL ERROR, STEERING SPEED CANNOT EXCEED 4*PI" << std::endl;
        return;
    }

    switch (id) {
        case TurnSig::LEFT:
            for (int i = 0; i < 4; i++) {
                if (i == 0 || i == 1) {
                    m_steer_motor_funcs[i]->Set_yconst(turn_speed);
                } else {
                    m_steer_motor_funcs[i]->Set_yconst(-turn_speed);
                }
            }
            break;

        case TurnSig::RIGHT:
            for (int i = 0; i < 4; i++) {
                if (i == 0 || i == 1) {
                    m_steer_motor_funcs[i]->Set_yconst(-turn_speed);
                } else {
                    m_steer_motor_funcs[i]->Set_yconst(turn_speed);
                }
            }
            break;

        case TurnSig::HOLD:
            for (int i = 0; i < 4; i++) {
                m_steer_motor_funcs[i]->Set_yconst(0.0);
            }
            break;

        default:
            break;
    }
    m_turn_state = id;
}

// turning angle ranges from -pi/3 to pi/3
double Viper::GetTurnAngle() const {
    return 2 * (m_steer_motors[0]->GetMotorRot());
}

TurnSig Viper::GetTurnState() const {
    return m_turn_state;
}

void Viper::Update() {
    UpdateDCMotorControl();
    UpdateSteeringControl();
}

// A sloppy DC motor control
// TODO: A better model is needed
void Viper::UpdateDCMotorControl() {
    if (!m_dc_motor_control)
        return;

    double speed_reading;
    double target_torque;
    for (int i = 0; i < 4; i++) {
        if (i == 0 || i == 2) {
            speed_reading = m_driven_shafts[i]->GetPos_dt();
        } else {
            speed_reading = -m_driven_shafts[i]->GetPos_dt();
        }

        if (speed_reading > m_no_load_speed[i]) {
            target_torque = 0;
        } else if (speed_reading < 0) {
            target_torque = m_stall_torque[i];
        } else {
            target_torque = m_stall_torque[i] * ((m_no_load_speed[i] - speed_reading) / m_no_load_speed[i]);
        }

        if (i == 0 || i == 2) {
            m_power_shafts[i]->SetAppliedTorque(-target_torque);
        } else {
            m_power_shafts[i]->SetAppliedTorque(target_torque);
        }
    }
}

void Viper::UpdateSteeringControl() {
    switch (m_turn_state) {
        case TurnSig::LEFT:
            for (int i = 0; i < 4; i++) {
                if (i == 0 || i == 1) {
                    if (m_steer_motors[i]->GetMotorRot() > CH_C_PI / 6) {
                        m_steer_motor_funcs[i]->Set_yconst(0.0);
                    }
                } else {
                    if (m_steer_motors[i]->GetMotorRot() < -CH_C_PI / 6) {
                        m_steer_motor_funcs[i]->Set_yconst(0.0);
                    }
                }
            }
            break;

        case TurnSig::RIGHT:
            for (int i = 0; i < 4; i++) {
                if (i == 0 || i == 1) {
                    if (m_steer_motors[i]->GetMotorRot() < -CH_C_PI / 6) {
                        m_steer_motor_funcs[i]->Set_yconst(0.0);
                    }
                } else {
                    if (m_steer_motors[i]->GetMotorRot() > CH_C_PI / 6) {
                        m_steer_motor_funcs[i]->Set_yconst(0.0);
                    }
                }
            }
            break;

        case TurnSig::HOLD:
            break;

        default:
            break;
    }
}

}  // namespace viper
}  // namespace chrono
