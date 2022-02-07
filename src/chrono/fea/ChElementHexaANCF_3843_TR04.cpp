// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Michael Taylor, Antonio Recuero, Radu Serban
// =============================================================================
// Fully Parameterized ANCF brick element with 8 nodes (96DOF). A Description of this element can be found in:
// Olshevskiy, A., Dmitrochenko, O., & Kim, C. W. (2014). Three-dimensional solid brick element using slopes in the
// absolute nodal coordinate formulation. Journal of Computational and Nonlinear Dynamics, 9(2).
// =============================================================================
//
// =============================================================================
// TR04 = Gerstmayr and Shabana without Precomputation
// =============================================================================
// Mass Matrix = Full 3Nx3N
// Reduced Number of GQ Points
// Nodal Coordinates in Matrix Form
// PK1 Stress
// No Precomputing Quantities
// Numeric Jacobian
// =============================================================================

#include "chrono/fea/ChElementHexaANCF_3843_TR04.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementHexaANCF_3843_TR04::ChElementHexaANCF_3843_TR04()
    : m_lenX(0), m_lenY(0), m_lenZ(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(8);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR04::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeB,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeC,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeD,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeE,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeF,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeG,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeH) {
    assert(nodeA);
    assert(nodeB);
    assert(nodeC);
    assert(nodeD);
    assert(nodeE);
    assert(nodeF);
    assert(nodeG);
    assert(nodeH);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;
    m_nodes[2] = nodeC;
    m_nodes[3] = nodeD;
    m_nodes[4] = nodeE;
    m_nodes[5] = nodeF;
    m_nodes[6] = nodeG;
    m_nodes[7] = nodeH;

    std::vector<ChVariables*> mvars;
    mvars.push_back(&m_nodes[0]->Variables());
    mvars.push_back(&m_nodes[0]->Variables_D());
    mvars.push_back(&m_nodes[0]->Variables_DD());
    mvars.push_back(&m_nodes[0]->Variables_DDD());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[1]->Variables_DDD());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->Variables_D());
    mvars.push_back(&m_nodes[2]->Variables_DD());
    mvars.push_back(&m_nodes[2]->Variables_DDD());
    mvars.push_back(&m_nodes[3]->Variables());
    mvars.push_back(&m_nodes[3]->Variables_D());
    mvars.push_back(&m_nodes[3]->Variables_DD());
    mvars.push_back(&m_nodes[3]->Variables_DDD());
    mvars.push_back(&m_nodes[4]->Variables());
    mvars.push_back(&m_nodes[4]->Variables_D());
    mvars.push_back(&m_nodes[4]->Variables_DD());
    mvars.push_back(&m_nodes[4]->Variables_DDD());
    mvars.push_back(&m_nodes[5]->Variables());
    mvars.push_back(&m_nodes[5]->Variables_D());
    mvars.push_back(&m_nodes[5]->Variables_DD());
    mvars.push_back(&m_nodes[5]->Variables_DDD());
    mvars.push_back(&m_nodes[6]->Variables());
    mvars.push_back(&m_nodes[6]->Variables_D());
    mvars.push_back(&m_nodes[6]->Variables_DD());
    mvars.push_back(&m_nodes[6]->Variables_DDD());
    mvars.push_back(&m_nodes[7]->Variables());
    mvars.push_back(&m_nodes[7]->Variables_D());
    mvars.push_back(&m_nodes[7]->Variables_DD());
    mvars.push_back(&m_nodes[7]->Variables_DDD());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.

void ChElementHexaANCF_3843_TR04::SetDimensions(double lenX, double lenY, double lenZ) {
    m_lenX = lenX;
    m_lenY = lenY;
    m_lenZ = lenZ;
}

// Specify the element material.

void ChElementHexaANCF_3843_TR04::SetMaterial(std::shared_ptr<ChMaterialHexaANCF> brick_mat) {
    m_material = brick_mat;
}

// Set the value for the single term structural damping coefficient.

void ChElementHexaANCF_3843_TR04::SetAlphaDamp(double a) {
    m_Alpha = a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

// -----------------------------------------------------------------------------
// Evaluate Strains and Stresses
// -----------------------------------------------------------------------------
// These functions are designed for single function calls.  If these values are needed at the same points in the element
// through out the simulation, then the adjusted normalized shape function derivative matrix (Sxi_D) for each query
// point should be cached and saved to increase the execution speed
// -----------------------------------------------------------------------------

// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) [-1...1]

ChMatrix33<> ChElementHexaANCF_3843_TR04::GetGreenLagrangeStrain(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = e_bar * Sxi_D;

    ChMatrix33<> I3x3;
    I3x3.setIdentity();
    return 0.5 * (F.transpose() * F - I3x3);
}

// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) [-1...1] at the
// current state of the element.

ChMatrix33<> ChElementHexaANCF_3843_TR04::GetPK2Stress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = e_bar * Sxi_D;

    // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
    ChVectorN<double, 6> epsilon_combined;
    epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
    epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
    epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);
    epsilon_combined(3) = F.col(1).dot(F.col(2));
    epsilon_combined(4) = F.col(0).dot(F.col(2));
    epsilon_combined(5) = F.col(0).dot(F.col(1));

    if (m_damping_enabled) {
        Matrix3xN ebardot;  // Element coordinate time derivatives in matrix form
        CalcCoordDerivMatrix(ebardot);

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sxi_D;

        // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
        epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    }

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    ChVectorN<double, 6> sigmaPK2 = D * epsilon_combined;  // 2nd Piola Kirchhoff Stress tensor in Voigt notation

    ChMatrix33<> SPK2;
    SPK2(0, 0) = sigmaPK2(0);
    SPK2(1, 1) = sigmaPK2(1);
    SPK2(2, 2) = sigmaPK2(2);
    SPK2(1, 2) = sigmaPK2(3);
    SPK2(2, 1) = sigmaPK2(3);
    SPK2(0, 2) = sigmaPK2(4);
    SPK2(2, 0) = sigmaPK2(4);
    SPK2(0, 1) = sigmaPK2(5);
    SPK2(1, 0) = sigmaPK2(5);

    return SPK2;
}

// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) [-1...1] at the current
// state of the element.

double ChElementHexaANCF_3843_TR04::GetVonMissesStress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = e_bar * Sxi_D;

    // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
    ChVectorN<double, 6> epsilon_combined;
    epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
    epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
    epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);
    epsilon_combined(3) = F.col(1).dot(F.col(2));
    epsilon_combined(4) = F.col(0).dot(F.col(2));
    epsilon_combined(5) = F.col(0).dot(F.col(1));

    if (m_damping_enabled) {
        Matrix3xN ebardot;  // Element coordinate time derivatives in matrix form
        CalcCoordDerivMatrix(ebardot);

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sxi_D;

        // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
        epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    }

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    ChVectorN<double, 6> sigmaPK2 = D * epsilon_combined;  // 2nd Piola Kirchhoff Stress tensor in Voigt notation

    ChMatrixNM<double, 3, 3> SPK2;  // 2nd Piola Kirchhoff Stress tensor
    SPK2(0, 0) = sigmaPK2(0);
    SPK2(1, 1) = sigmaPK2(1);
    SPK2(2, 2) = sigmaPK2(2);
    SPK2(1, 2) = sigmaPK2(3);
    SPK2(2, 1) = sigmaPK2(3);
    SPK2(0, 2) = sigmaPK2(4);
    SPK2(2, 0) = sigmaPK2(4);
    SPK2(0, 1) = sigmaPK2(5);
    SPK2(1, 0) = sigmaPK2(5);

    // Convert from 2ndPK Stress to Cauchy Stress
    ChMatrix33<double> S = (F * SPK2 * F.transpose()) / F.determinant();
    double SVonMises =
        sqrt(0.5 * ((S(0, 0) - S(1, 1)) * (S(0, 0) - S(1, 1)) + (S(1, 1) - S(2, 2)) * (S(1, 1) - S(2, 2)) +
                    (S(2, 2) - S(0, 0)) * (S(2, 2) - S(0, 0))) +
             3 * (S(1, 2) * S(1, 2) + S(2, 0) * S(2, 0) + S(0, 1) * S(0, 1)));

    return (SVonMises);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.

void ChElementHexaANCF_3843_TR04::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementHexaANCF_3843_TR04::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(21, 3) = m_nodes[1]->GetDDD().eigen();

    mD.segment(24, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(27, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(30, 3) = m_nodes[2]->GetDD().eigen();
    mD.segment(33, 3) = m_nodes[2]->GetDDD().eigen();

    mD.segment(36, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(39, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(42, 3) = m_nodes[3]->GetDD().eigen();
    mD.segment(45, 3) = m_nodes[3]->GetDDD().eigen();

    mD.segment(48, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(51, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(54, 3) = m_nodes[4]->GetDD().eigen();
    mD.segment(57, 3) = m_nodes[4]->GetDDD().eigen();

    mD.segment(60, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(63, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(66, 3) = m_nodes[5]->GetDD().eigen();
    mD.segment(69, 3) = m_nodes[5]->GetDDD().eigen();

    mD.segment(72, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(75, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(78, 3) = m_nodes[6]->GetDD().eigen();
    mD.segment(81, 3) = m_nodes[6]->GetDDD().eigen();

    mD.segment(84, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(87, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(90, 3) = m_nodes[7]->GetDD().eigen();
    mD.segment(93, 3) = m_nodes[7]->GetDDD().eigen();
}

// State update.

void ChElementHexaANCF_3843_TR04::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementHexaANCF_3843_TR04::ComputeMmatrixGlobal(ChMatrixRef M) {
    M = m_MassMatrix;
}

// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass

void ChElementHexaANCF_3843_TR04::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0, 0) + m_MassMatrix(0, 12) + m_MassMatrix(0, 24) + m_MassMatrix(0, 36) +
                               m_MassMatrix(0, 48) + m_MassMatrix(0, 60) + m_MassMatrix(0, 72) + m_MassMatrix(0, 84);
    m_nodes[1]->m_TotalMass += m_MassMatrix(12, 0) + m_MassMatrix(12, 12) + m_MassMatrix(12, 24) +
                               m_MassMatrix(12, 36) + m_MassMatrix(12, 48) + m_MassMatrix(12, 60) +
                               m_MassMatrix(12, 72) + m_MassMatrix(12, 84);
    m_nodes[2]->m_TotalMass += m_MassMatrix(24, 0) + m_MassMatrix(24, 12) + m_MassMatrix(24, 24) +
                               m_MassMatrix(24, 36) + m_MassMatrix(24, 48) + m_MassMatrix(24, 60) +
                               m_MassMatrix(24, 72) + m_MassMatrix(24, 84);
    m_nodes[3]->m_TotalMass += m_MassMatrix(36, 0) + m_MassMatrix(36, 12) + m_MassMatrix(36, 24) +
                               m_MassMatrix(36, 36) + m_MassMatrix(36, 48) + m_MassMatrix(36, 60) +
                               m_MassMatrix(36, 72) + m_MassMatrix(36, 84);
    m_nodes[4]->m_TotalMass += m_MassMatrix(48, 0) + m_MassMatrix(48, 12) + m_MassMatrix(48, 24) +
                               m_MassMatrix(48, 36) + m_MassMatrix(48, 48) + m_MassMatrix(48, 60) +
                               m_MassMatrix(48, 72) + m_MassMatrix(48, 84);
    m_nodes[5]->m_TotalMass += m_MassMatrix(60, 0) + m_MassMatrix(60, 12) + m_MassMatrix(60, 24) +
                               m_MassMatrix(60, 36) + m_MassMatrix(60, 48) + m_MassMatrix(60, 60) +
                               m_MassMatrix(60, 72) + m_MassMatrix(60, 84);
    m_nodes[6]->m_TotalMass += m_MassMatrix(72, 0) + m_MassMatrix(72, 12) + m_MassMatrix(72, 24) +
                               m_MassMatrix(72, 36) + m_MassMatrix(72, 48) + m_MassMatrix(72, 60) +
                               m_MassMatrix(72, 72) + m_MassMatrix(72, 84);
    m_nodes[7]->m_TotalMass += m_MassMatrix(84, 0) + m_MassMatrix(84, 12) + m_MassMatrix(84, 24) +
                               m_MassMatrix(84, 36) + m_MassMatrix(84, 48) + m_MassMatrix(84, 60) +
                               m_MassMatrix(84, 72) + m_MassMatrix(84, 84);
}

// Compute the generalized internal force vector for the current nodal coordinates and set the value in the Fi vector.

void ChElementHexaANCF_3843_TR04::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == 3 * NSF);

    Matrix3xN ebar;
    Matrix3xN ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);
    ComputeInternalForcesAtState(Fi, ebar, ebardot);
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementHexaANCF_3843_TR04::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                           double Kfactor,
                                                           double Rfactor,
                                                           double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    H.setZero();

    ChVectorDynamic<double> FiOrignal(3 * NSF);
    ChVectorDynamic<double> FiDelta(3 * NSF);
    Matrix3xN ebar;
    Matrix3xN ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);

    double delta = 1e-6;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForcesAtState(FiOrignal, ebar, ebardot);
    for (unsigned int i = 0; i < (3 * NSF); i++) {
        ebar(i % 3, i / 3) += delta;
        ComputeInternalForcesAtState(FiDelta, ebar, ebardot);
        H.col(i).noalias() = -Kfactor / delta * (FiDelta - FiOrignal);
        ebar(i % 3, i / 3) -= delta;

        ebardot(i % 3, i / 3) += delta;
        ComputeInternalForcesAtState(FiDelta, ebar, ebardot);
        H.col(i).noalias() += -Rfactor / delta * (FiDelta - FiOrignal);
        ebardot(i % 3, i / 3) -= delta;
    }

    // Add in the scaled Mass Matrix
    H.noalias() += Mfactor * m_MassMatrix;
}

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementHexaANCF_3843_TR04::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector<>& G_acc) {
    assert(Fg.size() == 3 * NSF);

    // Calculate and add the generalized force due to gravity to the generalized internal force vector for the element.
    // The generalized force due to gravity could be computed once prior to the start of the simulation if gravity was
    // assumed constant throughout the entire simulation.  However, this implementation assumes that the acceleration
    // due to gravity, while a constant for the entire system, can change from step to step which could be useful for
    // gravity loaded units tests as an example.  The generalized force due to gravity is calculated in compact matrix
    // form and is pre-mapped to the desired vector format
    Eigen::Map<MatrixNx3> GravForceCompact(Fg.data(), NSF, 3);
    GravForceCompact = m_GravForceScale * G_acc.eigen().transpose();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR04::EvaluateElementFrame(const double xi,
                                                       const double eta,
                                                       const double zeta,
                                                       ChVector<>& point,
                                                       ChQuaternion<>& rot) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta);
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector<double> Xdir = e_bar * Sxi_xi_compact * 2 / m_lenX;
    ChVector<double> Ydir = e_bar * Sxi_eta_compact * 2 / m_lenY;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the brick xi axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the brick eta axis
    ChMatrix33<> msect;
    msect.Set_A_Xdir(Xdir, Ydir);

    rot = msect.Get_A_quaternion();
}

void ChElementHexaANCF_3843_TR04::EvaluateElementPoint(const double xi,
                                                       const double eta,
                                                       const double zeta,
                                                       ChVector<>& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementHexaANCF_3843_TR04::EvaluateElementVel(double xi, double eta, const double zeta, ChVector<>& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

    Matrix3xN e_bardot;
    CalcCoordDerivMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).

void ChElementHexaANCF_3843_TR04::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD().eigen();

    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 27, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[2]->GetDD().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[2]->GetDDD().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[3]->GetDD().eigen();
    mD.segment(block_offset + 45, 3) = m_nodes[3]->GetDDD().eigen();

    mD.segment(block_offset + 48, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(block_offset + 54, 3) = m_nodes[4]->GetDD().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[4]->GetDDD().eigen();

    mD.segment(block_offset + 60, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(block_offset + 63, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[5]->GetDD().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[5]->GetDDD().eigen();

    mD.segment(block_offset + 72, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(block_offset + 75, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(block_offset + 78, 3) = m_nodes[6]->GetDD().eigen();
    mD.segment(block_offset + 81, 3) = m_nodes[6]->GetDDD().eigen();

    mD.segment(block_offset + 84, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(block_offset + 87, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(block_offset + 90, 3) = m_nodes[7]->GetDD().eigen();
    mD.segment(block_offset + 93, 3) = m_nodes[7]->GetDDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementHexaANCF_3843_TR04::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD_dt().eigen();

    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetPos_dt().eigen();
    mD.segment(block_offset + 27, 3) = m_nodes[2]->GetD_dt().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[2]->GetDD_dt().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[2]->GetDDD_dt().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[3]->GetPos_dt().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[3]->GetD_dt().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[3]->GetDD_dt().eigen();
    mD.segment(block_offset + 45, 3) = m_nodes[3]->GetDDD_dt().eigen();

    mD.segment(block_offset + 48, 3) = m_nodes[4]->GetPos_dt().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[4]->GetD_dt().eigen();
    mD.segment(block_offset + 54, 3) = m_nodes[4]->GetDD_dt().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[4]->GetDDD_dt().eigen();

    mD.segment(block_offset + 60, 3) = m_nodes[5]->GetPos_dt().eigen();
    mD.segment(block_offset + 63, 3) = m_nodes[5]->GetD_dt().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[5]->GetDD_dt().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[5]->GetDDD_dt().eigen();

    mD.segment(block_offset + 72, 3) = m_nodes[6]->GetPos_dt().eigen();
    mD.segment(block_offset + 75, 3) = m_nodes[6]->GetD_dt().eigen();
    mD.segment(block_offset + 78, 3) = m_nodes[6]->GetDD_dt().eigen();
    mD.segment(block_offset + 81, 3) = m_nodes[6]->GetDDD_dt().eigen();

    mD.segment(block_offset + 84, 3) = m_nodes[7]->GetPos_dt().eigen();
    mD.segment(block_offset + 87, 3) = m_nodes[7]->GetD_dt().eigen();
    mD.segment(block_offset + 90, 3) = m_nodes[7]->GetDD_dt().eigen();
    mD.segment(block_offset + 93, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

/// Increment all DOFs using a delta.

void ChElementHexaANCF_3843_TR04::LoadableStateIncrement(const unsigned int off_x,
                                                         ChState& x_new,
                                                         const ChState& x,
                                                         const unsigned int off_v,
                                                         const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 24, x_new, x, off_v + 24, Dv);
    m_nodes[3]->NodeIntStateIncrement(off_x + 36, x_new, x, off_v + 36, Dv);
    m_nodes[4]->NodeIntStateIncrement(off_x + 48, x_new, x, off_v + 48, Dv);
    m_nodes[5]->NodeIntStateIncrement(off_x + 60, x_new, x, off_v + 60, Dv);
    m_nodes[6]->NodeIntStateIncrement(off_x + 72, x_new, x, off_v + 72, Dv);
    m_nodes[7]->NodeIntStateIncrement(off_x + 84, x_new, x, off_v + 84, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.

void ChElementHexaANCF_3843_TR04::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementHexaANCF_3843_TR04::ComputeNF(
    const double xi,             // parametric coordinate in volume
    const double eta,            // parametric coordinate in volume
    const double zeta,           // parametric coordinate in volume
    ChVectorDynamic<>& Qi,       // Return result of N'*F  here, maybe with offset block_offset
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is = n.field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force component using the compact form of the shape
    // functions.  This requires a reshaping of the calculated matrix to get it into the correct vector order (just a
    // reinterpretation of the data since the matrix is in row-major format)
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    MatrixNx3 QiCompact;

    QiCompact = Sxi_compact * F.segment(0, 3).transpose();

    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Qi = QiReshaped;

    // Compute the generalized force vector for the applied moment component
    // See: Antonio M Recuero, Javier F Aceituno, Jose L Escalona, and Ahmed A Shabana.
    // A nonlinear approach for modeling rail flexibility using the absolute nodal coordinate
    // formulation. Nonlinear Dynamics, 83(1-2):463-481, 2016.

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;

    J_Cxi.noalias() = e_bar * Sxi_D;
    J_Cxi_Inv = J_Cxi.inverse();

    // Compute the unique pieces that make up the moment projection matrix "G"
    VectorN G_A = Sxi_D.col(0).transpose() * J_Cxi_Inv(0, 0) + Sxi_D.col(1).transpose() * J_Cxi_Inv(1, 0) +
                  Sxi_D.col(2).transpose() * J_Cxi_Inv(2, 0);
    VectorN G_B = Sxi_D.col(0).transpose() * J_Cxi_Inv(0, 1) + Sxi_D.col(1).transpose() * J_Cxi_Inv(1, 1) +
                  Sxi_D.col(2).transpose() * J_Cxi_Inv(2, 1);
    VectorN G_C = Sxi_D.col(0).transpose() * J_Cxi_Inv(0, 2) + Sxi_D.col(1).transpose() * J_Cxi_Inv(1, 2) +
                  Sxi_D.col(2).transpose() * J_Cxi_Inv(2, 2);

    ChVectorN<double, 3> M_scaled = 0.5 * F.segment(3, 3);

    // Compute G'M without actually forming the complete matrix "G" (since it has a sparsity pattern to it)
    for (unsigned int i = 0; i < NSF; i++) {
        Qi(3 * i) += M_scaled(1) * G_C(i) - M_scaled(2) * G_B(i);
        Qi((3 * i) + 1) += M_scaled(2) * G_A(i) - M_scaled(0) * G_C(i);
        Qi((3 * i) + 2) += M_scaled(0) * G_B(i) - M_scaled(1) * G_A(i);
    }

    // Compute the element Jacobian between the current configuration and the normalized configuration
    // This is different than the element Jacobian between the reference configuration and the normalized
    //  configuration used in the internal force calculations.  For this calculation, this is the ratio between the
    //  actual differential volume and the normalized differential volume.  The determinate of the element Jacobian is
    //  used to calculate this volume ratio for potential use in Gauss-Quadrature or similar numeric integration.
    detJ = J_Cxi.determinant();
}

// Return the element density (needed for ChLoaderVolumeGravity).

double ChElementHexaANCF_3843_TR04::GetDensity() {
    return GetMaterial()->Get_rho();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR04::ComputeMassMatrixAndGravityForce() {
    // For this element, the mass matrix integrand is of order 7 in xi, 7 in eta, and 7 in zeta.
    // 4 GQ Points are needed in the xi, eta, and zeta directions for exact integration of the element's mass matrix,
    // even if the reference configuration is not straight. Since the major pieces of the generalized force due to
    // gravity can also be used to calculate the mass matrix, these calculations are performed at the same time.  Only
    // the matrix that scales the acceleration due to gravity is calculated at this time so that any changes to the
    // acceleration due to gravity in the system are correctly accounted for in the generalized internal force
    // calculation.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = 3;  // 4 Point Gauss-Quadrature;

    // Mass Matrix in its compact matrix form.  Since the mass matrix is symmetric, just the upper diagonal entries will
    // be stored.
    MatrixNxN MassMatrixCompactSquare;

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MassMatrixCompactSquare.setZero();
    m_GravForceScale.setZero();

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

    // Sum the contribution to the mass matrix and generalized force due to gravity at the current point
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta_zeta][it_xi] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta);  // determinant of the element Jacobian (volume ratio)

                VectorN Sxi_compact;  // Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForceScale += (GQ_weight * rho * det_J_0xi) * Sxi_compact;
                MassMatrixCompactSquare += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }

    // Store the full mass matrix for this version
    m_MassMatrix.setZero();
    for (unsigned i = 0; i < NSF; i++) {
        for (unsigned j = 0; j < NSF; j++) {
            m_MassMatrix(3 * i, 3 * j) = MassMatrixCompactSquare(i, j);
            m_MassMatrix(3 * i + 1, 3 * j + 1) = MassMatrixCompactSquare(i, j);
            m_MassMatrix(3 * i + 2, 3 * j + 2) = MassMatrixCompactSquare(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR04::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
                                                               const Matrix3xN& ebar,
                                                               const Matrix3xN& ebardot) {
    // Set Fi to zero since the results from each GQ point will be added to this vector
    Fi.setZero();

    // Straight & Normalized Internal Force Integrand is of order: 12 in xi, order: 12 in eta, and order: 12 in zeta.
    // This requires GQ 7 points along the xi, eta, & zeta directions for "Full Integration"

    ChQuadratureTables* GQTable = GetStaticGQTables();

    // Get the components of the stiffness tensor in 6x6 matrix form that exclude the Poisson effect
    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    // Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[NP - 1].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[NP - 1].size(); it_zeta++) {
                double GQ_weight =
                    GQTable->Weight[NP - 1][it_xi] * GQTable->Weight[NP - 1][it_eta] * GQTable->Weight[NP - 1][it_zeta];
                double xi = GQTable->Lroots[NP - 1][it_xi];
                double eta = GQTable->Lroots[NP - 1][it_eta];
                double zeta = GQTable->Lroots[NP - 1][it_zeta];
                Vector3N Qi;
                ComputeInternalForcesSingleGQPnt(Qi, xi, eta, zeta, D, ebar, ebardot);

                Fi.noalias() += GQ_weight * Qi;
            }
        }
    }
}

void ChElementHexaANCF_3843_TR04::ComputeInternalForcesSingleGQPnt(Vector3N& Qi,
                                                                   double xi,
                                                                   double eta,
                                                                   double zeta,
                                                                   const ChMatrixNM<double, 6, 6>& D,
                                                                   const Matrix3xN& ebar,
                                                                   const Matrix3xN& ebardot) {
    // Calculate the normalized shape function derivatives with respect to xi, eta, and zeta
    // Note: This will later be modified by J_0xi_Inv to account for a potentially non-straight reference configuration
    MatrixNx3c Sbar_xi_D;
    Calc_Sxi_D(Sbar_xi_D, xi, eta, zeta);

    // Element Jacobian Matrix at the current point (xi, eta, zeta)
    ChMatrix33<double> J_0xi = m_ebar0 * Sbar_xi_D;

    // Modify the normalized shape function derivatives by J_0xi_Inv to account for a potentially non-straight reference
    // configuration
    Sbar_xi_D *= J_0xi.inverse();

    // Calculate the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

    // Calculate the time derivative of the Deformation Gradient at the current point
    ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

    // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation as well as the time derivative
    // of the Green-Lagrange strain tensor in Voigt notation and then multiply it by the element Jacobian
    ChVectorN<double, 6> epsilon_combined;
    epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1) + m_Alpha * F.col(0).dot(Fdot.col(0));
    epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1) + m_Alpha * F.col(1).dot(Fdot.col(1));
    epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1) + m_Alpha * F.col(2).dot(Fdot.col(2));
    epsilon_combined(3) = F.col(1).dot(F.col(2)) + m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
    epsilon_combined(4) = F.col(0).dot(F.col(2)) + m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
    epsilon_combined(5) = F.col(0).dot(F.col(1)) + m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    epsilon_combined *= -J_0xi.determinant();

    // Calculate the 2nd Piola Kirchhoff Stress tensor in Voigt notation
    ChVectorN<double, 6> sigmaPK2_combined = D * epsilon_combined;

    // 2nd Piola Kirchhoff Stress tensor in tensor form
    ChMatrixNM<double, 3, 3> SPK2;
    SPK2(0, 0) = sigmaPK2_combined(0);
    SPK2(1, 1) = sigmaPK2_combined(1);
    SPK2(2, 2) = sigmaPK2_combined(2);
    SPK2(1, 2) = sigmaPK2_combined(3);
    SPK2(2, 1) = sigmaPK2_combined(3);
    SPK2(0, 2) = sigmaPK2_combined(4);
    SPK2(2, 0) = sigmaPK2_combined(4);
    SPK2(0, 1) = sigmaPK2_combined(5);
    SPK2(1, 0) = sigmaPK2_combined(5);

    // Calculate the generalized internal force integrand for a Linear Kelvin-Voigt Viscoelastic material model
    Eigen::Map<MatrixNx3> QiMatrixForm(Qi.data(), NSF, 3);
    QiMatrixForm.noalias() = Sbar_xi_D * (SPK2 * F.transpose());
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_TR04::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) =
        0.0625 * (zeta - 1) * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + zeta * zeta + zeta - 2);
    Sxi_compact(1) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1) * (eta - 1);
    Sxi_compact(2) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1) * (xi - 1);
    Sxi_compact(3) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1) * (eta - 1);
    Sxi_compact(4) =
        -0.0625 * (zeta - 1) * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + zeta * zeta + zeta - 2);
    Sxi_compact(5) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1) * (eta - 1);
    Sxi_compact(6) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1) * (xi + 1);
    Sxi_compact(7) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1) * (eta - 1);
    Sxi_compact(8) =
        0.0625 * (zeta - 1) * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + zeta * zeta + zeta - 2);
    Sxi_compact(9) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1) * (eta + 1);
    Sxi_compact(10) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1) * (xi + 1);
    Sxi_compact(11) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1) * (eta + 1);
    Sxi_compact(12) =
        -0.0625 * (zeta - 1) * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + zeta * zeta + zeta - 2);
    Sxi_compact(13) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1) * (eta + 1);
    Sxi_compact(14) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1) * (xi - 1);
    Sxi_compact(15) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1) * (eta + 1);
    Sxi_compact(16) =
        -0.0625 * (zeta + 1) * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + zeta * zeta - zeta - 2);
    Sxi_compact(17) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1) * (eta - 1);
    Sxi_compact(18) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1) * (xi - 1);
    Sxi_compact(19) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1) * (eta - 1);
    Sxi_compact(20) =
        0.0625 * (zeta + 1) * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + zeta * zeta - zeta - 2);
    Sxi_compact(21) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1) * (eta - 1);
    Sxi_compact(22) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1) * (xi + 1);
    Sxi_compact(23) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1) * (eta - 1);
    Sxi_compact(24) =
        -0.0625 * (zeta + 1) * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + zeta * zeta - zeta - 2);
    Sxi_compact(25) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1) * (eta + 1);
    Sxi_compact(26) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1) * (xi + 1);
    Sxi_compact(27) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1) * (eta + 1);
    Sxi_compact(28) =
        0.0625 * (zeta + 1) * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + zeta * zeta - zeta - 2);
    Sxi_compact(29) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1) * (eta + 1);
    Sxi_compact(30) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1) * (xi - 1);
    Sxi_compact(31) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1) * (eta + 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_TR04::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = 0.0625 * (zeta - 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(1) = 0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(2) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1);
    Sxi_xi_compact(3) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(4) = -0.0625 * (zeta - 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(5) = 0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(6) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta - 1);
    Sxi_xi_compact(7) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta - 1);
    Sxi_xi_compact(8) = 0.0625 * (zeta - 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(9) = -0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(10) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1);
    Sxi_xi_compact(11) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(12) = -0.0625 * (zeta - 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta + zeta - 3);
    Sxi_xi_compact(13) = -0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(14) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta - 1);
    Sxi_xi_compact(15) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (eta + 1);
    Sxi_xi_compact(16) = -0.0625 * (zeta + 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(17) = -0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(18) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1);
    Sxi_xi_compact(19) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(20) = 0.0625 * (zeta + 1) * (eta - 1) * (eta * eta + eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(21) = -0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(22) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (zeta + 1);
    Sxi_xi_compact(23) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta - 1);
    Sxi_xi_compact(24) = -0.0625 * (zeta + 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(25) = 0.03125 * m_lenX * (xi + 1) * (3 * xi - 1) * (zeta + 1) * (eta + 1);
    Sxi_xi_compact(26) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1);
    Sxi_xi_compact(27) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta + 1);
    Sxi_xi_compact(28) = 0.0625 * (zeta + 1) * (eta + 1) * (eta * eta - eta + 3 * xi * xi + zeta * zeta - zeta - 3);
    Sxi_xi_compact(29) = 0.03125 * m_lenX * (3 * xi + 1) * (xi - 1) * (zeta + 1) * (eta + 1);
    Sxi_xi_compact(30) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (zeta + 1);
    Sxi_xi_compact(31) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (eta + 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_TR04::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = 0.0625 * (zeta - 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(1) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1);
    Sxi_eta_compact(2) = 0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(3) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(4) = -0.0625 * (zeta - 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(5) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1);
    Sxi_eta_compact(6) = -0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(7) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(8) = 0.0625 * (zeta - 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(9) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta - 1);
    Sxi_eta_compact(10) = -0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(11) = 0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi + 1);
    Sxi_eta_compact(12) = -0.0625 * (zeta - 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta + zeta - 3);
    Sxi_eta_compact(13) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta - 1);
    Sxi_eta_compact(14) = 0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(15) = -0.03125 * m_lenZ * (zeta + 1) * (zeta - 1) * (zeta - 1) * (xi - 1);
    Sxi_eta_compact(16) = -0.0625 * (zeta + 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(17) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1);
    Sxi_eta_compact(18) = -0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta + 1) * (xi - 1);
    Sxi_eta_compact(19) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1);
    Sxi_eta_compact(20) = 0.0625 * (zeta + 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(21) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1);
    Sxi_eta_compact(22) = 0.03125 * m_lenY * (3 * eta + 1) * (eta - 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(23) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(24) = -0.0625 * (zeta + 1) * (xi + 1) * (3 * eta * eta + xi * xi - xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(25) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (zeta + 1);
    Sxi_eta_compact(26) = 0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(27) = 0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi + 1);
    Sxi_eta_compact(28) = 0.0625 * (zeta + 1) * (xi - 1) * (3 * eta * eta + xi * xi + xi + zeta * zeta - zeta - 3);
    Sxi_eta_compact(29) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (zeta + 1);
    Sxi_eta_compact(30) = -0.03125 * m_lenY * (eta + 1) * (3 * eta - 1) * (zeta + 1) * (xi - 1);
    Sxi_eta_compact(31) = -0.03125 * m_lenZ * (zeta - 1) * (zeta + 1) * (zeta + 1) * (xi - 1);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_TR04::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
    Sxi_zeta_compact(0) = 0.0625 * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(1) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(2) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi - 1);
    Sxi_zeta_compact(3) = 0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(4) = -0.0625 * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(5) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(6) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi + 1);
    Sxi_zeta_compact(7) = -0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(8) = 0.0625 * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(9) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(10) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi + 1);
    Sxi_zeta_compact(11) = 0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(12) = -0.0625 * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(13) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta + 1);
    Sxi_zeta_compact(14) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi - 1);
    Sxi_zeta_compact(15) = -0.03125 * m_lenZ * (3 * zeta + 1) * (zeta - 1) * (xi - 1) * (eta + 1);
    Sxi_zeta_compact(16) = -0.0625 * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(17) = -0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(18) = -0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi - 1);
    Sxi_zeta_compact(19) = 0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi - 1) * (eta - 1);
    Sxi_zeta_compact(20) = 0.0625 * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(21) = -0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(22) = 0.03125 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi + 1);
    Sxi_zeta_compact(23) = -0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi + 1) * (eta - 1);
    Sxi_zeta_compact(24) = -0.0625 * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(25) = 0.03125 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(26) = 0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi + 1);
    Sxi_zeta_compact(27) = 0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi + 1) * (eta + 1);
    Sxi_zeta_compact(28) = 0.0625 * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi + 3 * zeta * zeta - 3);
    Sxi_zeta_compact(29) = 0.03125 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta + 1);
    Sxi_zeta_compact(30) = -0.03125 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi - 1);
    Sxi_zeta_compact(31) = -0.03125 * m_lenZ * (zeta + 1) * (3 * zeta - 1) * (xi - 1) * (eta + 1);
}

// Nx3 compact form of the partial derivatives of Normalized Shape Functions with respect to xi, eta, and zeta by
// columns

void ChElementHexaANCF_3843_TR04::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
    VectorN Sxi_D_col;
    Calc_Sxi_xi_compact(Sxi_D_col, xi, eta, zeta);
    Sxi_D.col(0) = Sxi_D_col;

    Calc_Sxi_eta_compact(Sxi_D_col, xi, eta, zeta);
    Sxi_D.col(1) = Sxi_D_col;

    Calc_Sxi_zeta_compact(Sxi_D_col, xi, eta, zeta);
    Sxi_D.col(2) = Sxi_D_col;
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR04::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    e.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    e.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(15, 3) = m_nodes[1]->GetD().eigen();
    e.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    e.segment(21, 3) = m_nodes[1]->GetDDD().eigen();

    e.segment(24, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(27, 3) = m_nodes[2]->GetD().eigen();
    e.segment(30, 3) = m_nodes[2]->GetDD().eigen();
    e.segment(33, 3) = m_nodes[2]->GetDDD().eigen();

    e.segment(36, 3) = m_nodes[3]->GetPos().eigen();
    e.segment(39, 3) = m_nodes[3]->GetD().eigen();
    e.segment(42, 3) = m_nodes[3]->GetDD().eigen();
    e.segment(45, 3) = m_nodes[3]->GetDDD().eigen();

    e.segment(48, 3) = m_nodes[4]->GetPos().eigen();
    e.segment(51, 3) = m_nodes[4]->GetD().eigen();
    e.segment(54, 3) = m_nodes[4]->GetDD().eigen();
    e.segment(57, 3) = m_nodes[4]->GetDDD().eigen();

    e.segment(60, 3) = m_nodes[5]->GetPos().eigen();
    e.segment(63, 3) = m_nodes[5]->GetD().eigen();
    e.segment(66, 3) = m_nodes[5]->GetDD().eigen();
    e.segment(69, 3) = m_nodes[5]->GetDDD().eigen();

    e.segment(72, 3) = m_nodes[6]->GetPos().eigen();
    e.segment(75, 3) = m_nodes[6]->GetD().eigen();
    e.segment(78, 3) = m_nodes[6]->GetDD().eigen();
    e.segment(81, 3) = m_nodes[6]->GetDDD().eigen();

    e.segment(84, 3) = m_nodes[7]->GetPos().eigen();
    e.segment(87, 3) = m_nodes[7]->GetD().eigen();
    e.segment(90, 3) = m_nodes[7]->GetDD().eigen();
    e.segment(93, 3) = m_nodes[7]->GetDDD().eigen();
}

void ChElementHexaANCF_3843_TR04::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetD().eigen();
    ebar.col(2) = m_nodes[0]->GetDD().eigen();
    ebar.col(3) = m_nodes[0]->GetDDD().eigen();

    ebar.col(4) = m_nodes[1]->GetPos().eigen();
    ebar.col(5) = m_nodes[1]->GetD().eigen();
    ebar.col(6) = m_nodes[1]->GetDD().eigen();
    ebar.col(7) = m_nodes[1]->GetDDD().eigen();

    ebar.col(8) = m_nodes[2]->GetPos().eigen();
    ebar.col(9) = m_nodes[2]->GetD().eigen();
    ebar.col(10) = m_nodes[2]->GetDD().eigen();
    ebar.col(11) = m_nodes[2]->GetDDD().eigen();

    ebar.col(12) = m_nodes[3]->GetPos().eigen();
    ebar.col(13) = m_nodes[3]->GetD().eigen();
    ebar.col(14) = m_nodes[3]->GetDD().eigen();
    ebar.col(15) = m_nodes[3]->GetDDD().eigen();

    ebar.col(16) = m_nodes[4]->GetPos().eigen();
    ebar.col(17) = m_nodes[4]->GetD().eigen();
    ebar.col(18) = m_nodes[4]->GetDD().eigen();
    ebar.col(19) = m_nodes[4]->GetDDD().eigen();

    ebar.col(20) = m_nodes[5]->GetPos().eigen();
    ebar.col(21) = m_nodes[5]->GetD().eigen();
    ebar.col(22) = m_nodes[5]->GetDD().eigen();
    ebar.col(23) = m_nodes[5]->GetDDD().eigen();

    ebar.col(24) = m_nodes[6]->GetPos().eigen();
    ebar.col(25) = m_nodes[6]->GetD().eigen();
    ebar.col(26) = m_nodes[6]->GetDD().eigen();
    ebar.col(27) = m_nodes[6]->GetDDD().eigen();

    ebar.col(28) = m_nodes[7]->GetPos().eigen();
    ebar.col(29) = m_nodes[7]->GetD().eigen();
    ebar.col(30) = m_nodes[7]->GetDD().eigen();
    ebar.col(31) = m_nodes[7]->GetDDD().eigen();
}

void ChElementHexaANCF_3843_TR04::CalcCoordDerivVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();
    edot.segment(9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    edot.segment(12, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(18, 3) = m_nodes[1]->GetDD_dt().eigen();
    edot.segment(21, 3) = m_nodes[1]->GetDDD_dt().eigen();

    edot.segment(24, 3) = m_nodes[2]->GetPos_dt().eigen();
    edot.segment(27, 3) = m_nodes[2]->GetD_dt().eigen();
    edot.segment(30, 3) = m_nodes[2]->GetDD_dt().eigen();
    edot.segment(33, 3) = m_nodes[2]->GetDDD_dt().eigen();

    edot.segment(36, 3) = m_nodes[3]->GetPos_dt().eigen();
    edot.segment(39, 3) = m_nodes[3]->GetD_dt().eigen();
    edot.segment(42, 3) = m_nodes[3]->GetDD_dt().eigen();
    edot.segment(45, 3) = m_nodes[3]->GetDDD_dt().eigen();

    edot.segment(48, 3) = m_nodes[4]->GetPos_dt().eigen();
    edot.segment(51, 3) = m_nodes[4]->GetD_dt().eigen();
    edot.segment(54, 3) = m_nodes[4]->GetDD_dt().eigen();
    edot.segment(57, 3) = m_nodes[4]->GetDDD_dt().eigen();

    edot.segment(60, 3) = m_nodes[5]->GetPos_dt().eigen();
    edot.segment(63, 3) = m_nodes[5]->GetD_dt().eigen();
    edot.segment(66, 3) = m_nodes[5]->GetDD_dt().eigen();
    edot.segment(69, 3) = m_nodes[5]->GetDDD_dt().eigen();

    edot.segment(72, 3) = m_nodes[6]->GetPos_dt().eigen();
    edot.segment(75, 3) = m_nodes[6]->GetD_dt().eigen();
    edot.segment(78, 3) = m_nodes[6]->GetDD_dt().eigen();
    edot.segment(81, 3) = m_nodes[6]->GetDDD_dt().eigen();

    edot.segment(84, 3) = m_nodes[7]->GetPos_dt().eigen();
    edot.segment(87, 3) = m_nodes[7]->GetD_dt().eigen();
    edot.segment(90, 3) = m_nodes[7]->GetDD_dt().eigen();
    edot.segment(93, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

void ChElementHexaANCF_3843_TR04::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.col(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetDD_dt().eigen();
    ebardot.col(3) = m_nodes[0]->GetDDD_dt().eigen();

    ebardot.col(4) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetD_dt().eigen();
    ebardot.col(6) = m_nodes[1]->GetDD_dt().eigen();
    ebardot.col(7) = m_nodes[1]->GetDDD_dt().eigen();

    ebardot.col(8) = m_nodes[2]->GetPos_dt().eigen();
    ebardot.col(9) = m_nodes[2]->GetD_dt().eigen();
    ebardot.col(10) = m_nodes[2]->GetDD_dt().eigen();
    ebardot.col(11) = m_nodes[2]->GetDDD_dt().eigen();

    ebardot.col(12) = m_nodes[3]->GetPos_dt().eigen();
    ebardot.col(13) = m_nodes[3]->GetD_dt().eigen();
    ebardot.col(14) = m_nodes[3]->GetDD_dt().eigen();
    ebardot.col(15) = m_nodes[3]->GetDDD_dt().eigen();

    ebardot.col(16) = m_nodes[4]->GetPos_dt().eigen();
    ebardot.col(17) = m_nodes[4]->GetD_dt().eigen();
    ebardot.col(18) = m_nodes[4]->GetDD_dt().eigen();
    ebardot.col(19) = m_nodes[4]->GetDDD_dt().eigen();

    ebardot.col(20) = m_nodes[5]->GetPos_dt().eigen();
    ebardot.col(21) = m_nodes[5]->GetD_dt().eigen();
    ebardot.col(22) = m_nodes[5]->GetDD_dt().eigen();
    ebardot.col(23) = m_nodes[5]->GetDDD_dt().eigen();

    ebardot.col(24) = m_nodes[6]->GetPos_dt().eigen();
    ebardot.col(25) = m_nodes[6]->GetD_dt().eigen();
    ebardot.col(26) = m_nodes[6]->GetDD_dt().eigen();
    ebardot.col(27) = m_nodes[6]->GetDDD_dt().eigen();

    ebardot.col(28) = m_nodes[7]->GetPos_dt().eigen();
    ebardot.col(29) = m_nodes[7]->GetD_dt().eigen();
    ebardot.col(30) = m_nodes[7]->GetDD_dt().eigen();
    ebardot.col(31) = m_nodes[7]->GetDDD_dt().eigen();
}

void ChElementHexaANCF_3843_TR04::CalcCombinedCoordMatrix(MatrixNx6& ebar_ebardot) {
    ebar_ebardot.template block<1, 3>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(1, 0) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(1, 3) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(2, 0) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(2, 3) = m_nodes[0]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(3, 0) = m_nodes[0]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(3, 3) = m_nodes[0]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(4, 0) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(4, 3) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(5, 0) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(5, 3) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(6, 0) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(6, 3) = m_nodes[1]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(7, 0) = m_nodes[1]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(7, 3) = m_nodes[1]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(8, 0) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(8, 3) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(9, 0) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(9, 3) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(10, 0) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(10, 3) = m_nodes[2]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(11, 0) = m_nodes[2]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(11, 3) = m_nodes[2]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(12, 0) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(12, 3) = m_nodes[3]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(13, 0) = m_nodes[3]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(13, 3) = m_nodes[3]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(14, 0) = m_nodes[3]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(14, 3) = m_nodes[3]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(15, 0) = m_nodes[3]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(15, 3) = m_nodes[3]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(16, 0) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(16, 3) = m_nodes[4]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(17, 0) = m_nodes[4]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(17, 3) = m_nodes[4]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(18, 0) = m_nodes[4]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(18, 3) = m_nodes[4]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(19, 0) = m_nodes[4]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(19, 3) = m_nodes[4]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(20, 0) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(20, 3) = m_nodes[5]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(21, 0) = m_nodes[5]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(21, 3) = m_nodes[5]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(22, 0) = m_nodes[5]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(22, 3) = m_nodes[5]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(23, 0) = m_nodes[5]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(23, 3) = m_nodes[5]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(24, 0) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(24, 3) = m_nodes[6]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(25, 0) = m_nodes[6]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(25, 3) = m_nodes[6]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(26, 0) = m_nodes[6]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(26, 3) = m_nodes[6]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(27, 0) = m_nodes[6]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(27, 3) = m_nodes[6]->GetDDD_dt().eigen();

    ebar_ebardot.template block<1, 3>(28, 0) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.template block<1, 3>(28, 3) = m_nodes[7]->GetPos_dt().eigen();
    ebar_ebardot.template block<1, 3>(29, 0) = m_nodes[7]->GetD().eigen();
    ebar_ebardot.template block<1, 3>(29, 3) = m_nodes[7]->GetD_dt().eigen();
    ebar_ebardot.template block<1, 3>(30, 0) = m_nodes[7]->GetDD().eigen();
    ebar_ebardot.template block<1, 3>(30, 3) = m_nodes[7]->GetDD_dt().eigen();
    ebar_ebardot.template block<1, 3>(31, 0) = m_nodes[7]->GetDDD().eigen();
    ebar_ebardot.template block<1, 3>(31, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementHexaANCF_3843_TR04::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementHexaANCF_3843_TR04::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3843_TR04(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementHexaANCF_3843_TR04::GetStaticGQTables() {
    return &static_tables_3843_TR04;
}

}  // namespace fea
}  // namespace chrono
