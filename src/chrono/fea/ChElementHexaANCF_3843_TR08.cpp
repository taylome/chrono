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
// TR08 = Vectorized Continuous Integration without Data Caching for the Jacobian
// =============================================================================
// Mass Matrix = Compact Upper Triangular
// Reduced Number of GQ Points
// Nodal Coordinates in Matrix Form
// PK1 Stress
// Precomputed Adjusted Shape Function Derivatives and minus Element Jacobians time the corresponding GQ Weight
// Analytic Jacobian without calculations cached from the internal force calculation
// =============================================================================

#include "chrono/fea/ChElementHexaANCF_3843_TR08.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementHexaANCF_3843_TR08::ChElementHexaANCF_3843_TR08()
    : m_lenX(0), m_lenY(0), m_lenZ(0), m_Alpha(0), m_damping_enabled(false), m_skipPrecomputation(false) {
    m_nodes.resize(8);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR08::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
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
    mvars.push_back(&m_nodes[0]->VariablesSlope1());
    mvars.push_back(&m_nodes[0]->VariablesSlope2());
    mvars.push_back(&m_nodes[0]->VariablesSlope3());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->VariablesSlope1());
    mvars.push_back(&m_nodes[1]->VariablesSlope2());
    mvars.push_back(&m_nodes[1]->VariablesSlope3());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->VariablesSlope1());
    mvars.push_back(&m_nodes[2]->VariablesSlope2());
    mvars.push_back(&m_nodes[2]->VariablesSlope3());
    mvars.push_back(&m_nodes[3]->Variables());
    mvars.push_back(&m_nodes[3]->VariablesSlope1());
    mvars.push_back(&m_nodes[3]->VariablesSlope2());
    mvars.push_back(&m_nodes[3]->VariablesSlope3());
    mvars.push_back(&m_nodes[4]->Variables());
    mvars.push_back(&m_nodes[4]->VariablesSlope1());
    mvars.push_back(&m_nodes[4]->VariablesSlope2());
    mvars.push_back(&m_nodes[4]->VariablesSlope3());
    mvars.push_back(&m_nodes[5]->Variables());
    mvars.push_back(&m_nodes[5]->VariablesSlope1());
    mvars.push_back(&m_nodes[5]->VariablesSlope2());
    mvars.push_back(&m_nodes[5]->VariablesSlope3());
    mvars.push_back(&m_nodes[6]->Variables());
    mvars.push_back(&m_nodes[6]->VariablesSlope1());
    mvars.push_back(&m_nodes[6]->VariablesSlope2());
    mvars.push_back(&m_nodes[6]->VariablesSlope3());
    mvars.push_back(&m_nodes[7]->Variables());
    mvars.push_back(&m_nodes[7]->VariablesSlope1());
    mvars.push_back(&m_nodes[7]->VariablesSlope2());
    mvars.push_back(&m_nodes[7]->VariablesSlope3());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions.

void ChElementHexaANCF_3843_TR08::SetDimensions(double lenX, double lenY, double lenZ) {
    m_lenX = lenX;
    m_lenY = lenY;
    m_lenZ = lenZ;
}

// Specify the element material.

void ChElementHexaANCF_3843_TR08::SetMaterial(std::shared_ptr<ChMaterialHexaANCF> brick_mat) {
    m_material = brick_mat;
}

// Set the value for the single term structural damping coefficient.

void ChElementHexaANCF_3843_TR08::SetAlphaDamp(double a) {
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

ChMatrix33d ChElementHexaANCF_3843_TR08::GetGreenLagrangeStrain(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> F = e_bar * Sxi_D;

    ChMatrix33d I3x3;
    I3x3.setIdentity();
    return 0.5 * (F.transpose() * F - I3x3);
}

// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) [-1...1] at the
// current state of the element.

ChMatrix33d ChElementHexaANCF_3843_TR08::GetPK2Stress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> F = e_bar * Sxi_D;

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
        CalcCoordDtMatrix(ebardot);

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNM_col<double, 3, 3> Fdot = ebardot * Sxi_D;

        // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
        epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    }

    const ChMatrix66d& D = GetMaterial()->Get_D();

    ChVectorN<double, 6> sigmaPK2 = D * epsilon_combined;  // 2nd Piola Kirchhoff Stress tensor in Voigt notation

    ChMatrix33d SPK2;
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

double ChElementHexaANCF_3843_TR08::GetVonMissesStress(const double xi, const double eta, const double zeta) {
    MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
    J_0xi.noalias() = m_ebar0 * Sxi_D;

    Sxi_D = Sxi_D * J_0xi.inverse();  // Adjust the shape function derivative matrix to account for the potentially
                                      // distorted reference configuration

    Matrix3xN e_bar;  // Element coordinates in matrix form
    CalcCoordMatrix(e_bar);

    // Calculate the Deformation Gradient at the current point
    ChMatrixNM_col<double, 3, 3> F = e_bar * Sxi_D;

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
        CalcCoordDtMatrix(ebardot);

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNM_col<double, 3, 3> Fdot = ebardot * Sxi_D;

        // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
        epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
    }

    const ChMatrix66d& D = GetMaterial()->Get_D();

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
    ChMatrix33d S = (F * SPK2 * F.transpose()) / F.determinant();
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

void ChElementHexaANCF_3843_TR08::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementHexaANCF_3843_TR08::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetSlope1().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetSlope2().eigen();
    mD.segment(9, 3) = m_nodes[0]->GetSlope3().eigen();

    mD.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetSlope1().eigen();
    mD.segment(18, 3) = m_nodes[1]->GetSlope2().eigen();
    mD.segment(21, 3) = m_nodes[1]->GetSlope3().eigen();

    mD.segment(24, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(27, 3) = m_nodes[2]->GetSlope1().eigen();
    mD.segment(30, 3) = m_nodes[2]->GetSlope2().eigen();
    mD.segment(33, 3) = m_nodes[2]->GetSlope3().eigen();

    mD.segment(36, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(39, 3) = m_nodes[3]->GetSlope1().eigen();
    mD.segment(42, 3) = m_nodes[3]->GetSlope2().eigen();
    mD.segment(45, 3) = m_nodes[3]->GetSlope3().eigen();

    mD.segment(48, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(51, 3) = m_nodes[4]->GetSlope1().eigen();
    mD.segment(54, 3) = m_nodes[4]->GetSlope2().eigen();
    mD.segment(57, 3) = m_nodes[4]->GetSlope3().eigen();

    mD.segment(60, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(63, 3) = m_nodes[5]->GetSlope1().eigen();
    mD.segment(66, 3) = m_nodes[5]->GetSlope2().eigen();
    mD.segment(69, 3) = m_nodes[5]->GetSlope3().eigen();

    mD.segment(72, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(75, 3) = m_nodes[6]->GetSlope1().eigen();
    mD.segment(78, 3) = m_nodes[6]->GetSlope2().eigen();
    mD.segment(81, 3) = m_nodes[6]->GetSlope3().eigen();

    mD.segment(84, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(87, 3) = m_nodes[7]->GetSlope1().eigen();
    mD.segment(90, 3) = m_nodes[7]->GetSlope2().eigen();
    mD.segment(93, 3) = m_nodes[7]->GetSlope3().eigen();
}

// State update.

void ChElementHexaANCF_3843_TR08::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementHexaANCF_3843_TR08::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Mass Matrix is Stored in Compact Upper Triangular Form
    // Expand it out into its Full Sparse Symmetric Form
    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            M(3 * i, 3 * j) = m_MassMatrix(idx);
            M(3 * i + 1, 3 * j + 1) = m_MassMatrix(idx);
            M(3 * i + 2, 3 * j + 2) = m_MassMatrix(idx);
            if (i != j) {
                M(3 * j, 3 * i) = m_MassMatrix(idx);
                M(3 * j + 1, 3 * i + 1) = m_MassMatrix(idx);
                M(3 * j + 2, 3 * i + 2) = m_MassMatrix(idx);
            }
            idx++;
        }
    }
}

// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass

void ChElementHexaANCF_3843_TR08::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4) + m_MassMatrix(8) + m_MassMatrix(12) +
                               m_MassMatrix(16) + m_MassMatrix(20) + m_MassMatrix(24) + m_MassMatrix(28);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(122) + m_MassMatrix(126) + m_MassMatrix(130) +
                               m_MassMatrix(134) + m_MassMatrix(138) + m_MassMatrix(142) + m_MassMatrix(146);
    m_nodes[2]->m_TotalMass += m_MassMatrix(8) + m_MassMatrix(126) + m_MassMatrix(228) + m_MassMatrix(232) +
                               m_MassMatrix(236) + m_MassMatrix(240) + m_MassMatrix(244) + m_MassMatrix(248);
    m_nodes[3]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(130) + m_MassMatrix(232) + m_MassMatrix(318) +
                               m_MassMatrix(322) + m_MassMatrix(326) + m_MassMatrix(330) + m_MassMatrix(334);
    m_nodes[4]->m_TotalMass += m_MassMatrix(16) + m_MassMatrix(134) + m_MassMatrix(236) + m_MassMatrix(322) +
                               m_MassMatrix(392) + m_MassMatrix(396) + m_MassMatrix(400) + m_MassMatrix(404);
    m_nodes[5]->m_TotalMass += m_MassMatrix(20) + m_MassMatrix(138) + m_MassMatrix(240) + m_MassMatrix(326) +
                               m_MassMatrix(396) + m_MassMatrix(450) + m_MassMatrix(454) + m_MassMatrix(458);
    m_nodes[6]->m_TotalMass += m_MassMatrix(24) + m_MassMatrix(142) + m_MassMatrix(244) + m_MassMatrix(330) +
                               m_MassMatrix(400) + m_MassMatrix(454) + m_MassMatrix(492) + m_MassMatrix(496);
    m_nodes[7]->m_TotalMass += m_MassMatrix(28) + m_MassMatrix(146) + m_MassMatrix(248) + m_MassMatrix(334) +
                               m_MassMatrix(404) + m_MassMatrix(458) + m_MassMatrix(496) + m_MassMatrix(518);
}

void ChElementHexaANCF_3843_TR08::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert((unsigned int)Fi.size() == GetNumCoordsPosLevel());

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    //      [F11     F12     F13    ]
    //      [F21     F22     F23    ]
    // FC = [F31     F32     F33    ]
    //      [Fdot11  Fdot12  Fdot13 ]
    //      [Fdot21  Fdot22  Fdot23 ]
    //      [Fdot31  Fdot32  Fdot33 ]
    // =============================================================================

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;  // ChMatrixNM<double, 6, 3 * NIP>

    Eigen::Map<ArrayNIP> F11(FC.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F12(FC.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F13(FC.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F21(FC.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F22(FC.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F23(FC.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F31(FC.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F32(FC.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F33(FC.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot11(FC.block<1, NIP>(3, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot12(FC.block<1, NIP>(3, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot13(FC.block<1, NIP>(3, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot21(FC.block<1, NIP>(4, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot22(FC.block<1, NIP>(4, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot23(FC.block<1, NIP>(4, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot31(FC.block<1, NIP>(5, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot32(FC.block<1, NIP>(5, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot33(FC.block<1, NIP>(5, 2 * NIP).data(), 1, NIP);

    // =============================================================================
    // Calculate the 2nd Piola-Kirchhoff Stresses in Voigt notation across all of the GQ Points at the same time
    // =============================================================================

    // Get the components of the stiffness tensor in 6x6 matrix form
    const ChMatrix66d& D = GetMaterial()->Get_D();

    ArrayNIP epsilon_1 = m_kGQ * ((0.5 * (F11 * F11 + F21 * F21 + F31 * F31) - 0.5) +
                                  m_Alpha * (F11 * Fdot11 + F21 * Fdot21 + F31 * Fdot31));
    ArrayNIP epsilon_2 = m_kGQ * ((0.5 * (F12 * F12 + F22 * F22 + F32 * F32) - 0.5) +
                                  m_Alpha * (F12 * Fdot12 + F22 * Fdot22 + F32 * Fdot32));
    ArrayNIP epsilon_3 = m_kGQ * ((0.5 * (F13 * F13 + F23 * F23 + F33 * F33) - 0.5) +
                                  m_Alpha * (F13 * Fdot13 + F23 * Fdot23 + F33 * Fdot33));
    ArrayNIP epsilon_4 =
        m_kGQ * ((F12 * F13 + F22 * F23 + F32 * F33) +
                 m_Alpha * (F12 * Fdot13 + F22 * Fdot23 + F32 * Fdot33 + Fdot12 * F13 + Fdot22 * F23 + Fdot32 * F33));
    ArrayNIP epsilon_5 =
        m_kGQ * ((F11 * F13 + F21 * F23 + F31 * F33) +
                 m_Alpha * (F11 * Fdot13 + F21 * Fdot23 + F31 * Fdot33 + Fdot11 * F13 + Fdot21 * F23 + Fdot31 * F33));
    ArrayNIP epsilon_6 =
        m_kGQ * ((F11 * F12 + F21 * F22 + F31 * F32) +
                 m_Alpha * (F11 * Fdot12 + F21 * Fdot22 + F31 * Fdot32 + Fdot11 * F12 + Fdot21 * F22 + Fdot31 * F32));

    ArrayNIP SPK2_1 = D(0, 0) * epsilon_1 + D(0, 1) * epsilon_2 + D(0, 2) * epsilon_3 + D(0, 3) * epsilon_4 +
                      D(0, 4) * epsilon_5 + D(0, 5) * epsilon_6;
    ArrayNIP SPK2_2 = D(1, 0) * epsilon_1 + D(1, 1) * epsilon_2 + D(1, 2) * epsilon_3 + D(1, 3) * epsilon_4 +
                      D(1, 4) * epsilon_5 + D(1, 5) * epsilon_6;
    ArrayNIP SPK2_3 = D(2, 0) * epsilon_1 + D(2, 1) * epsilon_2 + D(2, 2) * epsilon_3 + D(2, 3) * epsilon_4 +
                      D(2, 4) * epsilon_5 + D(2, 5) * epsilon_6;
    ArrayNIP SPK2_4 = D(3, 0) * epsilon_1 + D(3, 1) * epsilon_2 + D(3, 2) * epsilon_3 + D(3, 3) * epsilon_4 +
                      D(3, 4) * epsilon_5 + D(3, 5) * epsilon_6;
    ArrayNIP SPK2_5 = D(4, 0) * epsilon_1 + D(4, 1) * epsilon_2 + D(4, 2) * epsilon_3 + D(4, 3) * epsilon_4 +
                      D(4, 4) * epsilon_5 + D(4, 5) * epsilon_6;
    ArrayNIP SPK2_6 = D(5, 0) * epsilon_1 + D(5, 1) * epsilon_2 + D(5, 2) * epsilon_3 + D(5, 3) * epsilon_4 +
                      D(5, 4) * epsilon_5 + D(5, 5) * epsilon_6;

    // =============================================================================
    // Calculate the transpose of the 1st Piola-Kirchoff stresses in block tensor form whose entries have been
    // scaled by minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss point.
    // The entries are grouped by component in block matrices (column vectors)
    // P_Block = kGQ*P_transpose = kGQ*SPK2*F_transpose
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ]
    //         = [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ]
    //           [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ]
    // =============================================================================

    ChMatrixNM_col<double, 3 * NIP, 3> P_Block;

    P_Block.block<NIP, 1>(0 * NIP, 0).array().transpose() = F11 * SPK2_1 + F12 * SPK2_6 + F13 * SPK2_5;  // PT11
    P_Block.block<NIP, 1>(1 * NIP, 0).array().transpose() = F11 * SPK2_6 + F12 * SPK2_2 + F13 * SPK2_4;  // PT21
    P_Block.block<NIP, 1>(2 * NIP, 0).array().transpose() = F11 * SPK2_5 + F12 * SPK2_4 + F13 * SPK2_3;  // PT31

    P_Block.block<NIP, 1>(0 * NIP, 1).array().transpose() = F21 * SPK2_1 + F22 * SPK2_6 + F23 * SPK2_5;  // PT12
    P_Block.block<NIP, 1>(1 * NIP, 1).array().transpose() = F21 * SPK2_6 + F22 * SPK2_2 + F23 * SPK2_4;  // PT22
    P_Block.block<NIP, 1>(2 * NIP, 1).array().transpose() = F21 * SPK2_5 + F22 * SPK2_4 + F23 * SPK2_3;  // PT32

    P_Block.block<NIP, 1>(0 * NIP, 2).array().transpose() = F31 * SPK2_1 + F32 * SPK2_6 + F33 * SPK2_5;  // PT13
    P_Block.block<NIP, 1>(1 * NIP, 2).array().transpose() = F31 * SPK2_6 + F32 * SPK2_2 + F33 * SPK2_4;  // PT23
    P_Block.block<NIP, 1>(2 * NIP, 2).array().transpose() = F31 * SPK2_5 + F32 * SPK2_4 + F33 * SPK2_3;  // PT33

    // =============================================================================
    // Multiply the scaled first Piola-Kirchoff stresses by the shape function derivative matrix to get the generalized
    // force vector in matrix form (in the correct order if its calculated in row-major memory layout)
    // =============================================================================

    MatrixNx3 QiCompact = m_SD * P_Block;
    Eigen::Map<ChVectorN<double, 3 * NSF>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi.noalias() = QiReshaped;
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]

void ChElementHexaANCF_3843_TR08::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                           double Kfactor,
                                                           double Rfactor,
                                                           double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    //      [F11     F12     F13    ]
    //      [F21     F22     F23    ]
    // FC = [F31     F32     F33    ]
    //      [Fdot11  Fdot12  Fdot13 ]
    //      [Fdot21  Fdot22  Fdot23 ]
    //      [Fdot31  Fdot32  Fdot33 ]
    // =============================================================================
    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;  // ChMatrixNM<double, 6, 3 * NIP>

    Eigen::Map<ArrayNIP> F11(FC.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F12(FC.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F13(FC.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F21(FC.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F22(FC.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F23(FC.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> F31(FC.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F32(FC.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> F33(FC.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot11(FC.block<1, NIP>(3, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot12(FC.block<1, NIP>(3, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot13(FC.block<1, NIP>(3, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot21(FC.block<1, NIP>(4, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot22(FC.block<1, NIP>(4, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot23(FC.block<1, NIP>(4, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> Fdot31(FC.block<1, NIP>(5, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot32(FC.block<1, NIP>(5, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> Fdot33(FC.block<1, NIP>(5, 2 * NIP).data(), 1, NIP);

    ChMatrixNM<double, 3, 3 * NIP> FCS =
        (-Kfactor - m_Alpha * Rfactor) * FC.block<3, 3 * NIP>(0, 0) - (m_Alpha * Kfactor) * FC.block<3, 3 * NIP>(3, 0);
    for (auto i = 0; i < 3; i++) {
        FCS.block<1, NIP>(i, 0 * NIP).array() *= m_kGQ;
        FCS.block<1, NIP>(i, 1 * NIP).array() *= m_kGQ;
        FCS.block<1, NIP>(i, 2 * NIP).array() *= m_kGQ;
    }

    Eigen::Map<ArrayNIP> FS11(FCS.block<1, NIP>(0, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> FS12(FCS.block<1, NIP>(0, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> FS13(FCS.block<1, NIP>(0, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> FS21(FCS.block<1, NIP>(1, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> FS22(FCS.block<1, NIP>(1, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> FS23(FCS.block<1, NIP>(1, 2 * NIP).data(), 1, NIP);

    Eigen::Map<ArrayNIP> FS31(FCS.block<1, NIP>(2, 0 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> FS32(FCS.block<1, NIP>(2, 1 * NIP).data(), 1, NIP);
    Eigen::Map<ArrayNIP> FS33(FCS.block<1, NIP>(2, 2 * NIP).data(), 1, NIP);

    // Get the components of the stiffness tensor in 6x6 matrix form
    const ChMatrix66d& D = GetMaterial()->Get_D();

    // For this element, it is faster to calculate the depsilon'/de * D * depsilon/de terms in chunks rather than as one
    // very large matrix multiplication (outer product)
    ChMatrixNM<double, 3 * NSF, NIP> PE;
    ChMatrixNM<double, 3 * NSF, NIP> DPE;

    Matrix3Nx3N Jac;
    // Calculate the Contribution from the depsilon1/de terms
    {
        ArrayNIP S1A = D(0, 0) * FS11 + D(0, 5) * FS12 + D(0, 4) * FS13;
        ArrayNIP S2A = D(0, 5) * FS11 + D(0, 1) * FS12 + D(0, 3) * FS13;
        ArrayNIP S3A = D(0, 4) * FS11 + D(0, 3) * FS12 + D(0, 2) * FS13;

        ArrayNIP S1B = D(0, 0) * FS21 + D(0, 5) * FS22 + D(0, 4) * FS23;
        ArrayNIP S2B = D(0, 5) * FS21 + D(0, 1) * FS22 + D(0, 3) * FS23;
        ArrayNIP S3B = D(0, 4) * FS21 + D(0, 3) * FS22 + D(0, 2) * FS23;

        ArrayNIP S1C = D(0, 0) * FS31 + D(0, 5) * FS32 + D(0, 4) * FS33;
        ArrayNIP S2C = D(0, 5) * FS31 + D(0, 1) * FS32 + D(0, 3) * FS33;
        ArrayNIP S3C = D(0, 4) * FS31 + D(0, 3) * FS32 + D(0, 2) * FS33;

        for (auto i = 0; i < NSF; i++) {
            PE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F11;
            PE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F21;
            PE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * F31;
            DPE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            DPE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            DPE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() = PE * DPE.transpose();
    }
    // Calculate the Contribution from the depsilon2/de terms
    {
        ArrayNIP S1A = D(1, 0) * FS11 + D(1, 5) * FS12 + D(1, 4) * FS13;
        ArrayNIP S2A = D(1, 5) * FS11 + D(1, 1) * FS12 + D(1, 3) * FS13;
        ArrayNIP S3A = D(1, 4) * FS11 + D(1, 3) * FS12 + D(1, 2) * FS13;

        ArrayNIP S1B = D(1, 0) * FS21 + D(1, 5) * FS22 + D(1, 4) * FS23;
        ArrayNIP S2B = D(1, 5) * FS21 + D(1, 1) * FS22 + D(1, 3) * FS23;
        ArrayNIP S3B = D(1, 4) * FS21 + D(1, 3) * FS22 + D(1, 2) * FS23;

        ArrayNIP S1C = D(1, 0) * FS31 + D(1, 5) * FS32 + D(1, 4) * FS33;
        ArrayNIP S2C = D(1, 5) * FS31 + D(1, 1) * FS32 + D(1, 3) * FS33;
        ArrayNIP S3C = D(1, 4) * FS31 + D(1, 3) * FS32 + D(1, 2) * FS33;

        for (auto i = 0; i < NSF; i++) {
            PE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F12;
            PE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F22;
            PE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 1 * NIP).array() * F32;
            DPE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            DPE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            DPE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += PE * DPE.transpose();
    }
    // Calculate the Contribution from the depsilon3/de terms
    {
        ArrayNIP S1A = D(2, 0) * FS11 + D(2, 5) * FS12 + D(2, 4) * FS13;
        ArrayNIP S2A = D(2, 5) * FS11 + D(2, 1) * FS12 + D(2, 3) * FS13;
        ArrayNIP S3A = D(2, 4) * FS11 + D(2, 3) * FS12 + D(2, 2) * FS13;

        ArrayNIP S1B = D(2, 0) * FS21 + D(2, 5) * FS22 + D(2, 4) * FS23;
        ArrayNIP S2B = D(2, 5) * FS21 + D(2, 1) * FS22 + D(2, 3) * FS23;
        ArrayNIP S3B = D(2, 4) * FS21 + D(2, 3) * FS22 + D(2, 2) * FS23;

        ArrayNIP S1C = D(2, 0) * FS31 + D(2, 5) * FS32 + D(2, 4) * FS33;
        ArrayNIP S2C = D(2, 5) * FS31 + D(2, 1) * FS32 + D(2, 3) * FS33;
        ArrayNIP S3C = D(2, 4) * FS31 + D(2, 3) * FS32 + D(2, 2) * FS33;

        for (auto i = 0; i < NSF; i++) {
            PE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F13;
            PE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F23;
            PE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 2 * NIP).array() * F33;
            DPE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            DPE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            DPE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += PE * DPE.transpose();
    }
    // Calculate the Contribution from the depsilon4/de terms
    {
        ArrayNIP S1A = D(3, 0) * FS11 + D(3, 5) * FS12 + D(3, 4) * FS13;
        ArrayNIP S2A = D(3, 5) * FS11 + D(3, 1) * FS12 + D(3, 3) * FS13;
        ArrayNIP S3A = D(3, 4) * FS11 + D(3, 3) * FS12 + D(3, 2) * FS13;

        ArrayNIP S1B = D(3, 0) * FS21 + D(3, 5) * FS22 + D(3, 4) * FS23;
        ArrayNIP S2B = D(3, 5) * FS21 + D(3, 1) * FS22 + D(3, 3) * FS23;
        ArrayNIP S3B = D(3, 4) * FS21 + D(3, 3) * FS22 + D(3, 2) * FS23;

        ArrayNIP S1C = D(3, 0) * FS31 + D(3, 5) * FS32 + D(3, 4) * FS33;
        ArrayNIP S2C = D(3, 5) * FS31 + D(3, 1) * FS32 + D(3, 3) * FS33;
        ArrayNIP S3C = D(3, 4) * FS31 + D(3, 3) * FS32 + D(3, 2) * FS33;

        for (auto i = 0; i < NSF; i++) {
            PE.block<1, NIP>(3 * i + 0, 0).array() =
                m_SD.block<1, NIP>(i, 1 * NIP).array() * F13 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F12;
            PE.block<1, NIP>(3 * i + 1, 0).array() =
                m_SD.block<1, NIP>(i, 1 * NIP).array() * F23 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F22;
            PE.block<1, NIP>(3 * i + 2, 0).array() =
                m_SD.block<1, NIP>(i, 1 * NIP).array() * F33 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F32;
            DPE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            DPE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            DPE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += PE * DPE.transpose();
    }
    // Calculate the Contribution from the depsilon5/de terms
    {
        ArrayNIP S1A = D(4, 0) * FS11 + D(4, 5) * FS12 + D(4, 4) * FS13;
        ArrayNIP S2A = D(4, 5) * FS11 + D(4, 1) * FS12 + D(4, 3) * FS13;
        ArrayNIP S3A = D(4, 4) * FS11 + D(4, 3) * FS12 + D(4, 2) * FS13;

        ArrayNIP S1B = D(4, 0) * FS21 + D(4, 5) * FS22 + D(4, 4) * FS23;
        ArrayNIP S2B = D(4, 5) * FS21 + D(4, 1) * FS22 + D(4, 3) * FS23;
        ArrayNIP S3B = D(4, 4) * FS21 + D(4, 3) * FS22 + D(4, 2) * FS23;

        ArrayNIP S1C = D(4, 0) * FS31 + D(4, 5) * FS32 + D(4, 4) * FS33;
        ArrayNIP S2C = D(4, 5) * FS31 + D(4, 1) * FS32 + D(4, 3) * FS33;
        ArrayNIP S3C = D(4, 4) * FS31 + D(4, 3) * FS32 + D(4, 2) * FS33;

        for (auto i = 0; i < NSF; i++) {
            PE.block<1, NIP>(3 * i + 0, 0).array() =
                m_SD.block<1, NIP>(i, 0 * NIP).array() * F13 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F11;
            PE.block<1, NIP>(3 * i + 1, 0).array() =
                m_SD.block<1, NIP>(i, 0 * NIP).array() * F23 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F21;
            PE.block<1, NIP>(3 * i + 2, 0).array() =
                m_SD.block<1, NIP>(i, 0 * NIP).array() * F33 + m_SD.block<1, NIP>(i, 2 * NIP).array() * F31;
            DPE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            DPE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            DPE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += PE * DPE.transpose();
    }
    // Calculate the Contribution from the depsilon6/de terms
    {
        ArrayNIP S1A = D(5, 0) * FS11 + D(5, 5) * FS12 + D(5, 4) * FS13;
        ArrayNIP S2A = D(5, 5) * FS11 + D(5, 1) * FS12 + D(5, 3) * FS13;
        ArrayNIP S3A = D(5, 4) * FS11 + D(5, 3) * FS12 + D(5, 2) * FS13;

        ArrayNIP S1B = D(5, 0) * FS21 + D(5, 5) * FS22 + D(5, 4) * FS23;
        ArrayNIP S2B = D(5, 5) * FS21 + D(5, 1) * FS22 + D(5, 3) * FS23;
        ArrayNIP S3B = D(5, 4) * FS21 + D(5, 3) * FS22 + D(5, 2) * FS23;

        ArrayNIP S1C = D(5, 0) * FS31 + D(5, 5) * FS32 + D(5, 4) * FS33;
        ArrayNIP S2C = D(5, 5) * FS31 + D(5, 1) * FS32 + D(5, 3) * FS33;
        ArrayNIP S3C = D(5, 4) * FS31 + D(5, 3) * FS32 + D(5, 2) * FS33;

        for (auto i = 0; i < NSF; i++) {
            PE.block<1, NIP>(3 * i + 0, 0).array() =
                m_SD.block<1, NIP>(i, 0 * NIP).array() * F12 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F11;
            PE.block<1, NIP>(3 * i + 1, 0).array() =
                m_SD.block<1, NIP>(i, 0 * NIP).array() * F22 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F21;
            PE.block<1, NIP>(3 * i + 2, 0).array() =
                m_SD.block<1, NIP>(i, 0 * NIP).array() * F32 + m_SD.block<1, NIP>(i, 1 * NIP).array() * F31;
            DPE.block<1, NIP>(3 * i + 0, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1A +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2A +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3A;
            DPE.block<1, NIP>(3 * i + 1, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1B +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2B +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3B;
            DPE.block<1, NIP>(3 * i + 2, 0).array() = m_SD.block<1, NIP>(i, 0 * NIP).array() * S1C +
                                                      m_SD.block<1, NIP>(i, 1 * NIP).array() * S2C +
                                                      m_SD.block<1, NIP>(i, 2 * NIP).array() * S3C;
        }
        Jac.noalias() += PE * DPE.transpose();
    }

    // Calculate the PK2 stresses since they were not cached from the internal force calculation
    ArrayNIP epsilon_1 = m_kGQ * ((0.5 * (F11 * F11 + F21 * F21 + F31 * F31) - 0.5) +
                                  m_Alpha * (F11 * Fdot11 + F21 * Fdot21 + F31 * Fdot31));
    ArrayNIP epsilon_2 = m_kGQ * ((0.5 * (F12 * F12 + F22 * F22 + F32 * F32) - 0.5) +
                                  m_Alpha * (F12 * Fdot12 + F22 * Fdot22 + F32 * Fdot32));
    ArrayNIP epsilon_3 = m_kGQ * ((0.5 * (F13 * F13 + F23 * F23 + F33 * F33) - 0.5) +
                                  m_Alpha * (F13 * Fdot13 + F23 * Fdot23 + F33 * Fdot33));
    ArrayNIP epsilon_4 =
        m_kGQ * ((F12 * F13 + F22 * F23 + F32 * F33) +
                 m_Alpha * (F12 * Fdot13 + F22 * Fdot23 + F32 * Fdot33 + Fdot12 * F13 + Fdot22 * F23 + Fdot32 * F33));
    ArrayNIP epsilon_5 =
        m_kGQ * ((F11 * F13 + F21 * F23 + F31 * F33) +
                 m_Alpha * (F11 * Fdot13 + F21 * Fdot23 + F31 * Fdot33 + Fdot11 * F13 + Fdot21 * F23 + Fdot31 * F33));
    ArrayNIP epsilon_6 =
        m_kGQ * ((F11 * F12 + F21 * F22 + F31 * F32) +
                 m_Alpha * (F11 * Fdot12 + F21 * Fdot22 + F31 * Fdot32 + Fdot11 * F12 + Fdot21 * F22 + Fdot31 * F32));

    ArrayNIP SPK2_1 = D(0, 0) * epsilon_1 + D(0, 1) * epsilon_2 + D(0, 2) * epsilon_3 + D(0, 3) * epsilon_4 +
                      D(0, 4) * epsilon_5 + D(0, 5) * epsilon_6;
    ArrayNIP SPK2_2 = D(1, 0) * epsilon_1 + D(1, 1) * epsilon_2 + D(1, 2) * epsilon_3 + D(1, 3) * epsilon_4 +
                      D(1, 4) * epsilon_5 + D(1, 5) * epsilon_6;
    ArrayNIP SPK2_3 = D(2, 0) * epsilon_1 + D(2, 1) * epsilon_2 + D(2, 2) * epsilon_3 + D(2, 3) * epsilon_4 +
                      D(2, 4) * epsilon_5 + D(2, 5) * epsilon_6;
    ArrayNIP SPK2_4 = D(3, 0) * epsilon_1 + D(3, 1) * epsilon_2 + D(3, 2) * epsilon_3 + D(3, 3) * epsilon_4 +
                      D(3, 4) * epsilon_5 + D(3, 5) * epsilon_6;
    ArrayNIP SPK2_5 = D(4, 0) * epsilon_1 + D(4, 1) * epsilon_2 + D(4, 2) * epsilon_3 + D(4, 3) * epsilon_4 +
                      D(4, 4) * epsilon_5 + D(4, 5) * epsilon_6;
    ArrayNIP SPK2_6 = D(5, 0) * epsilon_1 + D(5, 1) * epsilon_2 + D(5, 2) * epsilon_3 + D(5, 3) * epsilon_4 +
                      D(5, 4) * epsilon_5 + D(5, 5) * epsilon_6;

    // Calculate the contribution from the Mass Matrix and expand(SD*SPK2*SD')
    // Since both a symmetric and in compact form, only the upper triangular entries need to be calculated and then
    // added to the correct corresponding locations in the Jacobian matrix
    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        ChVectorN<double, 3 * NIP> S_scaled_SD_row_i;
        S_scaled_SD_row_i.segment(0 * NIP, NIP).array() = SPK2_1 * m_SD.block<1, NIP>(i, 0 * NIP).array() +
                                                          SPK2_6 * m_SD.block<1, NIP>(i, 1 * NIP).array() +
                                                          SPK2_5 * m_SD.block<1, NIP>(i, 2 * NIP).array();
        S_scaled_SD_row_i.segment(1 * NIP, NIP).array() = SPK2_6 * m_SD.block<1, NIP>(i, 0 * NIP).array() +
                                                          SPK2_2 * m_SD.block<1, NIP>(i, 1 * NIP).array() +
                                                          SPK2_4 * m_SD.block<1, NIP>(i, 2 * NIP).array();
        S_scaled_SD_row_i.segment(2 * NIP, NIP).array() = SPK2_5 * m_SD.block<1, NIP>(i, 0 * NIP).array() +
                                                          SPK2_4 * m_SD.block<1, NIP>(i, 1 * NIP).array() +
                                                          SPK2_3 * m_SD.block<1, NIP>(i, 2 * NIP).array();

        for (unsigned int j = i; j < NSF; j++) {
            double d = Mfactor * m_MassMatrix(idx) - Kfactor * (S_scaled_SD_row_i.dot(m_SD.row(j)));

            Jac(3 * i, 3 * j) += d;
            Jac(3 * i + 1, 3 * j + 1) += d;
            Jac(3 * i + 2, 3 * j + 2) += d;
            if (i != j) {
                Jac(3 * j, 3 * i) += d;
                Jac(3 * j + 1, 3 * i + 1) += d;
                Jac(3 * j + 2, 3 * i + 2) += d;
            }
            idx++;
        }
    }

    H.noalias() = Jac;
}

// Compute the generalized force vector due to gravity using the efficient ANCF specific method
void ChElementHexaANCF_3843_TR08::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector3d& G_acc) {
    assert((unsigned int)Fg.size() == GetNumCoordsPosLevel());

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

void ChElementHexaANCF_3843_TR08::EvaluateElementFrame(const double xi,
                                                       const double eta,
                                                       const double zeta,
                                                       ChVector3d& point,
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
    ChVector3d Xdir = e_bar * Sxi_xi_compact * 2 / m_lenX;
    ChVector3d Ydir = e_bar * Sxi_eta_compact * 2 / m_lenY;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the brick xi axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the brick eta axis
    ChMatrix33d msect;
    msect.SetFromAxisX(Xdir, Ydir);

    rot = msect.GetQuaternion();
}

void ChElementHexaANCF_3843_TR08::EvaluateElementPoint(const double xi,
                                                       const double eta,
                                                       const double zeta,
                                                       ChVector3d& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementHexaANCF_3843_TR08::EvaluateElementVel(double xi, double eta, const double zeta, ChVector3d& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

    Matrix3xN e_bardot;
    CalcCoordDtMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).

void ChElementHexaANCF_3843_TR08::LoadableGetStateBlockPosLevel(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetSlope1().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetSlope2().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetSlope3().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetSlope1().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetSlope2().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetSlope3().eigen();

    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 27, 3) = m_nodes[2]->GetSlope1().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[2]->GetSlope2().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[2]->GetSlope3().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[3]->GetSlope1().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[3]->GetSlope2().eigen();
    mD.segment(block_offset + 45, 3) = m_nodes[3]->GetSlope3().eigen();

    mD.segment(block_offset + 48, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[4]->GetSlope1().eigen();
    mD.segment(block_offset + 54, 3) = m_nodes[4]->GetSlope2().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[4]->GetSlope3().eigen();

    mD.segment(block_offset + 60, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(block_offset + 63, 3) = m_nodes[5]->GetSlope1().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[5]->GetSlope2().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[5]->GetSlope3().eigen();

    mD.segment(block_offset + 72, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(block_offset + 75, 3) = m_nodes[6]->GetSlope1().eigen();
    mD.segment(block_offset + 78, 3) = m_nodes[6]->GetSlope2().eigen();
    mD.segment(block_offset + 81, 3) = m_nodes[6]->GetSlope3().eigen();

    mD.segment(block_offset + 84, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(block_offset + 87, 3) = m_nodes[7]->GetSlope1().eigen();
    mD.segment(block_offset + 90, 3) = m_nodes[7]->GetSlope2().eigen();
    mD.segment(block_offset + 93, 3) = m_nodes[7]->GetSlope3().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementHexaANCF_3843_TR08::LoadableGetStateBlockVelLevel(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPosDt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPosDt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetPosDt().eigen();
    mD.segment(block_offset + 27, 3) = m_nodes[2]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[2]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[2]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[3]->GetPosDt().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[3]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[3]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 45, 3) = m_nodes[3]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 48, 3) = m_nodes[4]->GetPosDt().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[4]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 54, 3) = m_nodes[4]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[4]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 60, 3) = m_nodes[5]->GetPosDt().eigen();
    mD.segment(block_offset + 63, 3) = m_nodes[5]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[5]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[5]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 72, 3) = m_nodes[6]->GetPosDt().eigen();
    mD.segment(block_offset + 75, 3) = m_nodes[6]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 78, 3) = m_nodes[6]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 81, 3) = m_nodes[6]->GetSlope3Dt().eigen();

    mD.segment(block_offset + 84, 3) = m_nodes[7]->GetPosDt().eigen();
    mD.segment(block_offset + 87, 3) = m_nodes[7]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 90, 3) = m_nodes[7]->GetSlope2Dt().eigen();
    mD.segment(block_offset + 93, 3) = m_nodes[7]->GetSlope3Dt().eigen();
}

/// Increment all DOFs using a delta.

void ChElementHexaANCF_3843_TR08::LoadableStateIncrement(const unsigned int off_x,
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

void ChElementHexaANCF_3843_TR08::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->VariablesSlope1());
        mvars.push_back(&m_nodes[i]->VariablesSlope2());
        mvars.push_back(&m_nodes[i]->VariablesSlope3());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementHexaANCF_3843_TR08::ComputeNF(
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

    ChMatrix33d J_Cxi;
    ChMatrix33d J_Cxi_Inv;

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

double ChElementHexaANCF_3843_TR08::GetDensity() {
    return GetMaterial()->GetDensity();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementHexaANCF_3843_TR08::ComputeMassMatrixAndGravityForce() {
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

    double rho = GetMaterial()->GetDensity();  // Density of the material for the element

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

    // Store just the unique entries in the Mass Matrix in Compact Upper Triangular Form
    // since the full Mass Matrix is both sparse and symmetric
    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            m_MassMatrix(idx) = MassMatrixCompactSquare(i, j);
            idx++;
        }
    }
}

// Precalculate constant matrices for the internal force calculations when using the "Continuous Integration" style
// method

void ChElementHexaANCF_3843_TR08::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();

    m_SD.resize(NSF, 3 * NIP);
    m_kGQ.resize(1, NIP);

    if (m_skipPrecomputation) {
        m_SD.setRandom();
        m_kGQ.setRandom();
    } else {
        // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
        // reference configuration & GQ Weights times the determinant of the element Jacobian for later use in the
        // generalized internal force and Jacobian calculations.
        for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
            for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[NP - 1].size(); it_eta++) {
                for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[NP - 1].size(); it_zeta++) {
                    double GQ_weight = GQTable->Weight[NP - 1][it_xi] * GQTable->Weight[NP - 1][it_eta] *
                                       GQTable->Weight[NP - 1][it_zeta];
                    double xi = GQTable->Lroots[NP - 1][it_xi];
                    double eta = GQTable->Lroots[NP - 1][it_eta];
                    double zeta = GQTable->Lroots[NP - 1][it_zeta];
                    auto index = it_zeta + it_eta * GQTable->Lroots[NP - 1].size() +
                                 it_xi * GQTable->Lroots[NP - 1].size() * GQTable->Lroots[NP - 1].size();
                    ChMatrix33d J_0xi;  // Element Jacobian between the reference and normalized configurations
                    MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

                    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                    J_0xi.noalias() = m_ebar0 * Sxi_D;

                    // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                    m_kGQ(index) = -J_0xi.determinant() * GQ_weight;
                    ChMatrixNM<double, NSF, 3> SD = Sxi_D * J_0xi.inverse();

                    // Group all of the columns together in blocks with the shape function derivatives for the section
                    // that does not include the Poisson effect at the beginning of m_SD
                    m_SD.col(index) = SD.col(0);
                    m_SD.col(index + NIP) = SD.col(1);
                    m_SD.col(index + 2 * NIP) = SD.col(2);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementHexaANCF_3843_TR08::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_TR08::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_TR08::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_TR08::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_TR08::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
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

void ChElementHexaANCF_3843_TR08::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetSlope1().eigen();
    e.segment(6, 3) = m_nodes[0]->GetSlope2().eigen();
    e.segment(9, 3) = m_nodes[0]->GetSlope3().eigen();

    e.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(15, 3) = m_nodes[1]->GetSlope1().eigen();
    e.segment(18, 3) = m_nodes[1]->GetSlope2().eigen();
    e.segment(21, 3) = m_nodes[1]->GetSlope3().eigen();

    e.segment(24, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(27, 3) = m_nodes[2]->GetSlope1().eigen();
    e.segment(30, 3) = m_nodes[2]->GetSlope2().eigen();
    e.segment(33, 3) = m_nodes[2]->GetSlope3().eigen();

    e.segment(36, 3) = m_nodes[3]->GetPos().eigen();
    e.segment(39, 3) = m_nodes[3]->GetSlope1().eigen();
    e.segment(42, 3) = m_nodes[3]->GetSlope2().eigen();
    e.segment(45, 3) = m_nodes[3]->GetSlope3().eigen();

    e.segment(48, 3) = m_nodes[4]->GetPos().eigen();
    e.segment(51, 3) = m_nodes[4]->GetSlope1().eigen();
    e.segment(54, 3) = m_nodes[4]->GetSlope2().eigen();
    e.segment(57, 3) = m_nodes[4]->GetSlope3().eigen();

    e.segment(60, 3) = m_nodes[5]->GetPos().eigen();
    e.segment(63, 3) = m_nodes[5]->GetSlope1().eigen();
    e.segment(66, 3) = m_nodes[5]->GetSlope2().eigen();
    e.segment(69, 3) = m_nodes[5]->GetSlope3().eigen();

    e.segment(72, 3) = m_nodes[6]->GetPos().eigen();
    e.segment(75, 3) = m_nodes[6]->GetSlope1().eigen();
    e.segment(78, 3) = m_nodes[6]->GetSlope2().eigen();
    e.segment(81, 3) = m_nodes[6]->GetSlope3().eigen();

    e.segment(84, 3) = m_nodes[7]->GetPos().eigen();
    e.segment(87, 3) = m_nodes[7]->GetSlope1().eigen();
    e.segment(90, 3) = m_nodes[7]->GetSlope2().eigen();
    e.segment(93, 3) = m_nodes[7]->GetSlope3().eigen();
}

void ChElementHexaANCF_3843_TR08::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetSlope1().eigen();
    ebar.col(2) = m_nodes[0]->GetSlope2().eigen();
    ebar.col(3) = m_nodes[0]->GetSlope3().eigen();

    ebar.col(4) = m_nodes[1]->GetPos().eigen();
    ebar.col(5) = m_nodes[1]->GetSlope1().eigen();
    ebar.col(6) = m_nodes[1]->GetSlope2().eigen();
    ebar.col(7) = m_nodes[1]->GetSlope3().eigen();

    ebar.col(8) = m_nodes[2]->GetPos().eigen();
    ebar.col(9) = m_nodes[2]->GetSlope1().eigen();
    ebar.col(10) = m_nodes[2]->GetSlope2().eigen();
    ebar.col(11) = m_nodes[2]->GetSlope3().eigen();

    ebar.col(12) = m_nodes[3]->GetPos().eigen();
    ebar.col(13) = m_nodes[3]->GetSlope1().eigen();
    ebar.col(14) = m_nodes[3]->GetSlope2().eigen();
    ebar.col(15) = m_nodes[3]->GetSlope3().eigen();

    ebar.col(16) = m_nodes[4]->GetPos().eigen();
    ebar.col(17) = m_nodes[4]->GetSlope1().eigen();
    ebar.col(18) = m_nodes[4]->GetSlope2().eigen();
    ebar.col(19) = m_nodes[4]->GetSlope3().eigen();

    ebar.col(20) = m_nodes[5]->GetPos().eigen();
    ebar.col(21) = m_nodes[5]->GetSlope1().eigen();
    ebar.col(22) = m_nodes[5]->GetSlope2().eigen();
    ebar.col(23) = m_nodes[5]->GetSlope3().eigen();

    ebar.col(24) = m_nodes[6]->GetPos().eigen();
    ebar.col(25) = m_nodes[6]->GetSlope1().eigen();
    ebar.col(26) = m_nodes[6]->GetSlope2().eigen();
    ebar.col(27) = m_nodes[6]->GetSlope3().eigen();

    ebar.col(28) = m_nodes[7]->GetPos().eigen();
    ebar.col(29) = m_nodes[7]->GetSlope1().eigen();
    ebar.col(30) = m_nodes[7]->GetSlope2().eigen();
    ebar.col(31) = m_nodes[7]->GetSlope3().eigen();
}

void ChElementHexaANCF_3843_TR08::CalcCoordDtVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPosDt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetSlope1Dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetSlope2Dt().eigen();
    edot.segment(9, 3) = m_nodes[0]->GetSlope3Dt().eigen();

    edot.segment(12, 3) = m_nodes[1]->GetPosDt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetSlope1Dt().eigen();
    edot.segment(18, 3) = m_nodes[1]->GetSlope2Dt().eigen();
    edot.segment(21, 3) = m_nodes[1]->GetSlope3Dt().eigen();

    edot.segment(24, 3) = m_nodes[2]->GetPosDt().eigen();
    edot.segment(27, 3) = m_nodes[2]->GetSlope1Dt().eigen();
    edot.segment(30, 3) = m_nodes[2]->GetSlope2Dt().eigen();
    edot.segment(33, 3) = m_nodes[2]->GetSlope3Dt().eigen();

    edot.segment(36, 3) = m_nodes[3]->GetPosDt().eigen();
    edot.segment(39, 3) = m_nodes[3]->GetSlope1Dt().eigen();
    edot.segment(42, 3) = m_nodes[3]->GetSlope2Dt().eigen();
    edot.segment(45, 3) = m_nodes[3]->GetSlope3Dt().eigen();

    edot.segment(48, 3) = m_nodes[4]->GetPosDt().eigen();
    edot.segment(51, 3) = m_nodes[4]->GetSlope1Dt().eigen();
    edot.segment(54, 3) = m_nodes[4]->GetSlope2Dt().eigen();
    edot.segment(57, 3) = m_nodes[4]->GetSlope3Dt().eigen();

    edot.segment(60, 3) = m_nodes[5]->GetPosDt().eigen();
    edot.segment(63, 3) = m_nodes[5]->GetSlope1Dt().eigen();
    edot.segment(66, 3) = m_nodes[5]->GetSlope2Dt().eigen();
    edot.segment(69, 3) = m_nodes[5]->GetSlope3Dt().eigen();

    edot.segment(72, 3) = m_nodes[6]->GetPosDt().eigen();
    edot.segment(75, 3) = m_nodes[6]->GetSlope1Dt().eigen();
    edot.segment(78, 3) = m_nodes[6]->GetSlope2Dt().eigen();
    edot.segment(81, 3) = m_nodes[6]->GetSlope3Dt().eigen();

    edot.segment(84, 3) = m_nodes[7]->GetPosDt().eigen();
    edot.segment(87, 3) = m_nodes[7]->GetSlope1Dt().eigen();
    edot.segment(90, 3) = m_nodes[7]->GetSlope2Dt().eigen();
    edot.segment(93, 3) = m_nodes[7]->GetSlope3Dt().eigen();
}

void ChElementHexaANCF_3843_TR08::CalcCoordDtMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPosDt().eigen();
    ebardot.col(1) = m_nodes[0]->GetSlope1Dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetSlope2Dt().eigen();
    ebardot.col(3) = m_nodes[0]->GetSlope3Dt().eigen();

    ebardot.col(4) = m_nodes[1]->GetPosDt().eigen();
    ebardot.col(5) = m_nodes[1]->GetSlope1Dt().eigen();
    ebardot.col(6) = m_nodes[1]->GetSlope2Dt().eigen();
    ebardot.col(7) = m_nodes[1]->GetSlope3Dt().eigen();

    ebardot.col(8) = m_nodes[2]->GetPosDt().eigen();
    ebardot.col(9) = m_nodes[2]->GetSlope1Dt().eigen();
    ebardot.col(10) = m_nodes[2]->GetSlope2Dt().eigen();
    ebardot.col(11) = m_nodes[2]->GetSlope3Dt().eigen();

    ebardot.col(12) = m_nodes[3]->GetPosDt().eigen();
    ebardot.col(13) = m_nodes[3]->GetSlope1Dt().eigen();
    ebardot.col(14) = m_nodes[3]->GetSlope2Dt().eigen();
    ebardot.col(15) = m_nodes[3]->GetSlope3Dt().eigen();

    ebardot.col(16) = m_nodes[4]->GetPosDt().eigen();
    ebardot.col(17) = m_nodes[4]->GetSlope1Dt().eigen();
    ebardot.col(18) = m_nodes[4]->GetSlope2Dt().eigen();
    ebardot.col(19) = m_nodes[4]->GetSlope3Dt().eigen();

    ebardot.col(20) = m_nodes[5]->GetPosDt().eigen();
    ebardot.col(21) = m_nodes[5]->GetSlope1Dt().eigen();
    ebardot.col(22) = m_nodes[5]->GetSlope2Dt().eigen();
    ebardot.col(23) = m_nodes[5]->GetSlope3Dt().eigen();

    ebardot.col(24) = m_nodes[6]->GetPosDt().eigen();
    ebardot.col(25) = m_nodes[6]->GetSlope1Dt().eigen();
    ebardot.col(26) = m_nodes[6]->GetSlope2Dt().eigen();
    ebardot.col(27) = m_nodes[6]->GetSlope3Dt().eigen();

    ebardot.col(28) = m_nodes[7]->GetPosDt().eigen();
    ebardot.col(29) = m_nodes[7]->GetSlope1Dt().eigen();
    ebardot.col(30) = m_nodes[7]->GetSlope2Dt().eigen();
    ebardot.col(31) = m_nodes[7]->GetSlope3Dt().eigen();
}

void ChElementHexaANCF_3843_TR08::CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot) {
    ebar_ebardot.block<3, 1>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 0) = m_nodes[0]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 1) = m_nodes[0]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 1) = m_nodes[0]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 2) = m_nodes[0]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 2) = m_nodes[0]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 3) = m_nodes[0]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 3) = m_nodes[0]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 4) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 4) = m_nodes[1]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 5) = m_nodes[1]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 5) = m_nodes[1]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 6) = m_nodes[1]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 6) = m_nodes[1]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 7) = m_nodes[1]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 7) = m_nodes[1]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 8) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 8) = m_nodes[2]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 9) = m_nodes[2]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 9) = m_nodes[2]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 10) = m_nodes[2]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 10) = m_nodes[2]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 11) = m_nodes[2]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 11) = m_nodes[2]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 12) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 12) = m_nodes[3]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 13) = m_nodes[3]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 13) = m_nodes[3]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 14) = m_nodes[3]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 14) = m_nodes[3]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 15) = m_nodes[3]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 15) = m_nodes[3]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 16) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 16) = m_nodes[4]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 17) = m_nodes[4]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 17) = m_nodes[4]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 18) = m_nodes[4]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 18) = m_nodes[4]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 19) = m_nodes[4]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 19) = m_nodes[4]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 20) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 20) = m_nodes[5]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 21) = m_nodes[5]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 21) = m_nodes[5]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 22) = m_nodes[5]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 22) = m_nodes[5]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 23) = m_nodes[5]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 23) = m_nodes[5]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 24) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 24) = m_nodes[6]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 25) = m_nodes[6]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 25) = m_nodes[6]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 26) = m_nodes[6]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 26) = m_nodes[6]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 27) = m_nodes[6]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 27) = m_nodes[6]->GetSlope3Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 28) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 28) = m_nodes[7]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 29) = m_nodes[7]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 29) = m_nodes[7]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 30) = m_nodes[7]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 30) = m_nodes[7]->GetSlope2Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 31) = m_nodes[7]->GetSlope3().eigen();
    ebar_ebardot.block<3, 1>(3, 31) = m_nodes[7]->GetSlope3Dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementHexaANCF_3843_TR08::Calc_J_0xi(ChMatrix33d& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementHexaANCF_3843_TR08::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33d J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3843_TR08(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementHexaANCF_3843_TR08::GetStaticGQTables() {
    return &static_tables_3843_TR08;
}

}  // namespace fea
}  // namespace chrono
