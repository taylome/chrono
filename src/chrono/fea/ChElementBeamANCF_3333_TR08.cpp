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
// Fully Parameterized ANCF beam element with 3 nodes (27DOF). A Description of this element and the Enhanced Continuum
// Mechanics based method can be found in: K. Nachbagauer, P. Gruber, and J. Gerstmayr. Structural and Continuum
// Mechanics Approaches for a 3D Shear Deformable ANCF Beam Finite Element : Application to Static and Linearized
// Dynamic Examples.J.Comput.Nonlinear Dynam, 8 (2) : 021004, 2012.
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

#include "chrono/fea/ChElementBeamANCF_3333_TR08.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBeamANCF_3333_TR08::ChElementBeamANCF_3333_TR08()
    : m_lenX(0), m_thicknessY(0), m_thicknessZ(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(3);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBeamANCF_3333_TR08::SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeB,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeC) {
    assert(nodeA);
    assert(nodeB);
    assert(nodeC);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;
    m_nodes[2] = nodeC;

    std::vector<ChVariables*> mvars;
    mvars.push_back(&m_nodes[0]->Variables());
    mvars.push_back(&m_nodes[0]->VariablesSlope1());
    mvars.push_back(&m_nodes[0]->VariablesSlope2());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->VariablesSlope1());
    mvars.push_back(&m_nodes[1]->VariablesSlope2());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->VariablesSlope1());
    mvars.push_back(&m_nodes[2]->VariablesSlope2());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
}

// -----------------------------------------------------------------------------
// Element Settings
// -----------------------------------------------------------------------------

// Specify the element dimensions (in the undeformed state - which is different than the reference configuration and it
// is a state the element potentially is never in).

void ChElementBeamANCF_3333_TR08::SetDimensions(double lenX, double thicknessY, double thicknessZ) {
    m_lenX = lenX;
    m_thicknessY = thicknessY;
    m_thicknessZ = thicknessZ;
}

// Specify the element material.

void ChElementBeamANCF_3333_TR08::SetMaterial(std::shared_ptr<ChMaterialBeamANCF> beam_mat) {
    m_material = beam_mat;
}

// Set the value for the single term structural damping coefficient.

void ChElementBeamANCF_3333_TR08::SetAlphaDamp(double a) {
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

ChMatrix33d ChElementBeamANCF_3333_TR08::GetGreenLagrangeStrain(const double xi, const double eta, const double zeta) {
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

ChMatrix33d ChElementBeamANCF_3333_TR08::GetPK2Stress(const double xi, const double eta, const double zeta) {
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

    ChMatrix66d D;
    GetMaterial()->Get_D(D);

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

double ChElementBeamANCF_3333_TR08::GetVonMissesStress(const double xi, const double eta, const double zeta) {
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

    ChMatrix66d D;
    GetMaterial()->Get_D(D);

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

void ChElementBeamANCF_3333_TR08::SetupInitial(ChSystem* system) {
    // Store the initial nodal coordinates. These values define the reference configuration of the element.
    CalcCoordMatrix(m_ebar0);

    // Compute and store the constant mass matrix and the matrix used to multiply the acceleration due to gravity to get
    // the generalized gravitational force vector for the element
    ComputeMassMatrixAndGravityForce();

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// Fill the D vector with the current field values at the element nodes.

void ChElementBeamANCF_3333_TR08::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetSlope1().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetSlope2().eigen();

    mD.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(12, 3) = m_nodes[1]->GetSlope1().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetSlope2().eigen();

    mD.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(21, 3) = m_nodes[2]->GetSlope1().eigen();
    mD.segment(24, 3) = m_nodes[2]->GetSlope2().eigen();
}

// State update.

void ChElementBeamANCF_3333_TR08::Update() {
    ChElementGeneric::Update();
}

// Return the mass matrix in full sparse form.

void ChElementBeamANCF_3333_TR08::ComputeMmatrixGlobal(ChMatrixRef M) {
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

void ChElementBeamANCF_3333_TR08::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(3) + m_MassMatrix(6);
    m_nodes[1]->m_TotalMass += m_MassMatrix(3) + m_MassMatrix(24) + m_MassMatrix(27);
    m_nodes[2]->m_TotalMass += m_MassMatrix(6) + m_MassMatrix(27) + m_MassMatrix(39);
}

// Compute the generalized internal force vector for the current nodal coordinates and set the value in the Fi vector.

void ChElementBeamANCF_3333_TR08::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert((unsigned int)Fi.size() == GetNumCoordsPosLevel());

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // =============================================================================
    // Calculate the deformation gradient and time derivative of the deformation gradient for all Gauss quadrature
    // points in a single matrix multiplication.  Note that since the shape function derivative matrix is ordered by
    // columns, the resulting deformation gradient will be ordered by block matrix (column vectors) components
    // Note that the indices of the components are in transposed order
    // D0 = No Poisson effect, Dv = Poisson effect along the beam axis
    //      [F11_D0     F12_D0     F13_D0     F11_Dv     F12_Dv     F13_Dv ]
    //      [F21_D0     F22_D0     F23_D0     F21_Dv     F22_Dv     F23_Dv ]
    // FC = [F31_D0     F32_D0     F33_D0     F31_Dv     F32_Dv     F33_Dv ]
    //      [Fdot11_D0  Fdot12_D0  Fdot13_D0  Fdot11_Dv  Fdot12_Dv  Fdot13_Dv ]
    //      [Fdot21_D0  Fdot22_D0  Fdot23_D0  Fdot21_Dv  Fdot22_Dv  Fdot23_Dv ]
    //      [Fdot31_D0  Fdot32_D0  Fdot33_D0  Fdot31_Dv  Fdot32_Dv  Fdot33_Dv ]
    // =============================================================================

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;  // ChMatrixNM<double, 6, 3 * NIP>

    Eigen::Map<ArrayNIP_D0> F11_D0(FC.block<1, NIP_D0>(0, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F12_D0(FC.block<1, NIP_D0>(0, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F13_D0(FC.block<1, NIP_D0>(0, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> F11_Dv(FC.block<1, NIP_Dv>(0, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F12_Dv(FC.block<1, NIP_Dv>(0, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F13_Dv(FC.block<1, NIP_Dv>(0, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> F21_D0(FC.block<1, NIP_D0>(1, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F22_D0(FC.block<1, NIP_D0>(1, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F23_D0(FC.block<1, NIP_D0>(1, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> F21_Dv(FC.block<1, NIP_Dv>(1, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F22_Dv(FC.block<1, NIP_Dv>(1, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F23_Dv(FC.block<1, NIP_Dv>(1, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> F31_D0(FC.block<1, NIP_D0>(2, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F32_D0(FC.block<1, NIP_D0>(2, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F33_D0(FC.block<1, NIP_D0>(2, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> F31_Dv(FC.block<1, NIP_Dv>(2, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F32_Dv(FC.block<1, NIP_Dv>(2, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F33_Dv(FC.block<1, NIP_Dv>(2, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> Fdot11_D0(FC.block<1, NIP_D0>(3, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot12_D0(FC.block<1, NIP_D0>(3, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot13_D0(FC.block<1, NIP_D0>(3, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> Fdot11_Dv(FC.block<1, NIP_Dv>(3, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot12_Dv(FC.block<1, NIP_Dv>(3, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot13_Dv(FC.block<1, NIP_Dv>(3, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> Fdot21_D0(FC.block<1, NIP_D0>(4, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot22_D0(FC.block<1, NIP_D0>(4, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot23_D0(FC.block<1, NIP_D0>(4, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> Fdot21_Dv(FC.block<1, NIP_Dv>(4, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot22_Dv(FC.block<1, NIP_Dv>(4, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot23_Dv(FC.block<1, NIP_Dv>(4, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> Fdot31_D0(FC.block<1, NIP_D0>(5, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot32_D0(FC.block<1, NIP_D0>(5, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot33_D0(FC.block<1, NIP_D0>(5, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> Fdot31_Dv(FC.block<1, NIP_Dv>(5, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot32_Dv(FC.block<1, NIP_Dv>(5, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot33_Dv(FC.block<1, NIP_Dv>(5, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    // =============================================================================
    // Calculate the 2nd Piola-Kirchhoff Stresses in Voigt notation across all of the D0 GQ Points at the same time
    // =============================================================================

    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();

    ArrayNIP_D0 SPK2_1_D0 = D0(0) * m_kGQ_D0 *
                            ((0.5 * (F11_D0 * F11_D0 + F21_D0 * F21_D0 + F31_D0 * F31_D0) - 0.5) +
                             m_Alpha * (F11_D0 * Fdot11_D0 + F21_D0 * Fdot21_D0 + F31_D0 * Fdot31_D0));
    ArrayNIP_D0 SPK2_2_D0 = D0(1) * m_kGQ_D0 *
                            ((0.5 * (F12_D0 * F12_D0 + F22_D0 * F22_D0 + F32_D0 * F32_D0) - 0.5) +
                             m_Alpha * (F12_D0 * Fdot12_D0 + F22_D0 * Fdot22_D0 + F32_D0 * Fdot32_D0));
    ArrayNIP_D0 SPK2_3_D0 = D0(2) * m_kGQ_D0 *
                            ((0.5 * (F13_D0 * F13_D0 + F23_D0 * F23_D0 + F33_D0 * F33_D0) - 0.5) +
                             m_Alpha * (F13_D0 * Fdot13_D0 + F23_D0 * Fdot23_D0 + F33_D0 * Fdot33_D0));
    ArrayNIP_D0 SPK2_4_D0 = D0(3) * m_kGQ_D0 *
                            ((F12_D0 * F13_D0 + F22_D0 * F23_D0 + F32_D0 * F33_D0) +
                             m_Alpha * (F12_D0 * Fdot13_D0 + F22_D0 * Fdot23_D0 + F32_D0 * Fdot33_D0 +
                                        Fdot12_D0 * F13_D0 + Fdot22_D0 * F23_D0 + Fdot32_D0 * F33_D0));
    ArrayNIP_D0 SPK2_5_D0 = D0(4) * m_kGQ_D0 *
                            ((F11_D0 * F13_D0 + F21_D0 * F23_D0 + F31_D0 * F33_D0) +
                             m_Alpha * (F11_D0 * Fdot13_D0 + F21_D0 * Fdot23_D0 + F31_D0 * Fdot33_D0 +
                                        Fdot11_D0 * F13_D0 + Fdot21_D0 * F23_D0 + Fdot31_D0 * F33_D0));
    ArrayNIP_D0 SPK2_6_D0 = D0(5) * m_kGQ_D0 *
                            ((F11_D0 * F12_D0 + F21_D0 * F22_D0 + F31_D0 * F32_D0) +
                             m_Alpha * (F11_D0 * Fdot12_D0 + F21_D0 * Fdot22_D0 + F31_D0 * Fdot32_D0 +
                                        Fdot11_D0 * F12_D0 + Fdot21_D0 * F22_D0 + Fdot31_D0 * F32_D0));

    const ChMatrix33d& Dv = GetMaterial()->Get_Dv();

    ArrayNIP_Dv epsilon_combined_1 =
        m_kGQ_Dv * ((0.5 * (F11_Dv * F11_Dv + F21_Dv * F21_Dv + F31_Dv * F31_Dv) - 0.5) +
                    m_Alpha * (F11_Dv * Fdot11_Dv + F21_Dv * Fdot21_Dv + F31_Dv * Fdot31_Dv));
    ArrayNIP_Dv epsilon_combined_2 =
        m_kGQ_Dv * ((0.5 * (F12_Dv * F12_Dv + F22_Dv * F22_Dv + F32_Dv * F32_Dv) - 0.5) +
                    m_Alpha * (F12_Dv * Fdot12_Dv + F22_Dv * Fdot22_Dv + F32_Dv * Fdot32_Dv));
    ArrayNIP_Dv epsilon_combined_3 =
        m_kGQ_Dv * ((0.5 * (F13_Dv * F13_Dv + F23_Dv * F23_Dv + F33_Dv * F33_Dv) - 0.5) +
                    m_Alpha * (F13_Dv * Fdot13_Dv + F23_Dv * Fdot23_Dv + F33_Dv * Fdot33_Dv));

    ArrayNIP_Dv Sdiag_1_Dv =
        Dv(0, 0) * epsilon_combined_1 + Dv(0, 1) * epsilon_combined_2 + Dv(0, 2) * epsilon_combined_3;
    ArrayNIP_Dv Sdiag_2_Dv =
        Dv(1, 0) * epsilon_combined_1 + Dv(1, 1) * epsilon_combined_2 + Dv(1, 2) * epsilon_combined_3;
    ArrayNIP_Dv Sdiag_3_Dv =
        Dv(2, 0) * epsilon_combined_1 + Dv(2, 1) * epsilon_combined_2 + Dv(2, 2) * epsilon_combined_3;

    // =============================================================================
    // Calculate the transpose of the 1st Piola-Kirchoff stresses in block tensor form whose entries have been
    // scaled by minus the Gauss quadrature weight times the element Jacobian at the corresponding Gauss point.
    // The entries are grouped by component in block matrices (column vectors)
    // P_Block = kGQ*P_transpose = kGQ*SPK2*F_transpose
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ] <-- D0 Block
    //           [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ] <-- D0 Block
    //         = [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ] <-- D0 Block
    //           [kGQ*(P_transpose)_11  kGQ*(P_transpose)_12  kGQ*(P_transpose)_13 ] <-- Dv Block
    //           [kGQ*(P_transpose)_21  kGQ*(P_transpose)_22  kGQ*(P_transpose)_23 ] <-- Dv Block
    //           [kGQ*(P_transpose)_31  kGQ*(P_transpose)_32  kGQ*(P_transpose)_33 ] <-- Dv Block
    // Note that the Dv Block entries will be calculated separately in a later step.
    // =============================================================================

    ChMatrixNM_col<double, 3 * NIP, 3> P_Block;

    P_Block.block<NIP_D0, 1>(0 * NIP_D0 + 0 * NIP_Dv, 0).array().transpose() =
        F11_D0 * SPK2_1_D0 + F12_D0 * SPK2_6_D0 + F13_D0 * SPK2_5_D0;
    P_Block.block<NIP_D0, 1>(1 * NIP_D0 + 0 * NIP_Dv, 0).array().transpose() =
        F11_D0 * SPK2_6_D0 + F12_D0 * SPK2_2_D0 + F13_D0 * SPK2_4_D0;
    P_Block.block<NIP_D0, 1>(2 * NIP_D0 + 0 * NIP_Dv, 0).array().transpose() =
        F11_D0 * SPK2_5_D0 + F12_D0 * SPK2_4_D0 + F13_D0 * SPK2_3_D0;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 0 * NIP_Dv, 0).array().transpose() = F11_Dv * Sdiag_1_Dv;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 1 * NIP_Dv, 0).array().transpose() = F12_Dv * Sdiag_2_Dv;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 2 * NIP_Dv, 0).array().transpose() = F13_Dv * Sdiag_3_Dv;

    P_Block.block<NIP_D0, 1>(0 * NIP_D0 + 0 * NIP_Dv, 1).array().transpose() =
        F21_D0 * SPK2_1_D0 + F22_D0 * SPK2_6_D0 + F23_D0 * SPK2_5_D0;
    P_Block.block<NIP_D0, 1>(1 * NIP_D0 + 0 * NIP_Dv, 1).array().transpose() =
        F21_D0 * SPK2_6_D0 + F22_D0 * SPK2_2_D0 + F23_D0 * SPK2_4_D0;
    P_Block.block<NIP_D0, 1>(2 * NIP_D0 + 0 * NIP_Dv, 1).array().transpose() =
        F21_D0 * SPK2_5_D0 + F22_D0 * SPK2_4_D0 + F23_D0 * SPK2_3_D0;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 0 * NIP_Dv, 1).array().transpose() = F21_Dv * Sdiag_1_Dv;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 1 * NIP_Dv, 1).array().transpose() = F22_Dv * Sdiag_2_Dv;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 2 * NIP_Dv, 1).array().transpose() = F23_Dv * Sdiag_3_Dv;

    P_Block.block<NIP_D0, 1>(0 * NIP_D0 + 0 * NIP_Dv, 2).array().transpose() =
        F31_D0 * SPK2_1_D0 + F32_D0 * SPK2_6_D0 + F33_D0 * SPK2_5_D0;
    P_Block.block<NIP_D0, 1>(1 * NIP_D0 + 0 * NIP_Dv, 2).array().transpose() =
        F31_D0 * SPK2_6_D0 + F32_D0 * SPK2_2_D0 + F33_D0 * SPK2_4_D0;
    P_Block.block<NIP_D0, 1>(2 * NIP_D0 + 0 * NIP_Dv, 2).array().transpose() =
        F31_D0 * SPK2_5_D0 + F32_D0 * SPK2_4_D0 + F33_D0 * SPK2_3_D0;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 0 * NIP_Dv, 2).array().transpose() = F31_Dv * Sdiag_1_Dv;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 1 * NIP_Dv, 2).array().transpose() = F32_Dv * Sdiag_2_Dv;
    P_Block.block<NIP_Dv, 1>(3 * NIP_D0 + 2 * NIP_Dv, 2).array().transpose() = F33_Dv * Sdiag_3_Dv;

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

void ChElementBeamANCF_3333_TR08::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                           double Kfactor,
                                                           double Rfactor,
                                                           double Mfactor) {
    assert((H.rows() == 3 * NSF) && (H.cols() == 3 * NSF));

    // Calculate the Jacobian of the generalize internal force vector using the "Continuous Integration" style of method
    // assuming a linear viscoelastic material model (single term damping model).  For this style of method, the
    // Jacobian of the generalized internal force vector is integrated across the volume of the element every time this
    // calculation is performed. For this element, this is likely more efficient than the "Pre-Integration" style
    // calculation method.

    // Retrieve the nodal coordinates and nodal coordinate time derivatives
    Matrix6xN ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    ChMatrixNM<double, 6, 3 * NIP> FC = ebar_ebardot * m_SD;  // ChMatrixNM<double, 6, 3 * NIP>

    Eigen::Map<ArrayNIP_D0> F11_D0(FC.block<1, NIP_D0>(0, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F12_D0(FC.block<1, NIP_D0>(0, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F13_D0(FC.block<1, NIP_D0>(0, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> F11_Dv(FC.block<1, NIP_Dv>(0, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F12_Dv(FC.block<1, NIP_Dv>(0, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F13_Dv(FC.block<1, NIP_Dv>(0, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> F21_D0(FC.block<1, NIP_D0>(1, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F22_D0(FC.block<1, NIP_D0>(1, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F23_D0(FC.block<1, NIP_D0>(1, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> F21_Dv(FC.block<1, NIP_Dv>(1, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F22_Dv(FC.block<1, NIP_Dv>(1, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F23_Dv(FC.block<1, NIP_Dv>(1, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> F31_D0(FC.block<1, NIP_D0>(2, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F32_D0(FC.block<1, NIP_D0>(2, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> F33_D0(FC.block<1, NIP_D0>(2, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> F31_Dv(FC.block<1, NIP_Dv>(2, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F32_Dv(FC.block<1, NIP_Dv>(2, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> F33_Dv(FC.block<1, NIP_Dv>(2, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> Fdot11_D0(FC.block<1, NIP_D0>(3, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot12_D0(FC.block<1, NIP_D0>(3, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot13_D0(FC.block<1, NIP_D0>(3, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> Fdot11_Dv(FC.block<1, NIP_Dv>(3, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot12_Dv(FC.block<1, NIP_Dv>(3, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot13_Dv(FC.block<1, NIP_Dv>(3, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> Fdot21_D0(FC.block<1, NIP_D0>(4, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot22_D0(FC.block<1, NIP_D0>(4, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot23_D0(FC.block<1, NIP_D0>(4, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> Fdot21_Dv(FC.block<1, NIP_Dv>(4, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot22_Dv(FC.block<1, NIP_Dv>(4, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot23_Dv(FC.block<1, NIP_Dv>(4, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> Fdot31_D0(FC.block<1, NIP_D0>(5, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot32_D0(FC.block<1, NIP_D0>(5, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> Fdot33_D0(FC.block<1, NIP_D0>(5, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> Fdot31_Dv(FC.block<1, NIP_Dv>(5, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot32_Dv(FC.block<1, NIP_Dv>(5, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> Fdot33_Dv(FC.block<1, NIP_Dv>(5, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    ChMatrixNM<double, 3 * NSF, 6 * NIP_D0 + 3 * NIP_Dv> PE;
    for (auto i = 0; i < NSF; i++) {
        PE.block<1, NIP_D0>(3 * i + 0, 0 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F11_D0;
        PE.block<1, NIP_D0>(3 * i + 0, 1 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F12_D0;
        PE.block<1, NIP_D0>(3 * i + 0, 2 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F13_D0;
        PE.block<1, NIP_D0>(3 * i + 0, 3 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F13_D0 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F12_D0;
        PE.block<1, NIP_D0>(3 * i + 0, 4 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F13_D0 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F11_D0;
        PE.block<1, NIP_D0>(3 * i + 0, 5 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F12_D0 +
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F11_D0;
        PE.block<1, NIP_Dv>(3 * i + 0, 6 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * F11_Dv;
        PE.block<1, NIP_Dv>(3 * i + 0, 6 * NIP_D0 + 1 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * F12_Dv;
        PE.block<1, NIP_Dv>(3 * i + 0, 6 * NIP_D0 + 2 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * F13_Dv;

        PE.block<1, NIP_D0>(3 * i + 1, 0 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F21_D0;
        PE.block<1, NIP_D0>(3 * i + 1, 1 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F22_D0;
        PE.block<1, NIP_D0>(3 * i + 1, 2 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F23_D0;
        PE.block<1, NIP_D0>(3 * i + 1, 3 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F23_D0 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F22_D0;
        PE.block<1, NIP_D0>(3 * i + 1, 4 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F23_D0 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F21_D0;
        PE.block<1, NIP_D0>(3 * i + 1, 5 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F22_D0 +
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F21_D0;
        PE.block<1, NIP_Dv>(3 * i + 1, 6 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * F21_Dv;
        PE.block<1, NIP_Dv>(3 * i + 1, 6 * NIP_D0 + 1 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * F22_Dv;
        PE.block<1, NIP_Dv>(3 * i + 1, 6 * NIP_D0 + 2 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * F23_Dv;

        PE.block<1, NIP_D0>(3 * i + 2, 0 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F31_D0;
        PE.block<1, NIP_D0>(3 * i + 2, 1 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F32_D0;
        PE.block<1, NIP_D0>(3 * i + 2, 2 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F33_D0;
        PE.block<1, NIP_D0>(3 * i + 2, 3 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F33_D0 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F32_D0;
        PE.block<1, NIP_D0>(3 * i + 2, 4 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F33_D0 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * F31_D0;
        PE.block<1, NIP_D0>(3 * i + 2, 5 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * F32_D0 +
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * F31_D0;
        PE.block<1, NIP_Dv>(3 * i + 2, 6 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * F31_Dv;
        PE.block<1, NIP_Dv>(3 * i + 2, 6 * NIP_D0 + 1 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * F32_Dv;
        PE.block<1, NIP_Dv>(3 * i + 2, 6 * NIP_D0 + 2 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * F33_Dv;
    }

    ChMatrixNM<double, 3, 3 * NIP> FCS =
        (-Kfactor - m_Alpha * Rfactor) * FC.block<3, 3 * NIP>(0, 0) - (m_Alpha * Kfactor) * FC.block<3, 3 * NIP>(3, 0);
    for (auto i = 0; i < 3; i++) {
        FCS.block<1, NIP_D0>(i, 0 * NIP_D0 + 0 * NIP_Dv).array() *= m_kGQ_D0;
        FCS.block<1, NIP_D0>(i, 1 * NIP_D0 + 0 * NIP_Dv).array() *= m_kGQ_D0;
        FCS.block<1, NIP_D0>(i, 2 * NIP_D0 + 0 * NIP_Dv).array() *= m_kGQ_D0;
        FCS.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() *= m_kGQ_Dv;
        FCS.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() *= m_kGQ_Dv;
        FCS.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() *= m_kGQ_Dv;
    }

    Eigen::Map<ArrayNIP_D0> FS11_D0(FCS.block<1, NIP_D0>(0, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> FS12_D0(FCS.block<1, NIP_D0>(0, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> FS13_D0(FCS.block<1, NIP_D0>(0, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> FS11_Dv(FCS.block<1, NIP_Dv>(0, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> FS12_Dv(FCS.block<1, NIP_Dv>(0, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> FS13_Dv(FCS.block<1, NIP_Dv>(0, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> FS21_D0(FCS.block<1, NIP_D0>(1, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> FS22_D0(FCS.block<1, NIP_D0>(1, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> FS23_D0(FCS.block<1, NIP_D0>(1, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> FS21_Dv(FCS.block<1, NIP_Dv>(1, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> FS22_Dv(FCS.block<1, NIP_Dv>(1, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> FS23_Dv(FCS.block<1, NIP_Dv>(1, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    Eigen::Map<ArrayNIP_D0> FS31_D0(FCS.block<1, NIP_D0>(2, 0 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> FS32_D0(FCS.block<1, NIP_D0>(2, 1 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_D0> FS33_D0(FCS.block<1, NIP_D0>(2, 2 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_D0);
    Eigen::Map<ArrayNIP_Dv> FS31_Dv(FCS.block<1, NIP_Dv>(2, 3 * NIP_D0 + 0 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> FS32_Dv(FCS.block<1, NIP_Dv>(2, 3 * NIP_D0 + 1 * NIP_Dv).data(), 1, NIP_Dv);
    Eigen::Map<ArrayNIP_Dv> FS33_Dv(FCS.block<1, NIP_Dv>(2, 3 * NIP_D0 + 2 * NIP_Dv).data(), 1, NIP_Dv);

    // =============================================================================
    // Get the diagonal terms of the 6x6 matrix (do not include the Poisson effect)
    // and the 3x3 upper block that includes the Poisson effect
    // =============================================================================

    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();
    const ChMatrix33d& Dv = GetMaterial()->Get_Dv();

    ArrayNIP_D0 FS11_D0_D1 = D0(0) * FS11_D0;
    ArrayNIP_D0 FS11_D0_D5 = D0(4) * FS11_D0;
    ArrayNIP_D0 FS11_D0_D6 = D0(5) * FS11_D0;
    ArrayNIP_D0 FS12_D0_D2 = D0(1) * FS12_D0;
    ArrayNIP_D0 FS12_D0_D4 = D0(3) * FS12_D0;
    ArrayNIP_D0 FS12_D0_D6 = D0(5) * FS12_D0;
    ArrayNIP_D0 FS13_D0_D3 = D0(2) * FS13_D0;
    ArrayNIP_D0 FS13_D0_D4 = D0(3) * FS13_D0;
    ArrayNIP_D0 FS13_D0_D5 = D0(4) * FS13_D0;
    ArrayNIP_D0 FS21_D0_D1 = D0(0) * FS21_D0;
    ArrayNIP_D0 FS21_D0_D5 = D0(4) * FS21_D0;
    ArrayNIP_D0 FS21_D0_D6 = D0(5) * FS21_D0;
    ArrayNIP_D0 FS22_D0_D2 = D0(1) * FS22_D0;
    ArrayNIP_D0 FS22_D0_D4 = D0(3) * FS22_D0;
    ArrayNIP_D0 FS22_D0_D6 = D0(5) * FS22_D0;
    ArrayNIP_D0 FS23_D0_D3 = D0(2) * FS23_D0;
    ArrayNIP_D0 FS23_D0_D4 = D0(3) * FS23_D0;
    ArrayNIP_D0 FS23_D0_D5 = D0(4) * FS23_D0;
    ArrayNIP_D0 FS31_D0_D1 = D0(0) * FS31_D0;
    ArrayNIP_D0 FS31_D0_D5 = D0(4) * FS31_D0;
    ArrayNIP_D0 FS31_D0_D6 = D0(5) * FS31_D0;
    ArrayNIP_D0 FS32_D0_D2 = D0(1) * FS32_D0;
    ArrayNIP_D0 FS32_D0_D4 = D0(3) * FS32_D0;
    ArrayNIP_D0 FS32_D0_D6 = D0(5) * FS32_D0;
    ArrayNIP_D0 FS33_D0_D3 = D0(2) * FS33_D0;
    ArrayNIP_D0 FS33_D0_D5 = D0(4) * FS33_D0;
    ArrayNIP_D0 FS33_D0_D4 = D0(3) * FS33_D0;

    ArrayNIP_Dv Dv11_FS11_Dv = Dv(0, 0) * FS11_Dv;
    ArrayNIP_Dv Dv21_FS11_Dv = Dv(1, 0) * FS11_Dv;
    ArrayNIP_Dv Dv31_FS11_Dv = Dv(2, 0) * FS11_Dv;
    ArrayNIP_Dv Dv12_FS12_Dv = Dv(0, 1) * FS12_Dv;
    ArrayNIP_Dv Dv22_FS12_Dv = Dv(1, 1) * FS12_Dv;
    ArrayNIP_Dv Dv32_FS12_Dv = Dv(2, 1) * FS12_Dv;
    ArrayNIP_Dv Dv13_FS13_Dv = Dv(0, 2) * FS13_Dv;
    ArrayNIP_Dv Dv23_FS13_Dv = Dv(1, 2) * FS13_Dv;
    ArrayNIP_Dv Dv33_FS13_Dv = Dv(2, 2) * FS13_Dv;

    ArrayNIP_Dv Dv11_FS21_Dv = Dv(0, 0) * FS21_Dv;
    ArrayNIP_Dv Dv21_FS21_Dv = Dv(1, 0) * FS21_Dv;
    ArrayNIP_Dv Dv31_FS21_Dv = Dv(2, 0) * FS21_Dv;
    ArrayNIP_Dv Dv12_FS22_Dv = Dv(0, 1) * FS22_Dv;
    ArrayNIP_Dv Dv22_FS22_Dv = Dv(1, 1) * FS22_Dv;
    ArrayNIP_Dv Dv32_FS22_Dv = Dv(2, 1) * FS22_Dv;
    ArrayNIP_Dv Dv13_FS23_Dv = Dv(0, 2) * FS23_Dv;
    ArrayNIP_Dv Dv23_FS23_Dv = Dv(1, 2) * FS23_Dv;
    ArrayNIP_Dv Dv33_FS23_Dv = Dv(2, 2) * FS23_Dv;

    ArrayNIP_Dv Dv11_FS31_Dv = Dv(0, 0) * FS31_Dv;
    ArrayNIP_Dv Dv21_FS31_Dv = Dv(1, 0) * FS31_Dv;
    ArrayNIP_Dv Dv31_FS31_Dv = Dv(2, 0) * FS31_Dv;
    ArrayNIP_Dv Dv12_FS32_Dv = Dv(0, 1) * FS32_Dv;
    ArrayNIP_Dv Dv22_FS32_Dv = Dv(1, 1) * FS32_Dv;
    ArrayNIP_Dv Dv32_FS32_Dv = Dv(2, 1) * FS32_Dv;
    ArrayNIP_Dv Dv13_FS33_Dv = Dv(0, 2) * FS33_Dv;
    ArrayNIP_Dv Dv23_FS33_Dv = Dv(1, 2) * FS33_Dv;
    ArrayNIP_Dv Dv33_FS33_Dv = Dv(2, 2) * FS33_Dv;

    ChMatrixNM<double, 3 * NSF, 6 * NIP_D0 + 3 * NIP_Dv> DPE;
    for (auto i = 0; i < NSF; i++) {
        DPE.block<1, NIP_D0>(3 * i + 0, 0 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS11_D0_D1;
        DPE.block<1, NIP_D0>(3 * i + 0, 1 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS12_D0_D2;
        DPE.block<1, NIP_D0>(3 * i + 0, 2 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS13_D0_D3;
        DPE.block<1, NIP_D0>(3 * i + 0, 3 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS13_D0_D4 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS12_D0_D4;
        DPE.block<1, NIP_D0>(3 * i + 0, 4 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS13_D0_D5 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS11_D0_D5;
        DPE.block<1, NIP_D0>(3 * i + 0, 5 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS12_D0_D6 +
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS11_D0_D6;
        DPE.block<1, NIP_Dv>(3 * i + 0, 6 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv11_FS11_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv12_FS12_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv13_FS13_Dv;
        DPE.block<1, NIP_Dv>(3 * i + 0, 6 * NIP_D0 + 1 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv21_FS11_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv22_FS12_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv23_FS13_Dv;
        DPE.block<1, NIP_Dv>(3 * i + 0, 6 * NIP_D0 + 2 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv31_FS11_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv32_FS12_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv33_FS13_Dv;

        DPE.block<1, NIP_D0>(3 * i + 1, 0 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS21_D0_D1;
        DPE.block<1, NIP_D0>(3 * i + 1, 1 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS22_D0_D2;
        DPE.block<1, NIP_D0>(3 * i + 1, 2 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS23_D0_D3;
        DPE.block<1, NIP_D0>(3 * i + 1, 3 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS23_D0_D4 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS22_D0_D4;
        DPE.block<1, NIP_D0>(3 * i + 1, 4 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS23_D0_D5 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS21_D0_D5;
        DPE.block<1, NIP_D0>(3 * i + 1, 5 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS22_D0_D6 +
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS21_D0_D6;
        DPE.block<1, NIP_Dv>(3 * i + 1, 6 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv11_FS21_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv12_FS22_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv13_FS23_Dv;
        DPE.block<1, NIP_Dv>(3 * i + 1, 6 * NIP_D0 + 1 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv21_FS21_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv22_FS22_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv23_FS23_Dv;
        DPE.block<1, NIP_Dv>(3 * i + 1, 6 * NIP_D0 + 2 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv31_FS21_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv32_FS22_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv33_FS23_Dv;

        DPE.block<1, NIP_D0>(3 * i + 2, 0 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS31_D0_D1;
        DPE.block<1, NIP_D0>(3 * i + 2, 1 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS32_D0_D2;
        DPE.block<1, NIP_D0>(3 * i + 2, 2 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS33_D0_D3;
        DPE.block<1, NIP_D0>(3 * i + 2, 3 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS33_D0_D4 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS32_D0_D4;
        DPE.block<1, NIP_D0>(3 * i + 2, 4 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS33_D0_D5 +
            m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array() * FS31_D0_D5;
        DPE.block<1, NIP_D0>(3 * i + 2, 5 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() * FS32_D0_D6 +
            m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() * FS31_D0_D6;
        DPE.block<1, NIP_Dv>(3 * i + 2, 6 * NIP_D0 + 0 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv11_FS31_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv12_FS32_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv13_FS33_Dv;
        DPE.block<1, NIP_Dv>(3 * i + 2, 6 * NIP_D0 + 1 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv21_FS31_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv22_FS32_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv23_FS33_Dv;
        DPE.block<1, NIP_Dv>(3 * i + 2, 6 * NIP_D0 + 2 * NIP_Dv).array() =
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array() * Dv31_FS31_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array() * Dv32_FS32_Dv +
            m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array() * Dv33_FS33_Dv;
    }

    Matrix3Nx3N Jac = PE * DPE.transpose();

    //==============================================================================
    //==============================================================================
    // Calculate the sparse and symmetric component of the Jacobian matrix
    //==============================================================================
    //==============================================================================
    // Calculate just the non-sparse upper triangular entires of the sparse and symmetric component of the Jacobian
    // matrix, combine this with the scaled mass matrix, and then expand them out to full size by summing the
    // contribution into the correct locations of the full sized Jacobian matrix  while adding in the Mass Matrix Which
    // is Stored in Compact Upper Triangular Form
    // =============================================================================

    ArrayNIP_D0 SPK2_1_D0 = D0(0) * m_kGQ_D0 *
                            ((0.5 * (F11_D0 * F11_D0 + F21_D0 * F21_D0 + F31_D0 * F31_D0) - 0.5) +
                             m_Alpha * (F11_D0 * Fdot11_D0 + F21_D0 * Fdot21_D0 + F31_D0 * Fdot31_D0));
    ArrayNIP_D0 SPK2_2_D0 = D0(1) * m_kGQ_D0 *
                            ((0.5 * (F12_D0 * F12_D0 + F22_D0 * F22_D0 + F32_D0 * F32_D0) - 0.5) +
                             m_Alpha * (F12_D0 * Fdot12_D0 + F22_D0 * Fdot22_D0 + F32_D0 * Fdot32_D0));
    ArrayNIP_D0 SPK2_3_D0 = D0(2) * m_kGQ_D0 *
                            ((0.5 * (F13_D0 * F13_D0 + F23_D0 * F23_D0 + F33_D0 * F33_D0) - 0.5) +
                             m_Alpha * (F13_D0 * Fdot13_D0 + F23_D0 * Fdot23_D0 + F33_D0 * Fdot33_D0));
    ArrayNIP_D0 SPK2_4_D0 = D0(3) * m_kGQ_D0 *
                            ((F12_D0 * F13_D0 + F22_D0 * F23_D0 + F32_D0 * F33_D0) +
                             m_Alpha * (F12_D0 * Fdot13_D0 + F22_D0 * Fdot23_D0 + F32_D0 * Fdot33_D0 +
                                        Fdot12_D0 * F13_D0 + Fdot22_D0 * F23_D0 + Fdot32_D0 * F33_D0));
    ArrayNIP_D0 SPK2_5_D0 = D0(4) * m_kGQ_D0 *
                            ((F11_D0 * F13_D0 + F21_D0 * F23_D0 + F31_D0 * F33_D0) +
                             m_Alpha * (F11_D0 * Fdot13_D0 + F21_D0 * Fdot23_D0 + F31_D0 * Fdot33_D0 +
                                        Fdot11_D0 * F13_D0 + Fdot21_D0 * F23_D0 + Fdot31_D0 * F33_D0));
    ArrayNIP_D0 SPK2_6_D0 = D0(5) * m_kGQ_D0 *
                            ((F11_D0 * F12_D0 + F21_D0 * F22_D0 + F31_D0 * F32_D0) +
                             m_Alpha * (F11_D0 * Fdot12_D0 + F21_D0 * Fdot22_D0 + F31_D0 * Fdot32_D0 +
                                        Fdot11_D0 * F12_D0 + Fdot21_D0 * F22_D0 + Fdot31_D0 * F32_D0));

    ArrayNIP_Dv epsilon_combined_1 =
        m_kGQ_Dv * ((0.5 * (F11_Dv * F11_Dv + F21_Dv * F21_Dv + F31_Dv * F31_Dv) - 0.5) +
                    m_Alpha * (F11_Dv * Fdot11_Dv + F21_Dv * Fdot21_Dv + F31_Dv * Fdot31_Dv));
    ArrayNIP_Dv epsilon_combined_2 =
        m_kGQ_Dv * ((0.5 * (F12_Dv * F12_Dv + F22_Dv * F22_Dv + F32_Dv * F32_Dv) - 0.5) +
                    m_Alpha * (F12_Dv * Fdot12_Dv + F22_Dv * Fdot22_Dv + F32_Dv * Fdot32_Dv));
    ArrayNIP_Dv epsilon_combined_3 =
        m_kGQ_Dv * ((0.5 * (F13_Dv * F13_Dv + F23_Dv * F23_Dv + F33_Dv * F33_Dv) - 0.5) +
                    m_Alpha * (F13_Dv * Fdot13_Dv + F23_Dv * Fdot23_Dv + F33_Dv * Fdot33_Dv));

    ArrayNIP_Dv Sdiag_1_Dv =
        Dv(0, 0) * epsilon_combined_1 + Dv(0, 1) * epsilon_combined_2 + Dv(0, 2) * epsilon_combined_3;
    ArrayNIP_Dv Sdiag_2_Dv =
        Dv(1, 0) * epsilon_combined_1 + Dv(1, 1) * epsilon_combined_2 + Dv(1, 2) * epsilon_combined_3;
    ArrayNIP_Dv Sdiag_3_Dv =
        Dv(2, 0) * epsilon_combined_1 + Dv(2, 1) * epsilon_combined_2 + Dv(2, 2) * epsilon_combined_3;

    ChMatrixNM<double, NSF, 3 * NIP> S_scaled_SD;
    for (auto i = 0; i < NSF; i++) {
        S_scaled_SD.block<1, NIP_D0>(i, 0) = SPK2_1_D0 * m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() +
                                             SPK2_6_D0 * m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() +
                                             SPK2_5_D0 * m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array();

        S_scaled_SD.block<1, NIP_D0>(i, NIP_D0) = SPK2_6_D0 * m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() +
                                                  SPK2_2_D0 * m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() +
                                                  SPK2_4_D0 * m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array();

        S_scaled_SD.block<1, NIP_D0>(i, 2 * NIP_D0) = SPK2_5_D0 * m_SD.block<1, NIP_D0>(i, 0 * NIP_D0).array() +
                                                      SPK2_4_D0 * m_SD.block<1, NIP_D0>(i, 1 * NIP_D0).array() +
                                                      SPK2_3_D0 * m_SD.block<1, NIP_D0>(i, 2 * NIP_D0).array();

        S_scaled_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0) =
            Sdiag_1_Dv * m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 0 * NIP_Dv).array();

        S_scaled_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + NIP_Dv) =
            Sdiag_2_Dv * m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 1 * NIP_Dv).array();

        S_scaled_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv) =
            Sdiag_3_Dv * m_SD.block<1, NIP_Dv>(i, 3 * NIP_D0 + 2 * NIP_Dv).array();
    }

    unsigned int idx = 0;
    for (unsigned int i = 0; i < NSF; i++) {
        for (unsigned int j = i; j < NSF; j++) {
            // double d = ScaledMassMatrix(idx) - Kfactor * m_SD.row(i) * S_scaled_SD.row(j).transpose();
            double d = Mfactor * m_MassMatrix(idx) - Kfactor * m_SD.row(i) * S_scaled_SD.row(j).transpose();

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
void ChElementBeamANCF_3333_TR08::ComputeGravityForces(ChVectorDynamic<>& Fg, const ChVector3d& G_acc) {
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
// Interface to ChElementBeam base class (and similar methods)
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3333_TR08::EvaluateSectionFrame(const double xi, ChVector3d& point, ChQuaternion<>& rot) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, 0, 0);
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector3d BeamAxisTangent = e_bar * Sxi_xi_compact * 2 / m_lenX;
    ChVector3d CrossSectionY = e_bar * Sxi_eta_compact * 2 / m_thicknessY;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33d msect;
    msect.SetFromAxisX(BeamAxisTangent, CrossSectionY);

    rot = msect.GetQuaternion();
}

void ChElementBeamANCF_3333_TR08::EvaluateSectionPoint(const double xi, ChVector3d& point) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // r = S*e written in compact form
    point = e_bar * Sxi_compact;
}

void ChElementBeamANCF_3333_TR08::EvaluateSectionVel(const double xi, ChVector3d& Result) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);

    Matrix3xN e_bardot;
    CalcCoordDtMatrix(e_bardot);

    // rdot = S*edot written in compact form
    Result = e_bardot * Sxi_compact;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).

void ChElementBeamANCF_3333_TR08::LoadableGetStateBlockPosLevel(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetSlope1().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetSlope2().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetSlope1().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetSlope2().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetSlope1().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetSlope2().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).

void ChElementBeamANCF_3333_TR08::LoadableGetStateBlockVelLevel(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPosDt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetSlope2Dt().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPosDt().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetSlope2Dt().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPosDt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetSlope1Dt().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetSlope2Dt().eigen();
}

/// Increment all DOFs using a delta.

void ChElementBeamANCF_3333_TR08::LoadableStateIncrement(const unsigned int off_x,
                                                         ChState& x_new,
                                                         const ChState& x,
                                                         const unsigned int off_v,
                                                         const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 9, x_new, x, off_v + 9, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 18, x_new, x, off_v + 18, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.

void ChElementBeamANCF_3333_TR08::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->VariablesSlope1());
        mvars.push_back(&m_nodes[i]->VariablesSlope2());
    }
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the beam axis coordinates (xi,0,0)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementBeamANCF_3333_TR08::ComputeNF(
    const double xi,             // parametric coordinate along the beam axis
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force component using the compact form of the shape
    // functions.  This requires a reshaping of the calculated matrix to get it into the correct vector order (just a
    // reinterpretation of the data since the matrix is in row-major format)
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, 0, 0);
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
    Calc_Sxi_D(Sxi_D, xi, 0, 0);

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
    //  actual differential length and the normalized differential length.  The vector 2 norm is used to calculate this
    //  length ratio for potential use in Gauss-Quadrature or similar numeric integration.
    detJ = J_Cxi.col(0).norm();
}

// Evaluate N'*F, which is the projection of the applied point force and moment at the coordinates (xi,eta,zeta)
// This calculation takes a slightly different form for ANCF elements
// For this ANCF element, only the first 6 entries in F are used in the calculation.  The first three entries is
// the applied force in global coordinates and the second 3 entries is the applied moment in global space.

void ChElementBeamANCF_3333_TR08::ComputeNF(
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

double ChElementBeamANCF_3333_TR08::GetDensity() {
    return GetMaterial()->GetDensity();
}

// Calculate tangent to the centerline at coordinate xi [-1 to 1].

ChVector3d ChElementBeamANCF_3333_TR08::ComputeTangent(const double xi) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, 0, 0);

    Matrix3xN e_bar;
    CalcCoordMatrix(e_bar);

    // partial derivative of the position vector with respect to xi (normalized coordinate along the beam axis).  In
    // general, this will not be a unit vector
    ChVector3d BeamAxisTangent = e_bar * Sxi_xi_compact;

    return BeamAxisTangent.GetNormalized();
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3333_TR08::ComputeMassMatrixAndGravityForce() {
    // For this element, the mass matrix integrand is of order 9 in xi, 3 in eta, and 3 in zeta.
    // 4 GQ Points are needed in the xi direction and 2 GQ Points are needed in the eta and zeta directions for
    // exact integration of the element's mass matrix, even if the reference configuration is not straight. Since the
    // major pieces of the generalized force due to gravity can also be used to calculate the mass matrix, these
    // calculations are performed at the same time.  Only the matrix that scales the acceleration due to gravity is
    // calculated at this time so that any changes to the acceleration due to gravity in the system are correctly
    // accounted for in the generalized internal force calculation.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 4;        // 5 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1;  // 2 Point Gauss-Quadrature;

    // Mass Matrix in its compact matrix form.  Since the mass matrix is symmetric, just the upper diagonal entries will
    // be stored.
    MatrixNxN MassMatrixCompactSquare;

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MassMatrixCompactSquare.setZero();
    m_GravForceScale.setZero();

    double rho = GetMaterial()->GetDensity();  // Density of the material for the element

    // Sum the contribution to the mass matrix and generalized force due to gravity at the current point
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
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

void ChElementBeamANCF_3333_TR08::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();

    m_SD.resize(NSF, 3 * NIP);
    m_kGQ_D0.resize(1, NIP_D0);
    m_kGQ_Dv.resize(1, NIP_Dv);

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinant of the element Jacobian for later use in the
    // generalized internal force and Jacobian calculations.

    // First calculate the matrices and constants for the portion of the Enhance Continuum Mechanics/Selective Reduced
    // Integration that does not include the Poisson effect.
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[NT - 1].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[NT - 1].size(); it_zeta++) {
                double GQ_weight =
                    GQTable->Weight[NP - 1][it_xi] * GQTable->Weight[NT - 1][it_eta] * GQTable->Weight[NT - 1][it_zeta];
                double xi = GQTable->Lroots[NP - 1][it_xi];
                double eta = GQTable->Lroots[NT - 1][it_eta];
                double zeta = GQTable->Lroots[NT - 1][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[NT - 1].size() +
                             it_xi * GQTable->Lroots[NT - 1].size() * GQTable->Lroots[NT - 1].size();
                ChMatrix33d J_0xi;  // Element Jacobian between the reference and normalized configurations
                MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                // Adjust the shape function derivative matrix to account for a potentially deformed reference state
                m_kGQ_D0(index) = -J_0xi.determinant() * GQ_weight;
                ChMatrixNM<double, NSF, 3> SD = Sxi_D * J_0xi.inverse();

                // Group all of the columns together in blocks with the shape function derivatives for the section that
                // does not include the Poisson effect at the beginning of m_SD
                m_SD.col(index) = SD.col(0);
                m_SD.col(index + NIP_D0) = SD.col(1);
                m_SD.col(index + 2 * NIP_D0) = SD.col(2);

                index++;
            }
        }
    }

    // Next calculate the matrices and constants for the portion of the Enhance Continuum Mechanics/Selective Reduced
    // Integration that includes the Poisson effect, but integrating across the volume of the element with one 1 point
    // GQ for the cross section directions.
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[NP - 1].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[NP - 1][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[NP - 1][it_xi];
        double eta = 0;
        double zeta = 0;
        ChMatrix33d J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
        MatrixNx3c Sxi_D;          // Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, eta, zeta);
        J_0xi.noalias() = m_ebar0 * Sxi_D;

        // Adjust the shape function derivative matrix to account for a potentially deformed reference state
        m_kGQ_Dv(it_xi) = -J_0xi.determinant() * GQ_weight;
        ChMatrixNM<double, NSF, 3> SD = Sxi_D * J_0xi.inverse();

        // Group all of the columns together in blocks with the shape function derivative for the Poisson effect at the
        // end of m_SD
        m_SD.col(3 * NIP_D0 + it_xi) = SD.col(0);
        m_SD.col(3 * NIP_D0 + NIP_Dv + it_xi) = SD.col(1);
        m_SD.col(3 * NIP_D0 + 2 * NIP_Dv + it_xi) = SD.col(2);
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// Nx1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]

void ChElementBeamANCF_3333_TR08::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) = 0.5 * (xi * xi - xi);
    Sxi_compact(1) = 0.25 * m_thicknessY * eta * (xi * xi - xi);
    Sxi_compact(2) = 0.25 * m_thicknessZ * zeta * (xi * xi - xi);
    Sxi_compact(3) = 0.5 * (xi * xi + xi);
    Sxi_compact(4) = 0.25 * m_thicknessY * eta * (xi * xi + xi);
    Sxi_compact(5) = 0.25 * m_thicknessZ * zeta * (xi * xi + xi);
    Sxi_compact(6) = 1.0 - xi * xi;
    Sxi_compact(7) = 0.5 * m_thicknessY * eta * (1.0 - xi * xi);
    Sxi_compact(8) = 0.5 * m_thicknessZ * zeta * (1.0 - xi * xi);
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]

void ChElementBeamANCF_3333_TR08::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = xi - 0.5;
    Sxi_xi_compact(1) = 0.25 * m_thicknessY * eta * (2.0 * xi - 1.0);
    Sxi_xi_compact(2) = 0.25 * m_thicknessZ * zeta * (2.0 * xi - 1.0);
    Sxi_xi_compact(3) = xi + 0.5;
    Sxi_xi_compact(4) = 0.25 * m_thicknessY * eta * (2.0 * xi + 1.0);
    Sxi_xi_compact(5) = 0.25 * m_thicknessZ * zeta * (2.0 * xi + 1.0);
    Sxi_xi_compact(6) = -2.0 * xi;
    Sxi_xi_compact(7) = -m_thicknessY * eta * xi;
    Sxi_xi_compact(8) = -m_thicknessZ * zeta * xi;
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]

void ChElementBeamANCF_3333_TR08::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = 0.0;
    Sxi_eta_compact(1) = 0.25 * m_thicknessY * (xi * xi - xi);
    Sxi_eta_compact(2) = 0.0;
    Sxi_eta_compact(3) = 0.0;
    Sxi_eta_compact(4) = 0.25 * m_thicknessY * (xi * xi + xi);
    Sxi_eta_compact(5) = 0.0;
    Sxi_eta_compact(6) = 0.0;
    Sxi_eta_compact(7) = 0.5 * m_thicknessY * (1 - xi * xi);
    Sxi_eta_compact(8) = 0.0;
}

// Nx1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]

void ChElementBeamANCF_3333_TR08::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
    Sxi_zeta_compact(0) = 0.0;
    Sxi_zeta_compact(1) = 0.0;
    Sxi_zeta_compact(2) = 0.25 * m_thicknessZ * (xi * xi - xi);
    Sxi_zeta_compact(3) = 0.0;
    Sxi_zeta_compact(4) = 0.0;
    Sxi_zeta_compact(5) = 0.25 * m_thicknessZ * (xi * xi + xi);
    Sxi_zeta_compact(6) = 0.0;
    Sxi_zeta_compact(7) = 0.0;
    Sxi_zeta_compact(8) = 0.5 * m_thicknessZ * (1 - xi * xi);
}

// Nx3 compact form of the partial derivatives of Normalized Shape Functions with respect to xi, eta, and zeta by
// columns

void ChElementBeamANCF_3333_TR08::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
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

void ChElementBeamANCF_3333_TR08::CalcCoordVector(Vector3N& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetSlope1().eigen();
    e.segment(6, 3) = m_nodes[0]->GetSlope2().eigen();

    e.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(12, 3) = m_nodes[1]->GetSlope1().eigen();
    e.segment(15, 3) = m_nodes[1]->GetSlope2().eigen();

    e.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(21, 3) = m_nodes[2]->GetSlope1().eigen();
    e.segment(24, 3) = m_nodes[2]->GetSlope2().eigen();
}

void ChElementBeamANCF_3333_TR08::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetSlope1().eigen();
    ebar.col(2) = m_nodes[0]->GetSlope2().eigen();

    ebar.col(3) = m_nodes[1]->GetPos().eigen();
    ebar.col(4) = m_nodes[1]->GetSlope1().eigen();
    ebar.col(5) = m_nodes[1]->GetSlope2().eigen();

    ebar.col(6) = m_nodes[2]->GetPos().eigen();
    ebar.col(7) = m_nodes[2]->GetSlope1().eigen();
    ebar.col(8) = m_nodes[2]->GetSlope2().eigen();
}

void ChElementBeamANCF_3333_TR08::CalcCoordDtVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPosDt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetSlope1Dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetSlope2Dt().eigen();

    edot.segment(9, 3) = m_nodes[1]->GetPosDt().eigen();
    edot.segment(12, 3) = m_nodes[1]->GetSlope1Dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetSlope2Dt().eigen();

    edot.segment(18, 3) = m_nodes[2]->GetPosDt().eigen();
    edot.segment(21, 3) = m_nodes[2]->GetSlope1Dt().eigen();
    edot.segment(24, 3) = m_nodes[2]->GetSlope2Dt().eigen();
}

void ChElementBeamANCF_3333_TR08::CalcCoordDtMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPosDt().eigen();
    ebardot.col(1) = m_nodes[0]->GetSlope1Dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetSlope2Dt().eigen();

    ebardot.col(3) = m_nodes[1]->GetPosDt().eigen();
    ebardot.col(4) = m_nodes[1]->GetSlope1Dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetSlope2Dt().eigen();

    ebardot.col(6) = m_nodes[2]->GetPosDt().eigen();
    ebardot.col(7) = m_nodes[2]->GetSlope1Dt().eigen();
    ebardot.col(8) = m_nodes[2]->GetSlope2Dt().eigen();
}

void ChElementBeamANCF_3333_TR08::CalcCombinedCoordMatrix(Matrix6xN& ebar_ebardot) {
    ebar_ebardot.block<3, 1>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 0) = m_nodes[0]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 1) = m_nodes[0]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 1) = m_nodes[0]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 2) = m_nodes[0]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 2) = m_nodes[0]->GetSlope2Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 3) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 3) = m_nodes[1]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 4) = m_nodes[1]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 4) = m_nodes[1]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 5) = m_nodes[1]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 5) = m_nodes[1]->GetSlope2Dt().eigen();

    ebar_ebardot.block<3, 1>(0, 6) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<3, 1>(3, 6) = m_nodes[2]->GetPosDt().eigen();
    ebar_ebardot.block<3, 1>(0, 7) = m_nodes[2]->GetSlope1().eigen();
    ebar_ebardot.block<3, 1>(3, 7) = m_nodes[2]->GetSlope1Dt().eigen();
    ebar_ebardot.block<3, 1>(0, 8) = m_nodes[2]->GetSlope2().eigen();
    ebar_ebardot.block<3, 1>(3, 8) = m_nodes[2]->GetSlope2Dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

void ChElementBeamANCF_3333_TR08::Calc_J_0xi(ChMatrix33d& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinant of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element

double ChElementBeamANCF_3333_TR08::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33d J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3333_TR08(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBeamANCF_3333_TR08::GetStaticGQTables() {
    return &static_tables_3333_TR08;
}

}  // namespace fea
}  // namespace chrono
