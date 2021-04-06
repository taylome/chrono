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
// Fully Parameterized ANCF brick element with 8 nodes. Description of this element
// and its internal forces may be found in: Olshevskiy, A., Dmitrochenko, O., & 
// Kim, C. W. (2014). Three-dimensional solid brick element using slopes in the 
// absolute nodal coordinate formulation. Journal of Computational and Nonlinear 
// Dynamics, 9(2).
// =============================================================================
// Internal Force Calculation Method is based on:  Gerstmayr, J., Shabana, A.A.:
// Efficient integration of the elastic forces and thin three-dimensional beam
// elements in the absolute nodal coordinate formulation.In: Proceedings of the
// Multibody Dynamics Eccomas thematic Conference, Madrid(2005)
// =============================================================================
// TR08S = a Gerstmayr style implementation of the element with pre-calculation
//     of the terms needed for the generalized internal force calculation with
//     an analytical Jacobian that is integrated across all GQ points at once
//
//  Mass Matrix = Constant, pre-calculated 32x32 matrix
//
//  Generalized Force due to gravity = Constant 96x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Gerstmayr method:
//     Dense Math: e_bar = 3x32 and S_bar = 32x1
//     Math is based on the method presented by Gerstmayr and Shabana
//     1 less than "Full Integration" Number of GQ Integration Points (6x6x2)
//     GQ integration is performed across all the GQ points at once
//     Pre-calculation of terms for the generalized internal force calculation
//
//  Jacobian of the Generalized Internal Force Vector = Analytical Jacobian that
//     is integrated across all GQ points at once
//     F and Strains are not cached from the internal force calculation but are
//     recalculated during the Jacobian calculations
//
// =============================================================================

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR08S.h"
#include <cmath>

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBrickANCF_3843_TR08S::ChElementBrickANCF_3843_TR08S()
    : m_gravity_on(false), m_lenX(0), m_lenY(0), m_lenZ(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(8);

    m_GQWeight_det_J_0xi_D.resize(216,1);
    m_SD_precompute_col_ordered.resize(32, 648);
    m_MassMatrix.resize(528, 1);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR08S::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
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
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementBrickANCF_3843_TR08S::SetupInitial(ChSystem* system) {
    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementBrickANCF_3843_TR08S::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementBrickANCF_3843_TR08S::GetStateBlock(ChVectorDynamic<>& mD) {
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

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
void ChElementBrickANCF_3843_TR08S::ComputeKRMmatricesGlobal(ChMatrixRef H,
    double Kfactor,
    double Rfactor,
    double Mfactor) {
#if true  // Analytical Jacobian
    if (m_damping_enabled) {  // If linear Kelvin-Voigt viscoelastic material model is enabled
        ComputeInternalJacobianDamping(H, -Kfactor, -Rfactor, Mfactor);
    }
    else {
        ComputeInternalJacobianNoDamping(H, -Kfactor, Mfactor);
    }
#else  // Numeric Jacobian
    Matrix3Nx3N JacobianMatrix;
    assert((H.rows() == JacobianMatrix.rows()) && (H.cols() == JacobianMatrix.cols()));

    // Calculate the linear combination Kfactor*[K] + Rfactor*[R]
    ComputeInternalJacobians(JacobianMatrix, Kfactor, Rfactor);

    // Load Jac + Mfactor*[M] into H
    H = JacobianMatrix;
    //Mass Matrix is Stored in Compact Upper Triangular Form
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = i; j < 32; j++) {
            H(3 * i, 3 * j) += Mfactor * m_MassMatrix(idx);
            H(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(idx);
            H(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(idx);
            if (i != j) {
                H(3 * j, 3 * i) += Mfactor * m_MassMatrix(idx);
                H(3 * j + 1, 3 * i + 1) += Mfactor * m_MassMatrix(idx);
                H(3 * j + 2, 3 * i + 2) += Mfactor * m_MassMatrix(idx);
            }
            idx++;
        }
    }
#endif
}

// Return the mass matrix.
void ChElementBrickANCF_3843_TR08S::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Inflate the Mass Matrix since it is stored in compact form.
    // In MATLAB notation:
    // M(1:3:end,1:3:end) = m_MassMatrix;
    // M(2:3:end,2:3:end) = m_MassMatrix;
    // M(3:3:end,3:3:end) = m_MassMatrix;

    //Mass Matrix is Stored in Compact Upper Triangular Form
    //Expand it out into its Full Sparse Symmetric Form
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = i; j < 32; j++) {
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

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------
void ChElementBrickANCF_3843_TR08S::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, 4 GQ Points are needed in the xi, eta, and zeta directions
    //  for exact integration of the element's mass matrix, even if
    //  the reference configuration is not straight
    // Mass Matrix Integrand is of order: 7 in xi, order: 7 in eta, and order: 7 in zeta.
    // Since the major pieces of the generalized force due to gravity
    //  can also be used to calculate the mass matrix, these calculations
    //  are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = 3;  // 4 Point Gauss-Quadrature;

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    ChMatrixNM<double, 32, 32> MassMatrixCompactSquare;
    MassMatrixCompactSquare.setZero();
    m_GravForce.setZero();

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta_zeta][it_xi] *
                    GQTable->Weight[GQ_idx_xi_eta_zeta][it_eta] *
                    GQTable->Weight[GQ_idx_xi_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta);  // determinate of the element Jacobian (volume ratio)
                Matrix3x3N Sxi;                                    // 3x72 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                VectorN Sxi_compact;  // 32x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight * rho * det_J_0xi) * Sxi.transpose() * g_acc.eigen();
                MassMatrixCompactSquare += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }

    //Store just the unique entries in the Mass Matrix in Compact Upper Triangular Form
    //since the full Mass Matrix is both sparse and symmetric
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = i; j < 32; j++) {
            m_MassMatrix(idx) = MassMatrixCompactSquare(i, j);
            idx++;
        }
    }
}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementBrickANCF_3843_TR08S::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4) + m_MassMatrix(8) + m_MassMatrix(12) + m_MassMatrix(16) + m_MassMatrix(20) + m_MassMatrix(24) + m_MassMatrix(28);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(122) + m_MassMatrix(126) + m_MassMatrix(130) + m_MassMatrix(134) + m_MassMatrix(138) + m_MassMatrix(142) + m_MassMatrix(146);
    m_nodes[2]->m_TotalMass += m_MassMatrix(8) + m_MassMatrix(126) + m_MassMatrix(228) + m_MassMatrix(232) + m_MassMatrix(236) + m_MassMatrix(240) + m_MassMatrix(244) + m_MassMatrix(248);
    m_nodes[3]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(130) + m_MassMatrix(232) + m_MassMatrix(318) + m_MassMatrix(322) + m_MassMatrix(326) + m_MassMatrix(330) + m_MassMatrix(334);
    m_nodes[4]->m_TotalMass += m_MassMatrix(16) + m_MassMatrix(134) + m_MassMatrix(236) + m_MassMatrix(322) + m_MassMatrix(392) + m_MassMatrix(396) + m_MassMatrix(400) + m_MassMatrix(404);
    m_nodes[5]->m_TotalMass += m_MassMatrix(20) + m_MassMatrix(138) + m_MassMatrix(240) + m_MassMatrix(326) + m_MassMatrix(396) + m_MassMatrix(450) + m_MassMatrix(454) + m_MassMatrix(458);
    m_nodes[6]->m_TotalMass += m_MassMatrix(24) + m_MassMatrix(142) + m_MassMatrix(244) + m_MassMatrix(330) + m_MassMatrix(400) + m_MassMatrix(454) + m_MassMatrix(492) + m_MassMatrix(496);
    m_nodes[7]->m_TotalMass += m_MassMatrix(28) + m_MassMatrix(146) + m_MassMatrix(248) + m_MassMatrix(334) + m_MassMatrix(404) + m_MassMatrix(458) + m_MassMatrix(496) + m_MassMatrix(518);
}

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementBrickANCF_3843_TR08S::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = 5;    // 6 Point Gauss-Quadrature;

    ChMatrixNM<double, 32, 3> SD_precompute_D;

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determiate of the element Jacobian for later Calculating the
    // portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta_zeta][it_xi] * GQTable->Weight[GQ_idx_xi_eta_zeta][it_eta] *
                    GQTable->Weight[GQ_idx_xi_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_xi_eta_zeta].size() +
                    it_xi * GQTable->Lroots[GQ_idx_xi_eta_zeta].size() * GQTable->Lroots[GQ_idx_xi_eta_zeta].size();
                ChMatrix33<double>
                    J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
                MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                SD_precompute_D = Sxi_D * J_0xi.inverse();

                m_SD_precompute_col_ordered.col(index) = SD_precompute_D.col(0);
                m_SD_precompute_col_ordered.col(index + 216) = SD_precompute_D.col(1);
                m_SD_precompute_col_ordered.col(index + 432) = SD_precompute_D.col(2);

                m_GQWeight_det_J_0xi_D(index) = -J_0xi.determinant() * GQ_weight;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementBrickANCF_3843_TR08S::SetAlphaDamp(double a) {
    m_Alpha = a;
    m_2Alpha = 2 * a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementBrickANCF_3843_TR08S::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == m_GravForce.size());
    // Runs faster if the internal force with or without damping calculations are not combined into the same function
    // using the common calculations with an if statement for the damping in the middle to calculate the different
    // P_transpose_scaled_Block components
    if (m_damping_enabled) {
        MatrixNx6 ebar_ebardot;
        CalcCombinedCoordMatrix(ebar_ebardot);

        ComputeInternalForcesAtState(Fi, ebar_ebardot);
        //Fi.setZero();
    }
    else {
        MatrixNx3 e_bar;
        CalcCoordMatrix(e_bar);

        ComputeInternalForcesAtStateNoDamping(Fi, e_bar);
    }

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementBrickANCF_3843_TR08S::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
    const MatrixNx6& ebar_ebardot) {
    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 8 in eta, and order : 8 in zeta.
    // This requires GQ 5 points along the xi direction and 5 points along the eta and zeta directions for "Full
    // Integration" However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in
    // roughly 1/3 of the calculations

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> F_Transpose_CombinedBlockDamping_col_ordered = m_SD_precompute_col_ordered.transpose() * ebar_ebardot;
    ChMatrixNMc<double, 648, 6> F_Transpose_CombinedBlockDamping_col_ordered = m_SD_precompute_col_ordered.transpose() * ebar_ebardot;

    VectorNIP E0_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0));
    E0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1));
    E0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2));
    E0_Block.array() -= 1;
    E0_Block *= 0.5;
    VectorNIP E_BlockDamping = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 5));
    E0_Block += m_Alpha * E_BlockDamping;
    E0_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E1_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0));
    E1_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1));
    E1_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2));
    E1_Block.array() -= 1;
    E1_Block *= 0.5;
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 5));
    E1_Block += m_Alpha * E_BlockDamping;
    E1_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E2_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0));
    E2_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1));
    E2_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2));
    E2_Block.array() -= 1;
    E2_Block *= 0.5;
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 5));
    E2_Block += m_Alpha * E_BlockDamping;
    E2_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E3_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0));
    E3_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1));
    E3_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2));
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 5));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 5));
    E3_Block += m_Alpha * E_BlockDamping;
    E3_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E4_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0));
    E4_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1));
    E4_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2));
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 5));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 5));
    E4_Block += m_Alpha * E_BlockDamping;
    E4_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E5_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0));
    E5_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1));
    E5_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2));
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 5));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 5));
    E5_Block += m_Alpha * E_BlockDamping;
    E5_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();
    VectorNIP SPK2_0_Block = D(0, 0)*E0_Block + D(0, 1)*E1_Block + D(0, 2)*E2_Block + D(0, 3)*E3_Block + D(0, 4)*E4_Block + D(0, 5)*E5_Block;
    VectorNIP SPK2_1_Block = D(1, 0)*E0_Block + D(1, 1)*E1_Block + D(1, 2)*E2_Block + D(1, 3)*E3_Block + D(1, 4)*E4_Block + D(1, 5)*E5_Block;
    VectorNIP SPK2_2_Block = D(2, 0)*E0_Block + D(2, 1)*E1_Block + D(2, 2)*E2_Block + D(2, 3)*E3_Block + D(2, 4)*E4_Block + D(2, 5)*E5_Block;
    VectorNIP SPK2_3_Block = D(3, 0)*E0_Block + D(3, 1)*E1_Block + D(3, 2)*E2_Block + D(3, 3)*E3_Block + D(3, 4)*E4_Block + D(3, 5)*E5_Block;
    VectorNIP SPK2_4_Block = D(4, 0)*E0_Block + D(4, 1)*E1_Block + D(4, 2)*E2_Block + D(4, 3)*E3_Block + D(4, 4)*E4_Block + D(4, 5)*E5_Block;
    VectorNIP SPK2_5_Block = D(5, 0)*E0_Block + D(5, 1)*E1_Block + D(5, 2)*E2_Block + D(5, 3)*E3_Block + D(5, 4)*E4_Block + D(5, 5)*E5_Block;

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> P_transpose_scaled_Block_col_ordered;
    //P_transpose_scaled_Block_col_ordered.resize(648, 3);
    ChMatrixNMc<double, 648, 3>
        P_transpose_scaled_Block_col_ordered;  // 1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled
                                               // across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(SPK2_0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 0) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(SPK2_0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 1) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(SPK2_0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 2) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(SPK2_4_Block);

    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(SPK2_5_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 0) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(SPK2_5_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 1) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(SPK2_5_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 2) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(SPK2_3_Block);

    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(SPK2_4_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 0) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(SPK2_4_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 1) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(SPK2_4_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 2) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(SPK2_2_Block);

    // =============================================================================

    MatrixNx3 QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementBrickANCF_3843_TR08S::ComputeInternalForcesAtStateNoDamping(ChVectorDynamic<>& Fi,
    const MatrixNx3& e_bar) {
    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full
    // Integration" However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in
    // roughly 1/3 of the calculations

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;
    ChMatrixNMc<double, 648, 3> F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;

    VectorNIP E0_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0));
    E0_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1));
    E0_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2));
    E0_Block.array() -= 1;
    E0_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E0_Block *= 0.5;

    VectorNIP E1_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0));
    E1_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1));
    E1_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2));
    E1_Block.array() -= 1;
    E1_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E1_Block *= 0.5;

    VectorNIP E2_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0));
    E2_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1));
    E2_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2));
    E2_Block.array() -= 1;
    E2_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E2_Block *= 0.5;

    VectorNIP E3_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0));
    E3_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1));
    E3_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2));
    E3_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E4_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0));
    E4_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1));
    E4_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2));
    E4_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E5_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0));
    E5_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1));
    E5_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2));
    E5_Block.array() *= m_GQWeight_det_J_0xi_D.array();



    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();
    VectorNIP SPK2_0_Block = D(0, 0)*E0_Block + D(0, 1)*E1_Block + D(0, 2)*E2_Block + D(0, 3)*E3_Block + D(0, 4)*E4_Block + D(0, 5)*E5_Block;
    VectorNIP SPK2_1_Block = D(1, 0)*E0_Block + D(1, 1)*E1_Block + D(1, 2)*E2_Block + D(1, 3)*E3_Block + D(1, 4)*E4_Block + D(1, 5)*E5_Block;
    VectorNIP SPK2_2_Block = D(2, 0)*E0_Block + D(2, 1)*E1_Block + D(2, 2)*E2_Block + D(2, 3)*E3_Block + D(2, 4)*E4_Block + D(2, 5)*E5_Block;
    VectorNIP SPK2_3_Block = D(3, 0)*E0_Block + D(3, 1)*E1_Block + D(3, 2)*E2_Block + D(3, 3)*E3_Block + D(3, 4)*E4_Block + D(3, 5)*E5_Block;
    VectorNIP SPK2_4_Block = D(4, 0)*E0_Block + D(4, 1)*E1_Block + D(4, 2)*E2_Block + D(4, 3)*E3_Block + D(4, 4)*E4_Block + D(4, 5)*E5_Block;
    VectorNIP SPK2_5_Block = D(5, 0)*E0_Block + D(5, 1)*E1_Block + D(5, 2)*E2_Block + D(5, 3)*E3_Block + D(5, 4)*E4_Block + D(5, 5)*E5_Block;

    ChMatrixNMc<double, 648, 3>
        P_transpose_scaled_Block_col_ordered;  // 1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled
                                               // across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(SPK2_0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 0) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).cwiseProduct(SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(SPK2_0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 1) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).cwiseProduct(SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(SPK2_0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(0, 2) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).cwiseProduct(SPK2_4_Block);

    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(SPK2_5_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 0) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(SPK2_5_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 1) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(SPK2_5_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(216, 2) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).cwiseProduct(SPK2_3_Block);

    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(SPK2_4_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 0) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).cwiseProduct(SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(SPK2_4_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 1) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).cwiseProduct(SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(SPK2_4_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<216, 1>(432, 2) +=
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).cwiseProduct(SPK2_2_Block);

    // =============================================================================

    MatrixNx3 QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR08S::ComputeInternalJacobians(Matrix3Nx3N& JacobianMatrix,
    double Kfactor,
    double Rfactor) {
    // The integrated quantity represents the 96x96 Jacobian
    //      Kfactor * [K] + Rfactor * [R]

    ChVectorDynamic<double> FiOrignal(96);
    ChVectorDynamic<double> FiDelta(96);

    double delta = 1e-6;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    // Runs faster if the internal force with or without damping calculations are not combined into the same function
    // using the common calculations with an if statement for the damping in the middle to calculate the different
    // P_transpose_scaled_Block components
    if (m_damping_enabled) {
        MatrixNx6 ebar_ebardot;
        CalcCombinedCoordMatrix(ebar_ebardot);

        ComputeInternalForcesAtState(FiOrignal, ebar_ebardot);
        for (unsigned int i = 0; i < 96; i++) {
            ebar_ebardot(i / 3, i % 3) = ebar_ebardot(i / 3, i % 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar_ebardot);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            ebar_ebardot(i / 3, i % 3) = ebar_ebardot(i / 3, i % 3) - delta;

            ebar_ebardot(i / 3, (i % 3) + 3) = ebar_ebardot(i / 3, (i % 3) + 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar_ebardot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            ebar_ebardot(i / 3, (i % 3) + 3) = ebar_ebardot(i / 3, (i % 3) + 3) - delta;
        }
    }
    else {
        MatrixNx3 e_bar;
        CalcCoordMatrix(e_bar);

        ComputeInternalForcesAtStateNoDamping(FiOrignal, e_bar);
        for (unsigned int i = 0; i < 96; i++) {
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) + delta;
            ComputeInternalForcesAtStateNoDamping(FiDelta, e_bar);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) - delta;
        }
    }
}


void ChElementBrickANCF_3843_TR08S::ComputeInternalJacobianDamping(ChMatrixRef& H,
    double Kfactor,
    double Rfactor,
    double Mfactor) {

    MatrixNx6 ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> F_Transpose_CombinedBlockDamping_col_ordered = m_SD_precompute_col_ordered.transpose() * ebar_ebardot;
    ChMatrixNMc<double, 648, 6> F_Transpose_CombinedBlockDamping_col_ordered = m_SD_precompute_col_ordered.transpose() * ebar_ebardot;

    // ChMatrixNM<double, 48, 180> partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> partial_epsilon_partial_e_Transpose;
    partial_epsilon_partial_e_Transpose.resize(96, 1296);

    for (auto i = 0; i < 32; i++) {
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).transpose());
    }

    //ChMatrixNMc<double, 648, 3> scaled_F_Transpose_col_ordered =
    //    (Kfactor + m_Alpha * Rfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 0) +
    //    (m_Alpha * Kfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 3);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> scaled_F_Transpose_col_ordered;
    //scaled_F_Transpose_col_ordered.resize(648, 3);
    //scaled_F_Transpose_col_ordered = (Kfactor + m_Alpha * Rfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 0) +
    //    (m_Alpha * Kfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 3);
    ChMatrixNMc<double, 648, 3> scaled_F_Transpose_col_ordered = (Kfactor + m_Alpha * Rfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 0) +
        (m_Alpha * Kfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 3);

    for (auto i = 0; i < 3; i++) {
        scaled_F_Transpose_col_ordered.block<216, 1>(0, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<216, 1>(216, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<216, 1>(432, i).array() *= m_GQWeight_det_J_0xi_D.array();
    }

    // ChMatrixNM<double, 48, 432> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.resize(96, 1296);


    for (auto i = 0; i < 32; i++) {
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 0).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 1).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 2).transpose());
    }

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    // ChMatrixNM<double, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    DScaled_Combined_partial_epsilon_partial_e_Transpose.resize(96, 1296);

    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) = D(0, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(0, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(0, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(0, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(0, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(0, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) = D(1, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(1, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(1, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(1, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(1, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(1, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) = D(2, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(2, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(2, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(2, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(2, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(2, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) = D(3, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(3, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(3, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(3, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(3, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(3, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) = D(4, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(4, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(4, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(4, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(4, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(4, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080) = D(5, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(5, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(5, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(5, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(5, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(5, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);

    H = partial_epsilon_partial_e_Transpose * DScaled_Combined_partial_epsilon_partial_e_Transpose.transpose();

    //===========================================================================================
    VectorNIP E0_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0));
    E0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1));
    E0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2));
    E0_Block.array() -= 1;
    E0_Block *= 0.5;
    VectorNIP E_BlockDamping = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 5));
    E0_Block += m_Alpha * E_BlockDamping;
    E0_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E1_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0));
    E1_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1));
    E1_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2));
    E1_Block.array() -= 1;
    E1_Block *= 0.5;
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 5));
    E1_Block += m_Alpha * E_BlockDamping;
    E1_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E2_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0));
    E2_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1));
    E2_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2));
    E2_Block.array() -= 1;
    E2_Block *= 0.5;
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 5));
    E2_Block += m_Alpha * E_BlockDamping;
    E2_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E3_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0));
    E3_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1));
    E3_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2));
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 5));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 5));
    E3_Block += m_Alpha * E_BlockDamping;
    E3_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E4_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0));
    E4_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1));
    E4_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2));
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 5));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(432, 5));
    E4_Block += m_Alpha * E_BlockDamping;
    E4_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E5_Block = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0));
    E5_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1));
    E5_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2));
    E_BlockDamping.noalias() = F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 5));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 3));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 4));
    E_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<216, 1>(216, 5));
    E5_Block += m_Alpha * E_BlockDamping;
    E5_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP SPK2_0_Block = D(0, 0)*E0_Block + D(0, 1)*E1_Block + D(0, 2)*E2_Block + D(0, 3)*E3_Block + D(0, 4)*E4_Block + D(0, 5)*E5_Block;
    VectorNIP SPK2_1_Block = D(1, 0)*E0_Block + D(1, 1)*E1_Block + D(1, 2)*E2_Block + D(1, 3)*E3_Block + D(1, 4)*E4_Block + D(1, 5)*E5_Block;
    VectorNIP SPK2_2_Block = D(2, 0)*E0_Block + D(2, 1)*E1_Block + D(2, 2)*E2_Block + D(2, 3)*E3_Block + D(2, 4)*E4_Block + D(2, 5)*E5_Block;
    VectorNIP SPK2_3_Block = D(3, 0)*E0_Block + D(3, 1)*E1_Block + D(3, 2)*E2_Block + D(3, 3)*E3_Block + D(3, 4)*E4_Block + D(3, 5)*E5_Block;
    VectorNIP SPK2_4_Block = D(4, 0)*E0_Block + D(4, 1)*E1_Block + D(4, 2)*E2_Block + D(4, 3)*E3_Block + D(4, 4)*E4_Block + D(4, 5)*E5_Block;
    VectorNIP SPK2_5_Block = D(5, 0)*E0_Block + D(5, 1)*E1_Block + D(5, 2)*E2_Block + D(5, 3)*E3_Block + D(5, 4)*E4_Block + D(5, 5)*E5_Block;


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_scaled_SD_precompute_col_ordered;
    S_scaled_SD_precompute_col_ordered.resize(32, 648);

    for (auto i = 0; i < 32; i++) {
        S_scaled_SD_precompute_col_ordered.block<1, 216>(i, 0).noalias() =
            SPK2_0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 0)) +
            SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 216)) +
            SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 432));

        S_scaled_SD_precompute_col_ordered.block<1, 216>(i, 216).noalias() =
            SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 0)) +
            SPK2_1_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 216)) +
            SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 432));

        S_scaled_SD_precompute_col_ordered.block<1, 216>(i, 432).noalias() =
            SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 0)) +
            SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 216)) +
            SPK2_2_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 432));
    }

    ChVectorN<double, 528> ScaledMassMatrix = Mfactor * m_MassMatrix;

    //Calculate Just the Non-sparse Upper Triangular Entires and then expand out to full size
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = i; j < 32; j++) {
            //double d = Kfactor * m_SD_precompute_col_ordered.block<1, 648>(i, 0) * S_scaled_SD_precompute_col_ordered.block<1, 648>(j, 0).transpose();
            double d = Kfactor * m_SD_precompute_col_ordered.row(i) * S_scaled_SD_precompute_col_ordered.row(j).transpose();
            d += ScaledMassMatrix(idx);

            H(3 * i, 3 * j) += d;
            H(3 * i + 1, 3 * j + 1) += d;
            H(3 * i + 2, 3 * j + 2) += d;
            if (i != j) {
                H(3 * j, 3 * i) += d;
                H(3 * j + 1, 3 * i + 1) += d;
                H(3 * j + 2, 3 * i + 2) += d;
            }
            idx++;
        }
    }
}

void ChElementBrickANCF_3843_TR08S::ComputeInternalJacobianNoDamping(ChMatrixRef& H, double Kfactor, double Mfactor) {

    MatrixNx3 e_bar;
    CalcCoordMatrix(e_bar);

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;
    ChMatrixNMc<double, 648, 3> F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;

    // ChMatrixNM<double, 48, 180> partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> partial_epsilon_partial_e_Transpose;
    partial_epsilon_partial_e_Transpose.resize(96, 1296);

    for (auto i = 0; i < 32; i++) {
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).transpose());
    }

    //ChMatrixNMc<double, 648, 3> scaled_F_Transpose_col_ordered =
    //    (Kfactor + m_Alpha * Rfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 0) +
    //    (m_Alpha * Kfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<648, 3>(0, 3);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 648, 3> scaled_F_Transpose_col_ordered;
    //scaled_F_Transpose_col_ordered.resize(648, 3);
    //scaled_F_Transpose_col_ordered = Kfactor * F_Transpose_CombinedBlock_col_ordered;
    ChMatrixNMc<double, 648, 3> scaled_F_Transpose_col_ordered = Kfactor * F_Transpose_CombinedBlock_col_ordered;

    for (auto i = 0; i < 3; i++) {
        scaled_F_Transpose_col_ordered.block<216, 1>(0, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<216, 1>(216, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<216, 1>(432, i).array() *= m_GQWeight_det_J_0xi_D.array();
    }

    // ChMatrixNM<double, 48, 432> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 96, 1296> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.resize(96, 1296);

    for (auto i = 0; i < 32; i++) {
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>(3 * i, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 0).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 1, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 1).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 216).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 432).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 648).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 864).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 432).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(432, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 216>((3 * i) + 2, 1080).noalias() = m_SD_precompute_col_ordered.block<1, 216>(i, 216).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 216>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<216, 1>(216, 2).transpose());
    }

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    // ChMatrixNM<double, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    DScaled_Combined_partial_epsilon_partial_e_Transpose.resize(96, 1296);

    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) = D(0, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(0, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(0, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(0, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(0, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(0, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) = D(1, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(1, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(1, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(1, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(1, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(1, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) = D(2, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(2, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(2, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(2, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(2, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(2, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) = D(3, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(3, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(3, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(3, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(3, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(3, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) = D(4, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(4, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(4, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(4, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(4, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(4, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080) = D(5, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 0) + D(5, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 216) + D(5, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 432) + D(5, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 648) + D(5, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 864) + D(5, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<96, 216>(0, 1080);

    H = partial_epsilon_partial_e_Transpose * DScaled_Combined_partial_epsilon_partial_e_Transpose.transpose();

    //===========================================================================================

    VectorNIP E0_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0));
    E0_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1));
    E0_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2));
    E0_Block.array() -= 1;
    E0_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E0_Block *= 0.5;

    VectorNIP E1_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0));
    E1_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1));
    E1_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2));
    E1_Block.array() -= 1;
    E1_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E1_Block *= 0.5;

    VectorNIP E2_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0));
    E2_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1));
    E2_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2));
    E2_Block.array() -= 1;
    E2_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E2_Block *= 0.5;

    VectorNIP E3_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0));
    E3_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1));
    E3_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2));
    E3_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E4_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 0));
    E4_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 1));
    E4_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(432, 2));
    E4_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E5_Block = F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 0));
    E5_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 1));
    E5_Block += F_Transpose_CombinedBlock_col_ordered.block<216, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<216, 1>(216, 2));
    E5_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP SPK2_0_Block = D(0, 0)*E0_Block + D(0, 1)*E1_Block + D(0, 2)*E2_Block + D(0, 3)*E3_Block + D(0, 4)*E4_Block + D(0, 5)*E5_Block;
    VectorNIP SPK2_1_Block = D(1, 0)*E0_Block + D(1, 1)*E1_Block + D(1, 2)*E2_Block + D(1, 3)*E3_Block + D(1, 4)*E4_Block + D(1, 5)*E5_Block;
    VectorNIP SPK2_2_Block = D(2, 0)*E0_Block + D(2, 1)*E1_Block + D(2, 2)*E2_Block + D(2, 3)*E3_Block + D(2, 4)*E4_Block + D(2, 5)*E5_Block;
    VectorNIP SPK2_3_Block = D(3, 0)*E0_Block + D(3, 1)*E1_Block + D(3, 2)*E2_Block + D(3, 3)*E3_Block + D(3, 4)*E4_Block + D(3, 5)*E5_Block;
    VectorNIP SPK2_4_Block = D(4, 0)*E0_Block + D(4, 1)*E1_Block + D(4, 2)*E2_Block + D(4, 3)*E3_Block + D(4, 4)*E4_Block + D(4, 5)*E5_Block;
    VectorNIP SPK2_5_Block = D(5, 0)*E0_Block + D(5, 1)*E1_Block + D(5, 2)*E2_Block + D(5, 3)*E3_Block + D(5, 4)*E4_Block + D(5, 5)*E5_Block;


    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_scaled_SD_precompute_col_ordered;
    S_scaled_SD_precompute_col_ordered.resize(32, 648);

    for (auto i = 0; i < 32; i++) {
        S_scaled_SD_precompute_col_ordered.block<1, 216>(i, 0).noalias() =
            SPK2_0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 0)) +
            SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 216)) +
            SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 432));

        S_scaled_SD_precompute_col_ordered.block<1, 216>(i, 216).noalias() =
            SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 0)) +
            SPK2_1_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 216)) +
            SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 432));

        S_scaled_SD_precompute_col_ordered.block<1, 216>(i, 432).noalias() =
            SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 0)) +
            SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 216)) +
            SPK2_2_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 216>(i, 432));
    }

    ChVectorN<double, 528> ScaledMassMatrix = Mfactor * m_MassMatrix;

    //Calculate Just the Non-sparse Upper Triangular Entires and then expand out to full size
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = i; j < 32; j++) {
            //double d = Kfactor * m_SD_precompute_col_ordered.block<1, 648>(i, 0) * S_scaled_SD_precompute_col_ordered.block<1, 648>(j, 0).transpose();
            double d = Kfactor * m_SD_precompute_col_ordered.row(i) * S_scaled_SD_precompute_col_ordered.row(j).transpose();
            d += ScaledMassMatrix(idx);

            H(3 * i, 3 * j) += d;
            H(3 * i + 1, 3 * j + 1) += d;
            H(3 * i + 2, 3 * j + 2) += d;
            if (i != j) {
                H(3 * j, 3 * i) += d;
                H(3 * j + 1, 3 * i + 1) += d;
                H(3 * j + 2, 3 * i + 2) += d;
            }
            idx++;
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x96 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi(Matrix3x3N& Sxi, double xi, double eta, double zeta) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    Sxi.setZero();

    for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
        Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
        Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
        Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
    }
}

// 32x1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) = 0.0625*(zeta - 1)*(xi - 1)*(eta - 1)*(eta*eta + eta + xi*xi + xi + zeta*zeta + zeta - 2);
    Sxi_compact(1) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta - 1)*(eta - 1);
    Sxi_compact(2) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta - 1)*(xi - 1);
    Sxi_compact(3) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi - 1)*(eta - 1);
    Sxi_compact(4) = -0.0625*(zeta - 1)*(xi + 1)*(eta - 1)*(eta*eta + eta + xi*xi - xi + zeta*zeta + zeta - 2);
    Sxi_compact(5) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta - 1)*(eta - 1);
    Sxi_compact(6) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta - 1)*(xi + 1);
    Sxi_compact(7) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi + 1)*(eta - 1);
    Sxi_compact(8) = 0.0625*(zeta - 1)*(xi + 1)*(eta + 1)*(eta*eta - eta + xi*xi - xi + zeta*zeta + zeta - 2);
    Sxi_compact(9) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta - 1)*(eta + 1);
    Sxi_compact(10) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta - 1)*(xi + 1);
    Sxi_compact(11) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi + 1)*(eta + 1);
    Sxi_compact(12) = -0.0625*(zeta - 1)*(xi - 1)*(eta + 1)*(eta*eta - eta + xi*xi + xi + zeta*zeta + zeta - 2);
    Sxi_compact(13) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta - 1)*(eta + 1);
    Sxi_compact(14) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta - 1)*(xi - 1);
    Sxi_compact(15) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi - 1)*(eta + 1);
    Sxi_compact(16) = -0.0625*(zeta + 1)*(xi - 1)*(eta - 1)*(eta*eta + eta + xi*xi + xi + zeta*zeta - zeta - 2);
    Sxi_compact(17) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta + 1)*(eta - 1);
    Sxi_compact(18) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta + 1)*(xi - 1);
    Sxi_compact(19) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi - 1)*(eta - 1);
    Sxi_compact(20) = 0.0625*(zeta + 1)*(xi + 1)*(eta - 1)*(eta*eta + eta + xi*xi - xi + zeta*zeta - zeta - 2);
    Sxi_compact(21) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta + 1)*(eta - 1);
    Sxi_compact(22) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta + 1)*(xi + 1);
    Sxi_compact(23) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi + 1)*(eta - 1);
    Sxi_compact(24) = -0.0625*(zeta + 1)*(xi + 1)*(eta + 1)*(eta*eta - eta + xi*xi - xi + zeta*zeta - zeta - 2);
    Sxi_compact(25) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta + 1)*(eta + 1);
    Sxi_compact(26) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta + 1)*(xi + 1);
    Sxi_compact(27) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi + 1)*(eta + 1);
    Sxi_compact(28) = 0.0625*(zeta + 1)*(xi - 1)*(eta + 1)*(eta*eta - eta + xi*xi + xi + zeta*zeta - zeta - 2);
    Sxi_compact(29) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta + 1)*(eta + 1);
    Sxi_compact(30) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta + 1)*(xi - 1);
    Sxi_compact(31) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi - 1)*(eta + 1);
}

// 3x96 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_xi(Matrix3x3N& Sxi_xi, double xi, double eta, double zeta) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta);
    Sxi_xi.setZero();

    for (unsigned int s = 0; s < Sxi_xi_compact.size(); s++) {
        Sxi_xi(0, 0 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(1, 1 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(2, 2 + (3 * s)) = Sxi_xi_compact(s);
    }
}

// 32x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = 0.0625*(zeta - 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta*zeta + zeta - 3);
    Sxi_xi_compact(1) = 0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta - 1)*(eta - 1);
    Sxi_xi_compact(2) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta - 1);
    Sxi_xi_compact(3) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta - 1);
    Sxi_xi_compact(4) = -0.0625*(zeta - 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta*zeta + zeta - 3);
    Sxi_xi_compact(5) = 0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta - 1)*(eta - 1);
    Sxi_xi_compact(6) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta - 1);
    Sxi_xi_compact(7) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta - 1);
    Sxi_xi_compact(8) = 0.0625*(zeta - 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta*zeta + zeta - 3);
    Sxi_xi_compact(9) = -0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta - 1)*(eta + 1);
    Sxi_xi_compact(10) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta - 1);
    Sxi_xi_compact(11) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta + 1);
    Sxi_xi_compact(12) = -0.0625*(zeta - 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta*zeta + zeta - 3);
    Sxi_xi_compact(13) = -0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta - 1)*(eta + 1);
    Sxi_xi_compact(14) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta - 1);
    Sxi_xi_compact(15) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta + 1);
    Sxi_xi_compact(16) = -0.0625*(zeta + 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta*zeta - zeta - 3);
    Sxi_xi_compact(17) = -0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta + 1)*(eta - 1);
    Sxi_xi_compact(18) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta + 1);
    Sxi_xi_compact(19) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta - 1);
    Sxi_xi_compact(20) = 0.0625*(zeta + 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta*zeta - zeta - 3);
    Sxi_xi_compact(21) = -0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta + 1)*(eta - 1);
    Sxi_xi_compact(22) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta + 1);
    Sxi_xi_compact(23) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta - 1);
    Sxi_xi_compact(24) = -0.0625*(zeta + 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta*zeta - zeta - 3);
    Sxi_xi_compact(25) = 0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta + 1)*(eta + 1);
    Sxi_xi_compact(26) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta + 1);
    Sxi_xi_compact(27) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta + 1);
    Sxi_xi_compact(28) = 0.0625*(zeta + 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta*zeta - zeta - 3);
    Sxi_xi_compact(29) = 0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta + 1)*(eta + 1);
    Sxi_xi_compact(30) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta + 1);
    Sxi_xi_compact(31) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta + 1);
}

// 3x96 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_eta(Matrix3x3N& Sxi_eta, double xi, double eta, double zeta) {
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta);
    Sxi_eta.setZero();

    for (unsigned int s = 0; s < Sxi_eta_compact.size(); s++) {
        Sxi_eta(0, 0 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(1, 1 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(2, 2 + (3 * s)) = Sxi_eta_compact(s);
    }
}

// 32x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = 0.0625*(zeta - 1)*(xi - 1)*(3 * eta*eta + xi*xi + xi + zeta*zeta + zeta - 3);
    Sxi_eta_compact(1) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta - 1);
    Sxi_eta_compact(2) = 0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta - 1)*(xi - 1);
    Sxi_eta_compact(3) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi - 1);
    Sxi_eta_compact(4) = -0.0625*(zeta - 1)*(xi + 1)*(3 * eta*eta + xi*xi - xi + zeta*zeta + zeta - 3);
    Sxi_eta_compact(5) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta - 1);
    Sxi_eta_compact(6) = -0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta - 1)*(xi + 1);
    Sxi_eta_compact(7) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi + 1);
    Sxi_eta_compact(8) = 0.0625*(zeta - 1)*(xi + 1)*(3 * eta*eta + xi*xi - xi + zeta*zeta + zeta - 3);
    Sxi_eta_compact(9) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta - 1);
    Sxi_eta_compact(10) = -0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta - 1)*(xi + 1);
    Sxi_eta_compact(11) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi + 1);
    Sxi_eta_compact(12) = -0.0625*(zeta - 1)*(xi - 1)*(3 * eta*eta + xi*xi + xi + zeta*zeta + zeta - 3);
    Sxi_eta_compact(13) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta - 1);
    Sxi_eta_compact(14) = 0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta - 1)*(xi - 1);
    Sxi_eta_compact(15) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi - 1);
    Sxi_eta_compact(16) = -0.0625*(zeta + 1)*(xi - 1)*(3 * eta*eta + xi*xi + xi + zeta*zeta - zeta - 3);
    Sxi_eta_compact(17) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta + 1);
    Sxi_eta_compact(18) = -0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta + 1)*(xi - 1);
    Sxi_eta_compact(19) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi - 1);
    Sxi_eta_compact(20) = 0.0625*(zeta + 1)*(xi + 1)*(3 * eta*eta + xi*xi - xi + zeta*zeta - zeta - 3);
    Sxi_eta_compact(21) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta + 1);
    Sxi_eta_compact(22) = 0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta + 1)*(xi + 1);
    Sxi_eta_compact(23) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi + 1);
    Sxi_eta_compact(24) = -0.0625*(zeta + 1)*(xi + 1)*(3 * eta*eta + xi*xi - xi + zeta*zeta - zeta - 3);
    Sxi_eta_compact(25) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta + 1);
    Sxi_eta_compact(26) = 0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta + 1)*(xi + 1);
    Sxi_eta_compact(27) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi + 1);
    Sxi_eta_compact(28) = 0.0625*(zeta + 1)*(xi - 1)*(3 * eta*eta + xi*xi + xi + zeta*zeta - zeta - 3);
    Sxi_eta_compact(29) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta + 1);
    Sxi_eta_compact(30) = -0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta + 1)*(xi - 1);
    Sxi_eta_compact(31) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi - 1);
}

// 3x96 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_zeta(Matrix3x3N& Sxi_zeta, double xi, double eta, double zeta) {
    VectorN Sxi_zeta_compact;
    Calc_Sxi_zeta_compact(Sxi_zeta_compact, xi, eta, zeta);
    Sxi_zeta.setZero();

    for (unsigned int s = 0; s < Sxi_zeta_compact.size(); s++) {
        Sxi_zeta(0, 0 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(1, 1 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(2, 2 + (3 * s)) = Sxi_zeta_compact(s);
    }
}

// 32x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]
void ChElementBrickANCF_3843_TR08S::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact,
                                                         double xi,
                                                         double eta,
                                                         double zeta) {
    Sxi_zeta_compact(0) = 0.0625*(xi - 1)*(eta - 1)*(eta*eta + eta + xi*xi + xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(1) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta - 1);
    Sxi_zeta_compact(2) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi - 1);
    Sxi_zeta_compact(3) = 0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi - 1)*(eta - 1);
    Sxi_zeta_compact(4) = -0.0625*(xi + 1)*(eta - 1)*(eta*eta + eta + xi*xi - xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(5) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta - 1);
    Sxi_zeta_compact(6) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi + 1);
    Sxi_zeta_compact(7) = -0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi + 1)*(eta - 1);
    Sxi_zeta_compact(8) = 0.0625*(xi + 1)*(eta + 1)*(eta*eta - eta + xi*xi - xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(9) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta + 1);
    Sxi_zeta_compact(10) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi + 1);
    Sxi_zeta_compact(11) = 0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi + 1)*(eta + 1);
    Sxi_zeta_compact(12) = -0.0625*(xi - 1)*(eta + 1)*(eta*eta - eta + xi*xi + xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(13) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta + 1);
    Sxi_zeta_compact(14) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi - 1);
    Sxi_zeta_compact(15) = -0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi - 1)*(eta + 1);
    Sxi_zeta_compact(16) = -0.0625*(xi - 1)*(eta - 1)*(eta*eta + eta + xi*xi + xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(17) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta - 1);
    Sxi_zeta_compact(18) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi - 1);
    Sxi_zeta_compact(19) = 0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi - 1)*(eta - 1);
    Sxi_zeta_compact(20) = 0.0625*(xi + 1)*(eta - 1)*(eta*eta + eta + xi*xi - xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(21) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta - 1);
    Sxi_zeta_compact(22) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi + 1);
    Sxi_zeta_compact(23) = -0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi + 1)*(eta - 1);
    Sxi_zeta_compact(24) = -0.0625*(xi + 1)*(eta + 1)*(eta*eta - eta + xi*xi - xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(25) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta + 1);
    Sxi_zeta_compact(26) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi + 1);
    Sxi_zeta_compact(27) = 0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi + 1)*(eta + 1);
    Sxi_zeta_compact(28) = 0.0625*(xi - 1)*(eta + 1)*(eta*eta - eta + xi*xi + xi + 3 * zeta*zeta - 3);
    Sxi_zeta_compact(29) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta + 1);
    Sxi_zeta_compact(30) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi - 1);
    Sxi_zeta_compact(31) = -0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi - 1)*(eta + 1);
}

void ChElementBrickANCF_3843_TR08S::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = 0.0625*(zeta - 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta * zeta + zeta - 3);
    Sxi_D(1, 0) = 0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta - 1)*(eta - 1);
    Sxi_D(2, 0) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta - 1);
    Sxi_D(3, 0) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta - 1);
    Sxi_D(4, 0) = -0.0625*(zeta - 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta * zeta + zeta - 3);
    Sxi_D(5, 0) = 0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta - 1)*(eta - 1);
    Sxi_D(6, 0) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta - 1);
    Sxi_D(7, 0) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta - 1);
    Sxi_D(8, 0) = 0.0625*(zeta - 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta * zeta + zeta - 3);
    Sxi_D(9, 0) = -0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta - 1)*(eta + 1);
    Sxi_D(10, 0) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta - 1);
    Sxi_D(11, 0) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta + 1);
    Sxi_D(12, 0) = -0.0625*(zeta - 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta * zeta + zeta - 3);
    Sxi_D(13, 0) = -0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta - 1)*(eta + 1);
    Sxi_D(14, 0) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta - 1);
    Sxi_D(15, 0) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(eta + 1);
    Sxi_D(16, 0) = -0.0625*(zeta + 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta * zeta - zeta - 3);
    Sxi_D(17, 0) = -0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta + 1)*(eta - 1);
    Sxi_D(18, 0) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta + 1);
    Sxi_D(19, 0) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta - 1);
    Sxi_D(20, 0) = 0.0625*(zeta + 1)*(eta - 1)*(eta*eta + eta + 3 * xi*xi + zeta * zeta - zeta - 3);
    Sxi_D(21, 0) = -0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta + 1)*(eta - 1);
    Sxi_D(22, 0) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(zeta + 1);
    Sxi_D(23, 0) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta - 1);
    Sxi_D(24, 0) = -0.0625*(zeta + 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta * zeta - zeta - 3);
    Sxi_D(25, 0) = 0.03125*m_lenX*(xi + 1)*(3 * xi - 1)*(zeta + 1)*(eta + 1);
    Sxi_D(26, 0) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta + 1);
    Sxi_D(27, 0) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta + 1);
    Sxi_D(28, 0) = 0.0625*(zeta + 1)*(eta + 1)*(eta*eta - eta + 3 * xi*xi + zeta * zeta - zeta - 3);
    Sxi_D(29, 0) = 0.03125*m_lenX*(3 * xi + 1)*(xi - 1)*(zeta + 1)*(eta + 1);
    Sxi_D(30, 0) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(zeta + 1);
    Sxi_D(31, 0) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(eta + 1);

    Sxi_D(0, 1) = 0.0625*(zeta - 1)*(xi - 1)*(3 * eta*eta + xi * xi + xi + zeta * zeta + zeta - 3);
    Sxi_D(1, 1) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta - 1);
    Sxi_D(2, 1) = 0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta - 1)*(xi - 1);
    Sxi_D(3, 1) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi - 1);
    Sxi_D(4, 1) = -0.0625*(zeta - 1)*(xi + 1)*(3 * eta*eta + xi * xi - xi + zeta * zeta + zeta - 3);
    Sxi_D(5, 1) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta - 1);
    Sxi_D(6, 1) = -0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta - 1)*(xi + 1);
    Sxi_D(7, 1) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi + 1);
    Sxi_D(8, 1) = 0.0625*(zeta - 1)*(xi + 1)*(3 * eta*eta + xi * xi - xi + zeta * zeta + zeta - 3);
    Sxi_D(9, 1) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta - 1);
    Sxi_D(10, 1) = -0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta - 1)*(xi + 1);
    Sxi_D(11, 1) = 0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi + 1);
    Sxi_D(12, 1) = -0.0625*(zeta - 1)*(xi - 1)*(3 * eta*eta + xi * xi + xi + zeta * zeta + zeta - 3);
    Sxi_D(13, 1) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta - 1);
    Sxi_D(14, 1) = 0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta - 1)*(xi - 1);
    Sxi_D(15, 1) = -0.03125*m_lenZ*(zeta + 1)*(zeta - 1)*(zeta - 1)*(xi - 1);
    Sxi_D(16, 1) = -0.0625*(zeta + 1)*(xi - 1)*(3 * eta*eta + xi * xi + xi + zeta * zeta - zeta - 3);
    Sxi_D(17, 1) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta + 1);
    Sxi_D(18, 1) = -0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta + 1)*(xi - 1);
    Sxi_D(19, 1) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi - 1);
    Sxi_D(20, 1) = 0.0625*(zeta + 1)*(xi + 1)*(3 * eta*eta + xi * xi - xi + zeta * zeta - zeta - 3);
    Sxi_D(21, 1) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta + 1);
    Sxi_D(22, 1) = 0.03125*m_lenY*(3 * eta + 1)*(eta - 1)*(zeta + 1)*(xi + 1);
    Sxi_D(23, 1) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi + 1);
    Sxi_D(24, 1) = -0.0625*(zeta + 1)*(xi + 1)*(3 * eta*eta + xi * xi - xi + zeta * zeta - zeta - 3);
    Sxi_D(25, 1) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(zeta + 1);
    Sxi_D(26, 1) = 0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta + 1)*(xi + 1);
    Sxi_D(27, 1) = 0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi + 1);
    Sxi_D(28, 1) = 0.0625*(zeta + 1)*(xi - 1)*(3 * eta*eta + xi * xi + xi + zeta * zeta - zeta - 3);
    Sxi_D(29, 1) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(zeta + 1);
    Sxi_D(30, 1) = -0.03125*m_lenY*(eta + 1)*(3 * eta - 1)*(zeta + 1)*(xi - 1);
    Sxi_D(31, 1) = -0.03125*m_lenZ*(zeta - 1)*(zeta + 1)*(zeta + 1)*(xi - 1);

    Sxi_D(0, 2) = 0.0625*(xi - 1)*(eta - 1)*(eta*eta + eta + xi * xi + xi + 3 * zeta*zeta - 3);
    Sxi_D(1, 2) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta - 1);
    Sxi_D(2, 2) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi - 1);
    Sxi_D(3, 2) = 0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi - 1)*(eta - 1);
    Sxi_D(4, 2) = -0.0625*(xi + 1)*(eta - 1)*(eta*eta + eta + xi * xi - xi + 3 * zeta*zeta - 3);
    Sxi_D(5, 2) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta - 1);
    Sxi_D(6, 2) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi + 1);
    Sxi_D(7, 2) = -0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi + 1)*(eta - 1);
    Sxi_D(8, 2) = 0.0625*(xi + 1)*(eta + 1)*(eta*eta - eta + xi * xi - xi + 3 * zeta*zeta - 3);
    Sxi_D(9, 2) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta + 1);
    Sxi_D(10, 2) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi + 1);
    Sxi_D(11, 2) = 0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi + 1)*(eta + 1);
    Sxi_D(12, 2) = -0.0625*(xi - 1)*(eta + 1)*(eta*eta - eta + xi * xi + xi + 3 * zeta*zeta - 3);
    Sxi_D(13, 2) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta + 1);
    Sxi_D(14, 2) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi - 1);
    Sxi_D(15, 2) = -0.03125*m_lenZ*(3 * zeta + 1)*(zeta - 1)*(xi - 1)*(eta + 1);
    Sxi_D(16, 2) = -0.0625*(xi - 1)*(eta - 1)*(eta*eta + eta + xi * xi + xi + 3 * zeta*zeta - 3);
    Sxi_D(17, 2) = -0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta - 1);
    Sxi_D(18, 2) = -0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi - 1);
    Sxi_D(19, 2) = 0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi - 1)*(eta - 1);
    Sxi_D(20, 2) = 0.0625*(xi + 1)*(eta - 1)*(eta*eta + eta + xi * xi - xi + 3 * zeta*zeta - 3);
    Sxi_D(21, 2) = -0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta - 1);
    Sxi_D(22, 2) = 0.03125*m_lenY*(eta + 1)*(eta - 1)*(eta - 1)*(xi + 1);
    Sxi_D(23, 2) = -0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi + 1)*(eta - 1);
    Sxi_D(24, 2) = -0.0625*(xi + 1)*(eta + 1)*(eta*eta - eta + xi * xi - xi + 3 * zeta*zeta - 3);
    Sxi_D(25, 2) = 0.03125*m_lenX*(xi - 1)*(xi + 1)*(xi + 1)*(eta + 1);
    Sxi_D(26, 2) = 0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi + 1);
    Sxi_D(27, 2) = 0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi + 1)*(eta + 1);
    Sxi_D(28, 2) = 0.0625*(xi - 1)*(eta + 1)*(eta*eta - eta + xi * xi + xi + 3 * zeta*zeta - 3);
    Sxi_D(29, 2) = 0.03125*m_lenX*(xi + 1)*(xi - 1)*(xi - 1)*(eta + 1);
    Sxi_D(30, 2) = -0.03125*m_lenY*(eta - 1)*(eta + 1)*(eta + 1)*(xi - 1);
    Sxi_D(31, 2) = -0.03125*m_lenZ*(zeta + 1)*(3 * zeta - 1)*(xi - 1)*(eta + 1);
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR08S::CalcCoordVector(Vector3N& e) {
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

void ChElementBrickANCF_3843_TR08S::CalcCoordMatrix(Matrix3xN& ebar) {
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

void ChElementBrickANCF_3843_TR08S::CalcCoordMatrix(MatrixNx3c& ebar) {
    ebar.row(0) = m_nodes[0]->GetPos().eigen();
    ebar.row(1) = m_nodes[0]->GetD().eigen();
    ebar.row(2) = m_nodes[0]->GetDD().eigen();
    ebar.row(3) = m_nodes[0]->GetDDD().eigen();

    ebar.row(4) = m_nodes[1]->GetPos().eigen();
    ebar.row(5) = m_nodes[1]->GetD().eigen();
    ebar.row(6) = m_nodes[1]->GetDD().eigen();
    ebar.row(7) = m_nodes[1]->GetDDD().eigen();

    ebar.row(8) = m_nodes[2]->GetPos().eigen();
    ebar.row(9) = m_nodes[2]->GetD().eigen();
    ebar.row(10) = m_nodes[2]->GetDD().eigen();
    ebar.row(11) = m_nodes[2]->GetDDD().eigen();

    ebar.row(12) = m_nodes[3]->GetPos().eigen();
    ebar.row(13) = m_nodes[3]->GetD().eigen();
    ebar.row(14) = m_nodes[3]->GetDD().eigen();
    ebar.row(15) = m_nodes[3]->GetDDD().eigen();

    ebar.row(16) = m_nodes[4]->GetPos().eigen();
    ebar.row(17) = m_nodes[4]->GetD().eigen();
    ebar.row(18) = m_nodes[4]->GetDD().eigen();
    ebar.row(19) = m_nodes[4]->GetDDD().eigen();

    ebar.row(20) = m_nodes[5]->GetPos().eigen();
    ebar.row(21) = m_nodes[5]->GetD().eigen();
    ebar.row(22) = m_nodes[5]->GetDD().eigen();
    ebar.row(23) = m_nodes[5]->GetDDD().eigen();

    ebar.row(24) = m_nodes[6]->GetPos().eigen();
    ebar.row(25) = m_nodes[6]->GetD().eigen();
    ebar.row(26) = m_nodes[6]->GetDD().eigen();
    ebar.row(27) = m_nodes[6]->GetDDD().eigen();

    ebar.row(28) = m_nodes[7]->GetPos().eigen();
    ebar.row(29) = m_nodes[7]->GetD().eigen();
    ebar.row(30) = m_nodes[7]->GetDD().eigen();
    ebar.row(31) = m_nodes[7]->GetDDD().eigen();
}

void ChElementBrickANCF_3843_TR08S::CalcCoordMatrix(MatrixNx3& ebar) {
    ebar.row(0) = m_nodes[0]->GetPos().eigen();
    ebar.row(1) = m_nodes[0]->GetD().eigen();
    ebar.row(2) = m_nodes[0]->GetDD().eigen();
    ebar.row(3) = m_nodes[0]->GetDDD().eigen();

    ebar.row(4) = m_nodes[1]->GetPos().eigen();
    ebar.row(5) = m_nodes[1]->GetD().eigen();
    ebar.row(6) = m_nodes[1]->GetDD().eigen();
    ebar.row(7) = m_nodes[1]->GetDDD().eigen();

    ebar.row(8) = m_nodes[2]->GetPos().eigen();
    ebar.row(9) = m_nodes[2]->GetD().eigen();
    ebar.row(10) = m_nodes[2]->GetDD().eigen();
    ebar.row(11) = m_nodes[2]->GetDDD().eigen();

    ebar.row(12) = m_nodes[3]->GetPos().eigen();
    ebar.row(13) = m_nodes[3]->GetD().eigen();
    ebar.row(14) = m_nodes[3]->GetDD().eigen();
    ebar.row(15) = m_nodes[3]->GetDDD().eigen();

    ebar.row(16) = m_nodes[4]->GetPos().eigen();
    ebar.row(17) = m_nodes[4]->GetD().eigen();
    ebar.row(18) = m_nodes[4]->GetDD().eigen();
    ebar.row(19) = m_nodes[4]->GetDDD().eigen();

    ebar.row(20) = m_nodes[5]->GetPos().eigen();
    ebar.row(21) = m_nodes[5]->GetD().eigen();
    ebar.row(22) = m_nodes[5]->GetDD().eigen();
    ebar.row(23) = m_nodes[5]->GetDDD().eigen();

    ebar.row(24) = m_nodes[6]->GetPos().eigen();
    ebar.row(25) = m_nodes[6]->GetD().eigen();
    ebar.row(26) = m_nodes[6]->GetDD().eigen();
    ebar.row(27) = m_nodes[6]->GetDDD().eigen();

    ebar.row(28) = m_nodes[7]->GetPos().eigen();
    ebar.row(29) = m_nodes[7]->GetD().eigen();
    ebar.row(30) = m_nodes[7]->GetDD().eigen();
    ebar.row(31) = m_nodes[7]->GetDDD().eigen();
}

void ChElementBrickANCF_3843_TR08S::CalcCoordDerivVector(Vector3N& edot) {
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

void ChElementBrickANCF_3843_TR08S::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
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

void ChElementBrickANCF_3843_TR08S::CalcCoordDerivMatrix(MatrixNx3c& ebardot) {
    ebardot.row(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.row(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.row(2) = m_nodes[0]->GetDD_dt().eigen();
    ebardot.row(3) = m_nodes[0]->GetDDD_dt().eigen();

    ebardot.row(4) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.row(5) = m_nodes[1]->GetD_dt().eigen();
    ebardot.row(6) = m_nodes[1]->GetDD_dt().eigen();
    ebardot.row(7) = m_nodes[1]->GetDDD_dt().eigen();

    ebardot.row(8) = m_nodes[2]->GetPos_dt().eigen();
    ebardot.row(9) = m_nodes[2]->GetD_dt().eigen();
    ebardot.row(10) = m_nodes[2]->GetDD_dt().eigen();
    ebardot.row(11) = m_nodes[2]->GetDDD_dt().eigen();

    ebardot.row(12) = m_nodes[3]->GetPos_dt().eigen();
    ebardot.row(13) = m_nodes[3]->GetD_dt().eigen();
    ebardot.row(14) = m_nodes[3]->GetDD_dt().eigen();
    ebardot.row(15) = m_nodes[3]->GetDDD_dt().eigen();

    ebardot.row(16) = m_nodes[4]->GetPos_dt().eigen();
    ebardot.row(17) = m_nodes[4]->GetD_dt().eigen();
    ebardot.row(18) = m_nodes[4]->GetDD_dt().eigen();
    ebardot.row(19) = m_nodes[4]->GetDDD_dt().eigen();

    ebardot.row(20) = m_nodes[5]->GetPos_dt().eigen();
    ebardot.row(21) = m_nodes[5]->GetD_dt().eigen();
    ebardot.row(22) = m_nodes[5]->GetDD_dt().eigen();
    ebardot.row(23) = m_nodes[5]->GetDDD_dt().eigen();

    ebardot.row(24) = m_nodes[6]->GetPos_dt().eigen();
    ebardot.row(25) = m_nodes[6]->GetD_dt().eigen();
    ebardot.row(26) = m_nodes[6]->GetDD_dt().eigen();
    ebardot.row(27) = m_nodes[6]->GetDDD_dt().eigen();

    ebardot.row(28) = m_nodes[7]->GetPos_dt().eigen();
    ebardot.row(29) = m_nodes[7]->GetD_dt().eigen();
    ebardot.row(30) = m_nodes[7]->GetDD_dt().eigen();
    ebardot.row(31) = m_nodes[7]->GetDDD_dt().eigen();
}

void ChElementBrickANCF_3843_TR08S::CalcCombinedCoordMatrix(MatrixNx6& ebar_ebardot) {
    ebar_ebardot.block<1, 3>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(1, 0) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.block<1, 3>(1, 3) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(2, 0) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(2, 3) = m_nodes[0]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(3, 0) = m_nodes[0]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(3, 3) = m_nodes[0]->GetDDD_dt().eigen();

    ebar_ebardot.block<1, 3>(4, 0) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(4, 3) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(5, 0) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.block<1, 3>(5, 3) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(6, 0) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(6, 3) = m_nodes[1]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(7, 0) = m_nodes[1]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(7, 3) = m_nodes[1]->GetDDD_dt().eigen();

    ebar_ebardot.block<1, 3>(8, 0) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(8, 3) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(9, 0) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.block<1, 3>(9, 3) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(10, 0) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(10, 3) = m_nodes[2]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(11, 0) = m_nodes[2]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(11, 3) = m_nodes[2]->GetDDD_dt().eigen();

    ebar_ebardot.block<1, 3>(12, 0) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(12, 3) = m_nodes[3]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(13, 0) = m_nodes[3]->GetD().eigen();
    ebar_ebardot.block<1, 3>(13, 3) = m_nodes[3]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(14, 0) = m_nodes[3]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(14, 3) = m_nodes[3]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(15, 0) = m_nodes[3]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(15, 3) = m_nodes[3]->GetDDD_dt().eigen();



    ebar_ebardot.block<1, 3>(16, 0) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(16, 3) = m_nodes[4]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(17, 0) = m_nodes[4]->GetD().eigen();
    ebar_ebardot.block<1, 3>(17, 3) = m_nodes[4]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(18, 0) = m_nodes[4]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(18, 3) = m_nodes[4]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(19, 0) = m_nodes[4]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(19, 3) = m_nodes[4]->GetDDD_dt().eigen();

    ebar_ebardot.block<1, 3>(20, 0) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(20, 3) = m_nodes[5]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(21, 0) = m_nodes[5]->GetD().eigen();
    ebar_ebardot.block<1, 3>(21, 3) = m_nodes[5]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(22, 0) = m_nodes[5]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(22, 3) = m_nodes[5]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(23, 0) = m_nodes[5]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(23, 3) = m_nodes[5]->GetDDD_dt().eigen();

    ebar_ebardot.block<1, 3>(24, 0) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(24, 3) = m_nodes[6]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(25, 0) = m_nodes[6]->GetD().eigen();
    ebar_ebardot.block<1, 3>(25, 3) = m_nodes[6]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(26, 0) = m_nodes[6]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(26, 3) = m_nodes[6]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(27, 0) = m_nodes[6]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(27, 3) = m_nodes[6]->GetDDD_dt().eigen();

    ebar_ebardot.block<1, 3>(28, 0) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(28, 3) = m_nodes[7]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(29, 0) = m_nodes[7]->GetD().eigen();
    ebar_ebardot.block<1, 3>(29, 3) = m_nodes[7]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(30, 0) = m_nodes[7]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(30, 3) = m_nodes[7]->GetDD_dt().eigen();
    ebar_ebardot.block<1, 3>(31, 0) = m_nodes[7]->GetDDD().eigen();
    ebar_ebardot.block<1, 3>(31, 3) = m_nodes[7]->GetDDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementBrickANCF_3843_TR08S::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementBrickANCF_3843_TR08S::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR08S::EvaluateSectionFrame(const double xi,
                                                        const double eta,
                                                        const double zeta,
                                                        ChVector<>& point,
                                                        ChQuaternion<>& rot) {
    Matrix3x3N Sxi;
    Matrix3x3N Sxi_xi;
    Matrix3x3N Sxi_eta;
    Calc_Sxi(Sxi, xi, eta, zeta);
    Calc_Sxi_xi(Sxi_xi, xi, eta, zeta);
    Calc_Sxi_eta(Sxi_eta, xi, eta, zeta);

    Vector3N e;
    CalcCoordVector(e);

    // r = Se
    point = Sxi * e;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector<double> MidsurfaceX = Sxi_xi * e;
    ChVector<double> MidsurfaceY = Sxi_eta * e;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the shell xi axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the shell eta axis
    ChMatrix33<> msect;
    msect.Set_A_Xdir(MidsurfaceX, MidsurfaceY);

    rot = msect.Get_A_quaternion();
}

void ChElementBrickANCF_3843_TR08S::EvaluateSectionPoint(const double u, const double v, const double w, ChVector<>& point) {
    Matrix3x3N Sxi;
    Calc_Sxi(Sxi, u, v, w);

    Vector3N e;
    CalcCoordVector(e);

    // r = Se
    point = Sxi * e;
}


// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).
void ChElementBrickANCF_3843_TR08S::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD().eigen();

    mD.segment(block_offset + 27, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[3]->GetDD().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[4]->GetDD().eigen();

    mD.segment(block_offset + 45, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(block_offset + 48, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[5]->GetDD().eigen();

    mD.segment(block_offset + 54, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(block_offset + 60, 3) = m_nodes[6]->GetDD().eigen();

    mD.segment(block_offset + 63, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[7]->GetDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).
void ChElementBrickANCF_3843_TR08S::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD_dt().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD_dt().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD_dt().eigen();

    mD.segment(block_offset + 27, 3) = m_nodes[3]->GetPos_dt().eigen();
    mD.segment(block_offset + 30, 3) = m_nodes[3]->GetD_dt().eigen();
    mD.segment(block_offset + 33, 3) = m_nodes[3]->GetDD_dt().eigen();

    mD.segment(block_offset + 36, 3) = m_nodes[4]->GetPos_dt().eigen();
    mD.segment(block_offset + 39, 3) = m_nodes[4]->GetD_dt().eigen();
    mD.segment(block_offset + 42, 3) = m_nodes[4]->GetDD_dt().eigen();

    mD.segment(block_offset + 45, 3) = m_nodes[5]->GetPos_dt().eigen();
    mD.segment(block_offset + 48, 3) = m_nodes[5]->GetD_dt().eigen();
    mD.segment(block_offset + 51, 3) = m_nodes[5]->GetDD_dt().eigen();

    mD.segment(block_offset + 54, 3) = m_nodes[6]->GetPos_dt().eigen();
    mD.segment(block_offset + 57, 3) = m_nodes[6]->GetD_dt().eigen();
    mD.segment(block_offset + 60, 3) = m_nodes[6]->GetDD_dt().eigen();

    mD.segment(block_offset + 63, 3) = m_nodes[7]->GetPos_dt().eigen();
    mD.segment(block_offset + 66, 3) = m_nodes[7]->GetD_dt().eigen();
    mD.segment(block_offset + 69, 3) = m_nodes[7]->GetDD_dt().eigen();
}

/// Increment all DOFs using a delta.
void ChElementBrickANCF_3843_TR08S::LoadableStateIncrement(const unsigned int off_x,
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
void ChElementBrickANCF_3843_TR08S::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U,V,W) coordinates of the surface.
void ChElementBrickANCF_3843_TR08S::ComputeNF(
    const double U,              // parametric coordinate in volume
    const double V,              // parametric coordinate in volume
    const double W,              // parametric coordinate in volume
    ChVectorDynamic<>& Qi,       // Return result of N'*F  here, maybe with offset block_offset
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is = n.field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force
    Matrix3x3N Sxi;
    Calc_Sxi(Sxi, U, V, W);
    Qi = Sxi.transpose() * F.segment(0, 3);

    // Compute the generalized force vector for the applied moment
    Matrix3x3N Sxi_xi;
    Matrix3x3N Sxi_eta;
    Matrix3x3N Sxi_zeta;
    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;
    Matrix3x3N G;
    Vector3N e;

    CalcCoordVector(e);

    Calc_Sxi_xi(Sxi_xi, U, V, W);
    Calc_Sxi_eta(Sxi_eta, U, V, W);
    Calc_Sxi_zeta(Sxi_zeta, U, V, W);

    J_Cxi.row(0) = Sxi_xi * e;
    J_Cxi.row(1) = Sxi_eta * e;
    J_Cxi.row(2) = Sxi_zeta * e;

    J_Cxi_Inv = J_Cxi.inverse();

    G.row(0) =
        0.5 * (Sxi_xi.row(2) * J_Cxi_Inv(0, 1) + Sxi_eta.row(2) * J_Cxi_Inv(1, 1) + Sxi_zeta.row(2) * J_Cxi_Inv(2, 1) -
               Sxi_xi.row(1) * J_Cxi_Inv(0, 2) - Sxi_eta.row(1) * J_Cxi_Inv(1, 2) - Sxi_zeta.row(1) * J_Cxi_Inv(2, 2));
    G.row(1) =
        0.5 * (Sxi_xi.row(0) * J_Cxi_Inv(0, 2) + Sxi_eta.row(0) * J_Cxi_Inv(1, 2) + Sxi_zeta.row(0) * J_Cxi_Inv(2, 2) -
               Sxi_xi.row(2) * J_Cxi_Inv(0, 0) - Sxi_eta.row(2) * J_Cxi_Inv(1, 0) - Sxi_zeta.row(2) * J_Cxi_Inv(2, 0));
    G.row(2) =
        0.5 * (Sxi_xi.row(1) * J_Cxi_Inv(0, 0) + Sxi_eta.row(1) * J_Cxi_Inv(1, 0) + Sxi_zeta.row(1) * J_Cxi_Inv(2, 0) -
               Sxi_xi.row(0) * J_Cxi_Inv(0, 1) - Sxi_eta.row(0) * J_Cxi_Inv(1, 1) - Sxi_zeta.row(0) * J_Cxi_Inv(2, 1));

    Qi += G.transpose() * F.segment(3, 3);

    // Compute the element Jacobian between the current configuration and the normalized configuration
    // This is different than the element Jacobian between the reference configuration and the normalized
    //  configuration used in the internal force calculations
    detJ = J_Cxi.determinant();
}

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// Calculate average element density (needed for ChLoaderVolumeGravity).
double ChElementBrickANCF_3843_TR08S::GetDensity() {
    return GetMaterial()->Get_rho();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3843_TR08S(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBrickANCF_3843_TR08S::GetStaticGQTables() {
    return &static_tables_3843_TR08S;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBrickANCF_3843_TR08S methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBrickANCF_3843_TR08S::ChMaterialBrickANCF_3843_TR08S(double rho,  // material density
                                                             double E,    // Young's modulus
                                                             double nu    // Poisson ratio
                                                             )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G));
}

// Construct a (possibly) orthotropic material.
ChMaterialBrickANCF_3843_TR08S::ChMaterialBrickANCF_3843_TR08S(
    double rho,            // material density
    const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
    const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
    const ChVector<>& G    // shear moduli (G_xy, G_xz, G_yz)
    )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G);
}

// Calculate the matrix form of two stiffness tensors used by the ANCF shell for selective reduced integration of the
// Poisson effect
void ChMaterialBrickANCF_3843_TR08S::Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G) {
    // orthotropic material ref: http://homes.civil.aau.dk/lda/Continuum/material.pdf
    // except position of the shear terms is different to match the original ANCF reference paper

    double nu_12 = nu.x();
    double nu_13 = nu.y();
    double nu_23 = nu.z();
    double nu_21 = nu_12 * E.y() / E.x();
    double nu_31 = nu_13 * E.z() / E.x();
    double nu_32 = nu_23 * E.z() / E.y();
    double k = 1.0 - nu_23 * nu_32 - nu_12 * nu_21 - nu_13 * nu_31 - nu_12 * nu_23 * nu_31 - nu_21 * nu_32 * nu_13;

    ChMatrixNM<double, 6, 6> D;
    D.setZero();
    D(0, 0) = E.x() * (1 - nu_23 * nu_32) / k;
    D(1, 0) = E.y() * (nu_13 * nu_32 + nu_12) / k;
    D(2, 0) = E.z() * (nu_12 * nu_23 + nu_13) / k;

    D(0, 1) = E.x() * (nu_23 * nu_31 + nu_21) / k;
    D(1, 1) = E.y() * (1 - nu_13 * nu_31) / k;
    D(2, 1) = E.z() * (nu_13 * nu_21 + nu_23) / k;

    D(0, 2) = E.x() * (nu_21 * nu_32 + nu_31) / k;
    D(1, 2) = E.y() * (nu_12 * nu_31 + nu_32) / k;
    D(2, 2) = E.z() * (1 - nu_12 * nu_21) / k;

    D(3, 3) = G.z();
    D(4, 4) = G.y();
    D(5, 5) = G.x();

    m_D = D;

    // Component of Stiffness Tensor that does not contain the Poisson Effect
    m_D0.setZero();
    m_D0(0, 0) = E.x();
    m_D0(1, 1) = E.y();
    m_D0(2, 2) = E.z();
    m_D0(3, 3) = G.z();
    m_D0(4, 4) = G.y();
    m_D0(5, 5) = G.x();

    // Remaining components of the Stiffness Tensor that contain the Poisson Effect
    m_Dv = D - m_D0;
}

}  // end of namespace fea
}  // end of namespace chrono
