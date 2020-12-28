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
// ANCF beam element with 3 nodes. Description of this element and its internal
// forces may be found in Nachbagauer, Gruber, Gerstmayr, "Structural and Continuum
// Mechanics Approaches for a 3D Shear Deformable ANCF Beam Finite Element:
// Application to Static and Linearized Dynamic Examples", Journal of Computational
// and Nonlinear Dynamics, 2013, April 2013, Vol. 8, 021004.
// =============================================================================
// Internal Force Calculation Method is based on:  Gerstmayr, J., Shabana, A.A.:
// Efficient integration of the elastic forces and thin three-dimensional beam
// elements in the absolute nodal coordinate formulation.In: Proceedings of the
// Multibody Dynamics Eccomas thematic Conference, Madrid(2005)
// =============================================================================
// TR08 = a Gerstmayr style implementation of the element with pre-calculation
//     of the terms needed for the generalized internal force calculation with
//     an analytical Jacobian that is integrated across all GQ points at once
//
//  Mass Matrix = Constant, pre-calculated 9x9 matrix
//
//  Generalized Force due to gravity = Constant 27x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Gerstmayr method:
//     Dense Math: e_bar = 3x9 and S_bar = 9x1
//     Math is based on the method presented by Gerstmayr and Shabana
//     1 less than "Full Integration" Number of GQ Integration Points (4x2x2)
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
#include "chrono/fea/ChElementBeamANCF_TR08.h"
#include <cmath>
#include <Eigen/Dense>

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBeamANCF_TR08::ChElementBeamANCF_TR08()
    : m_gravity_on(false), m_thicknessY(0), m_thicknessZ(0), m_lenX(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(3);

    // m_F_Transpose_CombinedBlock_col_ordered.setZero();
    // m_F_Transpose_CombinedBlockDamping_col_ordered.setZero();
    // m_SPK2_0_D0_Block.setZero();
    // m_SPK2_1_D0_Block.setZero();
    // m_SPK2_2_D0_Block.setZero();
    // m_SPK2_3_D0_Block.setZero();
    // m_SPK2_4_D0_Block.setZero();
    // m_SPK2_5_D0_Block.setZero();
    // m_Sdiag_0_Dv_Block.setZero();
    // m_Sdiag_1_Dv_Block.setZero();
    // m_Sdiag_2_Dv_Block.setZero();
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBeamANCF_TR08::SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
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
    mvars.push_back(&m_nodes[0]->Variables_D());
    mvars.push_back(&m_nodes[0]->Variables_DD());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->Variables_D());
    mvars.push_back(&m_nodes[2]->Variables_DD());

    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_e0_bar);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementBeamANCF_TR08::SetupInitial(ChSystem* system) {
    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementBeamANCF_TR08::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementBeamANCF_TR08::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(24, 3) = m_nodes[2]->GetDD().eigen();
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
void ChElementBeamANCF_TR08::ComputeKRMmatricesGlobal(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 27) && (H.cols() == 27));

    ////Use H to accumulate the Dense part of the Jacobian, so set it to all zeros
    ////H.setZero();

#if true

    if (m_damping_enabled) {  // If linear Kelvin-Voigt viscoelastic material model is enabled
        ComputeInternalJacobianDamping(H, -Kfactor, -Rfactor, Mfactor);
    } else {
        ComputeInternalJacobianNoDamping(H, -Kfactor, Mfactor);
    }

#else
    ChMatrixNM<double, 27, 27> JacobianMatrix;
    ComputeInternalJacobians(JacobianMatrix, Kfactor, Rfactor);
    // Load Jac + Mfactor*[M] into H
    H = JacobianMatrix;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) += Mfactor * m_MassMatrix(i, j);
            H(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(i, j);
            H(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(i, j);
        }
    }
#endif  // true
}

// Return the mass matrix.
void ChElementBeamANCF_TR08::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Inflate the Mass Matrix since it is stored in compact form.
    // In MATLAB notation:
    // M(1:3:end,1:3:end) = m_MassMatrix;
    // M(2:3:end,2:3:end) = m_MassMatrix;
    // M(3:3:end,3:3:end) = m_MassMatrix;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            M(3 * i, 3 * j) = m_MassMatrix(i, j);
            M(3 * i + 1, 3 * j + 1) = m_MassMatrix(i, j);
            M(3 * i + 2, 3 * j + 2) = m_MassMatrix(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------
void ChElementBeamANCF_TR08::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, 5 GQ Points are needed in the xi direction
    //  and 2 GQ Points are needed in the eta & zeta directions
    //  for exact integration of the element's mass matrix, even if
    //  the reference configuration is not straight
    // Since the major pieces of the generalized force due to gravity
    //  can also be used to calculate the mass matrix, these calculations
    //  are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 4;        // 5 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1;  // 2 Point Gauss-Quadrature;

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    m_MassMatrix.setZero();
    m_GravForce.setZero();

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta);  // determinate of the element Jacobian (volume ratio)
                ChMatrixNM<double, 3, 27> Sxi;                     // 3x27 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                ChVectorN<double, 9> Sxi_compact;  // 9x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight * rho * det_J_0xi) * Sxi.transpose() * g_acc.eigen();
                m_MassMatrix += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }
}

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementBeamANCF_TR08::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 3;        // 4 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1;  // 2 Point Gauss-Quadrature;

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determiate of the element Jacobian for later Calculating the
    // portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_eta_zeta].size() +
                             it_xi * GQTable->Lroots[GQ_idx_eta_zeta].size() * GQTable->Lroots[GQ_idx_eta_zeta].size();
                ChMatrix33<double>
                    J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
                ChMatrixNMc<double, 9, 3> Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_e0_bar.transpose() * Sxi_D;

                ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
                m_SD_precompute_D0.block(0, 3 * index, 9, 3) = SD;
                m_SD_precompute_D0_col0_block.block(0, index, 9, 1) = SD.col(0);
                m_SD_precompute_D0_col1_block.block(0, index, 9, 1) = SD.col(1);
                m_SD_precompute_D0_col2_block.block(0, index, 9, 1) = SD.col(2);
                m_GQWeight_det_J_0xi_D0(index) = -J_0xi.determinant() * GQ_weight;
            }
        }
    }

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determiate of the element Jacobian for later Calculate the portion
    // of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
        ChMatrix33<double> J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
        ChMatrixNMc<double, 9, 3> Sxi_D;  // Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, 0, 0);
        J_0xi.noalias() = m_e0_bar.transpose() * Sxi_D;

        ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
        m_SD_precompute_Dv.block(0, 3 * it_xi, 9, 3) = SD;
        m_SD_precompute_Dv_col0_block.block(0, it_xi, 9, 1) = SD.col(0);
        m_SD_precompute_Dv_col1_block.block(0, it_xi, 9, 1) = SD.col(1);
        m_SD_precompute_Dv_col2_block.block(0, it_xi, 9, 1) = SD.col(2);
        m_GQWeight_det_J_0xi_Dv(it_xi) = -J_0xi.determinant() * GQ_weight;
    }

    m_SD_precompute.block(0, 0, 9, 48) = m_SD_precompute_D0;
    m_SD_precompute.block(0, 48, 9, 12) = m_SD_precompute_Dv;

    m_SD_precompute_col_ordered.block(0, 0, 9, 16) = m_SD_precompute_D0_col0_block;
    m_SD_precompute_col_ordered.block(0, 16, 9, 16) = m_SD_precompute_D0_col1_block;
    m_SD_precompute_col_ordered.block(0, 32, 9, 16) = m_SD_precompute_D0_col2_block;
    m_SD_precompute_col_ordered.block(0, 48, 9, 4) = m_SD_precompute_Dv_col0_block;
    m_SD_precompute_col_ordered.block(0, 52, 9, 4) = m_SD_precompute_Dv_col1_block;
    m_SD_precompute_col_ordered.block(0, 56, 9, 4) = m_SD_precompute_Dv_col2_block;

    m_GQWeight_det_J_0xi.block(0, 0, 16, 1) = m_GQWeight_det_J_0xi_D0;
    m_GQWeight_det_J_0xi.block(16, 0, 4, 1) = m_GQWeight_det_J_0xi_Dv;
}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementBeamANCF_TR08::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0, 0) + m_MassMatrix(0, 3) + m_MassMatrix(0, 6);
    m_nodes[1]->m_TotalMass += m_MassMatrix(3, 3) + m_MassMatrix(3, 0) + m_MassMatrix(3, 6);
    m_nodes[2]->m_TotalMass += m_MassMatrix(6, 6) + m_MassMatrix(6, 0) + m_MassMatrix(6, 3);
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementBeamANCF_TR08::SetAlphaDamp(double a) {
    m_Alpha = a;
    m_2Alpha = 2 * a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementBeamANCF_TR08::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    // Runs faster if the internal force with or without damping calculations are not combined into the same function
    // using the common calculations with an if statement for the damping in the middle to calculate the different
    // P_transpose_scaled_Block components
    if (m_damping_enabled) {
        ChMatrixNM<double, 9, 6> ebar_ebardot;
        CalcCombinedCoordMatrix(ebar_ebardot);

        ComputeInternalForcesAtState(Fi, ebar_ebardot);
    } else {
        ChMatrixNM<double, 9, 3> e_bar;
        CalcCoordMatrix(e_bar);

        ComputeInternalForcesAtStateNoDamping(Fi, e_bar);
    }

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementBeamANCF_TR08::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
                                                          const ChMatrixNM<double, 9, 6>& ebar_ebardot) {
    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full
    // Integration" However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in
    // roughly 1/3 of the calculations

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    ChMatrixNMc<double, 60, 6> F_Transpose_CombinedBlockDamping_col_ordered =
        m_SD_precompute_col_ordered.transpose() * ebar_ebardot;

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    ChVectorN<double, 16> SPK2_0_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0));
    SPK2_0_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1));
    SPK2_0_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2));
    SPK2_0_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_0_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3));
    SPK2_0_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4));
    SPK2_0_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5));
    SPK2_0_D0_Block += m_2Alpha * SPK2_0_D0_BlockDamping;
    SPK2_0_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_0_D0_Block *= (0.5 * D0(0));

    ChVectorN<double, 16> SPK2_1_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0));
    SPK2_1_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1));
    SPK2_1_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2));
    SPK2_1_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_1_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3));
    SPK2_1_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4));
    SPK2_1_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5));
    SPK2_1_D0_Block += m_2Alpha * SPK2_1_D0_BlockDamping;
    SPK2_1_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_1_D0_Block *= (0.5 * D0(1));

    ChVectorN<double, 16> SPK2_2_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0));
    SPK2_2_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1));
    SPK2_2_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2));
    SPK2_2_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_2_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3));
    SPK2_2_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4));
    SPK2_2_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5));
    SPK2_2_D0_Block += m_2Alpha * SPK2_2_D0_BlockDamping;
    SPK2_2_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_2_D0_Block *= (0.5 * D0(2));

    ChVectorN<double, 16> SPK2_3_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0));
    SPK2_3_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1));
    SPK2_3_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2));
    ChVectorN<double, 16> SPK2_3_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5));
    SPK2_3_D0_Block += m_Alpha * SPK2_3_D0_BlockDamping;
    SPK2_3_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_3_D0_Block *= D0(3);

    ChVectorN<double, 16> SPK2_4_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0));
    SPK2_4_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1));
    SPK2_4_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2));
    ChVectorN<double, 16> SPK2_4_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5));
    SPK2_4_D0_Block += m_Alpha * SPK2_4_D0_BlockDamping;
    SPK2_4_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_4_D0_Block *= D0(4);

    ChVectorN<double, 16> SPK2_5_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0));
    SPK2_5_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1));
    SPK2_5_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2));
    ChVectorN<double, 16> SPK2_5_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5));
    SPK2_5_D0_Block += m_Alpha * SPK2_5_D0_BlockDamping;
    SPK2_5_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_5_D0_Block *= D0(5);

    ChMatrixNMc<double, 60, 3>
        P_transpose_scaled_Block_col_ordered;  // 1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled
                                               // across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(SPK2_0_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(SPK2_0_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(SPK2_0_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(SPK2_4_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(SPK2_5_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(SPK2_5_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(SPK2_5_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(SPK2_3_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(SPK2_4_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(SPK2_4_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(SPK2_4_D0_Block) +
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) +=
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(SPK2_2_D0_Block);

    // =============================================================================

    ChVectorN<double, 4> Ediag_0_Dv_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2));
    Ediag_0_Dv_Block.array() -= 1;
    Ediag_0_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_0_Dv_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 3));
    Ediag_0_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 4));
    Ediag_0_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 5));
    Ediag_0_Dv_Block += m_Alpha * Ediag_0_Dv_BlockDamping;
    Ediag_0_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_1_Dv_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2));
    Ediag_1_Dv_Block.array() -= 1;
    Ediag_1_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_1_Dv_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 3));
    Ediag_1_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 4));
    Ediag_1_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 5));
    Ediag_1_Dv_Block += m_Alpha * Ediag_1_Dv_BlockDamping;
    Ediag_1_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_2_Dv_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2));
    Ediag_2_Dv_Block.array() -= 1;
    Ediag_2_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_2_Dv_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 3));
    Ediag_2_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 4));
    Ediag_2_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 5));
    Ediag_2_Dv_Block += m_Alpha * Ediag_2_Dv_BlockDamping;
    Ediag_2_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    ChVectorN<double, 4> Sdiag_0_Dv_Block =
        Dv(0, 0) * Ediag_0_Dv_Block + Dv(1, 0) * Ediag_1_Dv_Block + Dv(2, 0) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block =
        Dv(0, 1) * Ediag_0_Dv_Block + Dv(1, 1) * Ediag_1_Dv_Block + Dv(2, 1) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block =
        Dv(0, 2) * Ediag_0_Dv_Block + Dv(1, 2) * Ediag_1_Dv_Block + Dv(2, 2) * Ediag_2_Dv_Block;

    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).cwiseProduct(Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).cwiseProduct(Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).cwiseProduct(Sdiag_0_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).cwiseProduct(Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).cwiseProduct(Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).cwiseProduct(Sdiag_1_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 0) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).cwiseProduct(Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 1) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).cwiseProduct(Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 2) =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).cwiseProduct(Sdiag_2_Dv_Block);

    // =============================================================================

    ChMatrixNM<double, 9, 3> QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementBeamANCF_TR08::ComputeInternalForcesAtStateNoDamping(ChVectorDynamic<>& Fi,
                                                                   const ChMatrixNM<double, 9, 3>& e_bar) {
    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full
    // Integration" However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in
    // roughly 1/3 of the calculations

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    ChMatrixNMc<double, 60, 3> F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    ChVectorN<double, 16> SPK2_0_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0));
    SPK2_0_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1));
    SPK2_0_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2));
    SPK2_0_D0_Block.array() -= 1;
    SPK2_0_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_0_D0_Block *= (0.5 * D0(0));

    ChVectorN<double, 16> SPK2_1_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0));
    SPK2_1_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1));
    SPK2_1_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2));
    SPK2_1_D0_Block.array() -= 1;
    SPK2_1_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_1_D0_Block *= (0.5 * D0(1));

    ChVectorN<double, 16> SPK2_2_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0));
    SPK2_2_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1));
    SPK2_2_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2));
    SPK2_2_D0_Block.array() -= 1;
    SPK2_2_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_2_D0_Block *= (0.5 * D0(2));

    ChVectorN<double, 16> SPK2_3_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0));
    SPK2_3_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1));
    SPK2_3_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2));
    SPK2_3_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_3_D0_Block *= D0(3);

    ChVectorN<double, 16> SPK2_4_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0));
    SPK2_4_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1));
    SPK2_4_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2));
    SPK2_4_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_4_D0_Block *= D0(4);

    ChVectorN<double, 16> SPK2_5_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0));
    SPK2_5_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1));
    SPK2_5_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2));
    SPK2_5_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_5_D0_Block *= D0(5);

    ChMatrixNMc<double, 60, 3>
        P_transpose_scaled_Block_col_ordered;  // 1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled
                                               // across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(SPK2_0_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 0) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).cwiseProduct(SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(SPK2_0_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 1) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).cwiseProduct(SPK2_4_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(SPK2_0_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(SPK2_5_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(0, 2) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).cwiseProduct(SPK2_4_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(SPK2_5_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 0) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(SPK2_5_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 1) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(SPK2_5_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(SPK2_1_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(16, 2) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).cwiseProduct(SPK2_3_D0_Block);

    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(SPK2_4_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 0) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).cwiseProduct(SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(SPK2_4_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 1) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).cwiseProduct(SPK2_2_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(SPK2_4_D0_Block) +
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(SPK2_3_D0_Block);
    P_transpose_scaled_Block_col_ordered.block<16, 1>(32, 2) +=
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).cwiseProduct(SPK2_2_D0_Block);

    // =============================================================================

    ChVectorN<double, 4> Ediag_0_Dv_Block = F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2));
    Ediag_0_Dv_Block.array() -= 1;
    Ediag_0_Dv_Block *= 0.5;
    Ediag_0_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_1_Dv_Block = F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2));
    Ediag_1_Dv_Block.array() -= 1;
    Ediag_1_Dv_Block *= 0.5;
    Ediag_1_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_2_Dv_Block = F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2));
    Ediag_2_Dv_Block.array() -= 1;
    Ediag_2_Dv_Block *= 0.5;
    Ediag_2_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    ChVectorN<double, 4> Sdiag_0_Dv_Block =
        Dv(0, 0) * Ediag_0_Dv_Block + Dv(1, 0) * Ediag_1_Dv_Block + Dv(2, 0) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block =
        Dv(0, 1) * Ediag_0_Dv_Block + Dv(1, 1) * Ediag_1_Dv_Block + Dv(2, 1) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block =
        Dv(0, 2) * Ediag_0_Dv_Block + Dv(1, 2) * Ediag_1_Dv_Block + Dv(2, 2) * Ediag_2_Dv_Block;

    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0).cwiseProduct(Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1).cwiseProduct(Sdiag_0_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(48, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2).cwiseProduct(Sdiag_0_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0).cwiseProduct(Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1).cwiseProduct(Sdiag_1_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(52, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2).cwiseProduct(Sdiag_1_Dv_Block);

    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 0) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0).cwiseProduct(Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 1) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1).cwiseProduct(Sdiag_2_Dv_Block);
    P_transpose_scaled_Block_col_ordered.block<4, 1>(56, 2) =
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2).cwiseProduct(Sdiag_2_Dv_Block);

    // =============================================================================

    ChMatrixNM<double, 9, 3> QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

// Calculate the calculate the Jacobian of the internal force integrand with damping included
void ChElementBeamANCF_TR08::ComputeInternalJacobianDamping(ChMatrixRef H,
                                                            double Kfactor,
                                                            double Rfactor,
                                                            double Mfactor) {
    ChMatrixNM<double, 9, 6> ebar_ebardot;
    CalcCombinedCoordMatrix(ebar_ebardot);

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    ChMatrixNMc<double, 60, 6> F_Transpose_CombinedBlockDamping_col_ordered;
    F_Transpose_CombinedBlockDamping_col_ordered.noalias() = m_SD_precompute_col_ordered.transpose() * ebar_ebardot;

    // ChMatrixNM<double, 27, 180> partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 180> partial_epsilon_partial_e_Transpose;
    partial_epsilon_partial_e_Transpose.resize(27, 108);

    for (auto i = 0; i < 9; i++) {
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 0).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 16).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 32).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 48).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 64).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 80).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>(3 * i, 96).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>(3 * i, 100).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>(3 * i, 104).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 0).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 16).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 32).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 48).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 64).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 80).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 1, 96).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 1, 100).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 1, 104).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 0).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 16).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 32).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 48).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 64).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 80).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 2, 96).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 2, 100).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 2, 104).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
                F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).transpose());
    }

    ChMatrixNMc<double, 60, 3> scaled_F_Transpose_col_ordered =
        (Kfactor + m_Alpha * Rfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<60, 3>(0, 0) +
        (m_Alpha * Kfactor) * F_Transpose_CombinedBlockDamping_col_ordered.block<60, 3>(0, 3);

    for (auto i = 0; i < 3; i++) {
        scaled_F_Transpose_col_ordered.block<16, 1>(0, i).array() *= m_GQWeight_det_J_0xi_D0.array();
        scaled_F_Transpose_col_ordered.block<16, 1>(16, i).array() *= m_GQWeight_det_J_0xi_D0.array();
        scaled_F_Transpose_col_ordered.block<16, 1>(32, i).array() *= m_GQWeight_det_J_0xi_D0.array();
        scaled_F_Transpose_col_ordered.block<4, 1>(48, i).array() *= m_GQWeight_det_J_0xi_Dv.array();
        scaled_F_Transpose_col_ordered.block<4, 1>(52, i).array() *= m_GQWeight_det_J_0xi_Dv.array();
        scaled_F_Transpose_col_ordered.block<4, 1>(56, i).array() *= m_GQWeight_det_J_0xi_Dv.array();
    }

    // ChMatrixNM<double, 27, 108> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 180>
        Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.resize(27, 108);

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    for (auto i = 0; i < 9; i++) {
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 0).noalias() =
            D0(0) * m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(0, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 16).noalias() =
            D0(1) * m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(16, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 32).noalias() =
            D0(2) * m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(32, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 48).noalias() =
            D0(3) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 0).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 0).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 64).noalias() =
            D0(4) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 0).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 0).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 80).noalias() =
            D0(5) * (m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 0).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 0).transpose()));

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 0).noalias() =
            D0(0) * m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(0, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 16).noalias() =
            D0(1) * m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(16, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 32).noalias() =
            D0(2) * m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(32, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 48).noalias() =
            D0(3) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 1).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 1).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 64).noalias() =
            D0(4) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 1).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 1).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 80).noalias() =
            D0(5) * (m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 1).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 1).transpose()));

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 0).noalias() =
            D0(0) * m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(0, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 16).noalias() =
            D0(1) * m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(16, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 32).noalias() =
            D0(2) * m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(32, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 48).noalias() =
            D0(3) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 2).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 2).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 64).noalias() =
            D0(4) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 2).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 2).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 80).noalias() =
            D0(5) * (m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 2).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 2).transpose()));
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 4> ChunkSDvA;
    ChunkSDvA.resize(27, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 4> ChunkSDvB;
    ChunkSDvB.resize(27, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 4> ChunkSDvC;
    ChunkSDvC.resize(27, 4);

    for (auto i = 0; i < 9; i++) {
        ChunkSDvA.block<1, 4>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(48, 0).transpose());
        ChunkSDvB.block<1, 4>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(52, 0).transpose());
        ChunkSDvC.block<1, 4>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(56, 0).transpose());

        ChunkSDvA.block<1, 4>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(48, 1).transpose());
        ChunkSDvB.block<1, 4>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(52, 1).transpose());
        ChunkSDvC.block<1, 4>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(56, 1).transpose());

        ChunkSDvA.block<1, 4>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(48, 2).transpose());
        ChunkSDvB.block<1, 4>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(52, 2).transpose());
        ChunkSDvC.block<1, 4>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(56, 2).transpose());
    }

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    Scaled_Combined_partial_epsilon_partial_e_Transpose.block<27, 4>(0, 96).noalias() =
        Dv(0, 0) * ChunkSDvA + Dv(0, 1) * ChunkSDvB + Dv(0, 2) * ChunkSDvC;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.block<27, 4>(0, 100).noalias() =
        Dv(1, 0) * ChunkSDvA + Dv(1, 1) * ChunkSDvB + Dv(1, 2) * ChunkSDvC;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.block<27, 4>(0, 104).noalias() =
        Dv(2, 0) * ChunkSDvA + Dv(2, 1) * ChunkSDvB + Dv(2, 2) * ChunkSDvC;

    H = partial_epsilon_partial_e_Transpose * Scaled_Combined_partial_epsilon_partial_e_Transpose.transpose();

    //===========================================================================================

    ChVectorN<double, 16> SPK2_0_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0));
    SPK2_0_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1));
    SPK2_0_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2));
    SPK2_0_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_0_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3));
    SPK2_0_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4));
    SPK2_0_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5));
    SPK2_0_D0_Block += m_2Alpha * SPK2_0_D0_BlockDamping;
    SPK2_0_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_0_D0_Block *= (0.5 * D0(0));

    ChVectorN<double, 16> SPK2_1_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0));
    SPK2_1_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1));
    SPK2_1_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2));
    SPK2_1_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_1_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3));
    SPK2_1_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4));
    SPK2_1_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5));
    SPK2_1_D0_Block += m_2Alpha * SPK2_1_D0_BlockDamping;
    SPK2_1_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_1_D0_Block *= (0.5 * D0(1));

    ChVectorN<double, 16> SPK2_2_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0));
    SPK2_2_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1));
    SPK2_2_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2));
    SPK2_2_D0_Block.array() -= 1;
    ChVectorN<double, 16> SPK2_2_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3));
    SPK2_2_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4));
    SPK2_2_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5));
    SPK2_2_D0_Block += m_2Alpha * SPK2_2_D0_BlockDamping;
    SPK2_2_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_2_D0_Block *= (0.5 * D0(2));

    ChVectorN<double, 16> SPK2_3_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0));
    SPK2_3_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1));
    SPK2_3_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2));
    ChVectorN<double, 16> SPK2_3_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4));
    SPK2_3_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5));
    SPK2_3_D0_Block += m_Alpha * SPK2_3_D0_BlockDamping;
    SPK2_3_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_3_D0_Block *= D0(3);

    ChVectorN<double, 16> SPK2_4_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0));
    SPK2_4_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1));
    SPK2_4_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2));
    ChVectorN<double, 16> SPK2_4_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 3));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 4));
    SPK2_4_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(32, 5));
    SPK2_4_D0_Block += m_Alpha * SPK2_4_D0_BlockDamping;
    SPK2_4_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_4_D0_Block *= D0(4);

    ChVectorN<double, 16> SPK2_5_D0_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0));
    SPK2_5_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1));
    SPK2_5_D0_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2));
    ChVectorN<double, 16> SPK2_5_D0_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 3));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 4));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 5));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 3));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 4));
    SPK2_5_D0_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<16, 1>(16, 5));
    SPK2_5_D0_Block += m_Alpha * SPK2_5_D0_BlockDamping;
    SPK2_5_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_5_D0_Block *= D0(5);

    ChVectorN<double, 4> Ediag_0_Dv_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2));
    Ediag_0_Dv_Block.array() -= 1;
    Ediag_0_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_0_Dv_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 3));
    Ediag_0_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 4));
    Ediag_0_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(48, 5));
    Ediag_0_Dv_Block += m_Alpha * Ediag_0_Dv_BlockDamping;
    Ediag_0_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_1_Dv_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2));
    Ediag_1_Dv_Block.array() -= 1;
    Ediag_1_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_1_Dv_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 3));
    Ediag_1_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 4));
    Ediag_1_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(52, 5));
    Ediag_1_Dv_Block += m_Alpha * Ediag_1_Dv_BlockDamping;
    Ediag_1_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_2_Dv_Block =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2));
    Ediag_2_Dv_Block.array() -= 1;
    Ediag_2_Dv_Block *= 0.5;
    ChVectorN<double, 4> Ediag_2_Dv_BlockDamping =
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 0).cwiseProduct(
            F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 3));
    Ediag_2_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 1).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 4));
    Ediag_2_Dv_BlockDamping += F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 2).cwiseProduct(
        F_Transpose_CombinedBlockDamping_col_ordered.block<4, 1>(56, 5));
    Ediag_2_Dv_Block += m_Alpha * Ediag_2_Dv_BlockDamping;
    Ediag_2_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Sdiag_0_Dv_Block =
        Dv(0, 0) * Ediag_0_Dv_Block + Dv(1, 0) * Ediag_1_Dv_Block + Dv(2, 0) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block =
        Dv(0, 1) * Ediag_0_Dv_Block + Dv(1, 1) * Ediag_1_Dv_Block + Dv(2, 1) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block =
        Dv(0, 2) * Ediag_0_Dv_Block + Dv(1, 2) * Ediag_1_Dv_Block + Dv(2, 2) * Ediag_2_Dv_Block;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 60> S_scaled_SD_precompute_col_ordered;
    S_scaled_SD_precompute_col_ordered.resize(9, 60);
    // ChMatrixNM<double, 9, 60> S_scaled_SD_precompute_col_ordered;

    for (auto i = 0; i < 9; i++) {
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0).noalias() =
        // SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) +=
        // SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) +=
        // SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16).noalias() =
        // SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) +=
        // SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) +=
        // SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32).noalias() =
        // SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) +=
        // SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) +=
        // SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0).noalias() =
            SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0)) +
            SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16)) +
            SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16).noalias() =
            SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0)) +
            SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16)) +
            SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32).noalias() =
            SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0)) +
            SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16)) +
            SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 48).noalias() =
            Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 48));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 52).noalias() =
            Sdiag_1_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 52));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 56).noalias() =
            Sdiag_2_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 56));
    }

    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 9> Jacobian_CompactPart;
    // Jacobian_CompactPart.resize(9, 9);
    // ChMatrixNM<double, 9, 9> Jacobian_CompactPart = Mfactor * m_MassMatrix + Kfactor * m_SD_precompute_col_ordered *
    // S_scaled_SD_precompute_col_ordered.transpose();
    ChMatrixNM<double, 9, 9> Jacobian_CompactPart =
        m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();
    Jacobian_CompactPart *= Kfactor;
    Jacobian_CompactPart += Mfactor * m_MassMatrix;
    // Jacobian_CompactPart.noalias() = Mfactor * m_MassMatrix + Kfactor * m_SD_precompute_col_ordered *
    // S_scaled_SD_precompute_col_ordered.transpose();

    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) += Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j + 1) += Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 2) += Jacobian_CompactPart(i, j);
        }
    }
}

// Calculate the calculate the Jacobian of the internal force integrand without damping included
void ChElementBeamANCF_TR08::ComputeInternalJacobianNoDamping(ChMatrixRef H, double Kfactor, double Mfactor) {
    ChMatrixNM<double, 9, 3> e_bar;
    CalcCoordMatrix(e_bar);

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    ChMatrixNMc<double, 60, 3> F_Transpose_CombinedBlock_col_ordered;
    F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;

    // ChMatrixNM<double, 27, 180> partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 180> partial_epsilon_partial_e_Transpose;
    partial_epsilon_partial_e_Transpose.resize(27, 108);

    for (auto i = 0; i < 9; i++) {
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 0).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 16).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 32).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 48).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 64).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 80).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>(3 * i, 96).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>(3 * i, 100).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>(3 * i, 104).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 0).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 16).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 32).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 48).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 64).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 80).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 1, 96).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 1, 100).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 1, 104).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 0).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 16).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 32).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 48).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 64).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 80).noalias() =
            m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 2, 96).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 2, 100).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 4>((3 * i) + 2, 104).noalias() =
            m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
                F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2).transpose());
    }

    ChMatrixNMc<double, 60, 3> scaled_F_Transpose_col_ordered =
        Kfactor * F_Transpose_CombinedBlock_col_ordered.block<60, 3>(0, 0);

    for (auto i = 0; i < 3; i++) {
        scaled_F_Transpose_col_ordered.block<16, 1>(0, i).array() *= m_GQWeight_det_J_0xi_D0.array();
        scaled_F_Transpose_col_ordered.block<16, 1>(16, i).array() *= m_GQWeight_det_J_0xi_D0.array();
        scaled_F_Transpose_col_ordered.block<16, 1>(32, i).array() *= m_GQWeight_det_J_0xi_D0.array();
        scaled_F_Transpose_col_ordered.block<4, 1>(48, i).array() *= m_GQWeight_det_J_0xi_Dv.array();
        scaled_F_Transpose_col_ordered.block<4, 1>(52, i).array() *= m_GQWeight_det_J_0xi_Dv.array();
        scaled_F_Transpose_col_ordered.block<4, 1>(56, i).array() *= m_GQWeight_det_J_0xi_Dv.array();
    }

    // ChMatrixNM<double, 27, 108> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 180>
        Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.resize(27, 108);

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    for (auto i = 0; i < 9; i++) {
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 0).noalias() =
            D0(0) * m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(0, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 16).noalias() =
            D0(1) * m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(16, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 32).noalias() =
            D0(2) * m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(32, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 48).noalias() =
            D0(3) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 0).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 0).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 64).noalias() =
            D0(4) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 0).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 0).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>(3 * i, 80).noalias() =
            D0(5) * (m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 0).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 0).transpose()));

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 0).noalias() =
            D0(0) * m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(0, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 16).noalias() =
            D0(1) * m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(16, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 32).noalias() =
            D0(2) * m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(32, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 48).noalias() =
            D0(3) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 1).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 1).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 64).noalias() =
            D0(4) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 1).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 1).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 1, 80).noalias() =
            D0(5) * (m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 1).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 1).transpose()));

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 0).noalias() =
            D0(0) * m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(0, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 16).noalias() =
            D0(1) * m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(16, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 32).noalias() =
            D0(2) * m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                        scaled_F_Transpose_col_ordered.block<16, 1>(32, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 48).noalias() =
            D0(3) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 2).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 2).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 64).noalias() =
            D0(4) * (m_SD_precompute_col_ordered.block<1, 16>(i, 32).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 2).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(32, 2).transpose()));
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 16>((3 * i) + 2, 80).noalias() =
            D0(5) * (m_SD_precompute_col_ordered.block<1, 16>(i, 16).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(0, 2).transpose()) +
                     m_SD_precompute_col_ordered.block<1, 16>(i, 0).cwiseProduct(
                         scaled_F_Transpose_col_ordered.block<16, 1>(16, 2).transpose()));
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 4> ChunkSDvA;
    ChunkSDvA.resize(27, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 4> ChunkSDvB;
    ChunkSDvB.resize(27, 4);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 27, 4> ChunkSDvC;
    ChunkSDvC.resize(27, 4);

    for (auto i = 0; i < 9; i++) {
        ChunkSDvA.block<1, 4>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(48, 0).transpose());
        ChunkSDvB.block<1, 4>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(52, 0).transpose());
        ChunkSDvC.block<1, 4>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(56, 0).transpose());

        ChunkSDvA.block<1, 4>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(48, 1).transpose());
        ChunkSDvB.block<1, 4>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(52, 1).transpose());
        ChunkSDvC.block<1, 4>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(56, 1).transpose());

        ChunkSDvA.block<1, 4>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 48).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(48, 2).transpose());
        ChunkSDvB.block<1, 4>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 52).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(52, 2).transpose());
        ChunkSDvC.block<1, 4>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 4>(i, 56).cwiseProduct(
            scaled_F_Transpose_col_ordered.block<4, 1>(56, 2).transpose());
    }

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    Scaled_Combined_partial_epsilon_partial_e_Transpose.block<27, 4>(0, 96).noalias() =
        Dv(0, 0) * ChunkSDvA + Dv(0, 1) * ChunkSDvB + Dv(0, 2) * ChunkSDvC;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.block<27, 4>(0, 100).noalias() =
        Dv(1, 0) * ChunkSDvA + Dv(1, 1) * ChunkSDvB + Dv(1, 2) * ChunkSDvC;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.block<27, 4>(0, 104).noalias() =
        Dv(2, 0) * ChunkSDvA + Dv(2, 1) * ChunkSDvB + Dv(2, 2) * ChunkSDvC;

    H = partial_epsilon_partial_e_Transpose * Scaled_Combined_partial_epsilon_partial_e_Transpose.transpose();

    //===========================================================================================

    ChVectorN<double, 16> SPK2_0_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0));
    SPK2_0_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1));
    SPK2_0_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2));
    SPK2_0_D0_Block.array() -= 1;
    SPK2_0_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_0_D0_Block *= (0.5 * D0(0));

    ChVectorN<double, 16> SPK2_1_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0));
    SPK2_1_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1));
    SPK2_1_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2));
    SPK2_1_D0_Block.array() -= 1;
    SPK2_1_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_1_D0_Block *= (0.5 * D0(1));

    ChVectorN<double, 16> SPK2_2_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0));
    SPK2_2_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1));
    SPK2_2_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2));
    SPK2_2_D0_Block.array() -= 1;
    SPK2_2_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_2_D0_Block *= (0.5 * D0(2));

    ChVectorN<double, 16> SPK2_3_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0));
    SPK2_3_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1));
    SPK2_3_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2));
    SPK2_3_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_3_D0_Block *= D0(3);

    ChVectorN<double, 16> SPK2_4_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 0));
    SPK2_4_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 1));
    SPK2_4_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(32, 2));
    SPK2_4_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_4_D0_Block *= D0(4);

    ChVectorN<double, 16> SPK2_5_D0_Block = F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 0));
    SPK2_5_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 1));
    SPK2_5_D0_Block += F_Transpose_CombinedBlock_col_ordered.block<16, 1>(0, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<16, 1>(16, 2));
    SPK2_5_D0_Block.array() *= m_GQWeight_det_J_0xi_D0.array();
    SPK2_5_D0_Block *= D0(5);

    ChVectorN<double, 4> Ediag_0_Dv_Block = F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 0));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 1));
    Ediag_0_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(48, 2));
    Ediag_0_Dv_Block.array() -= 1;
    Ediag_0_Dv_Block *= 0.5;
    Ediag_0_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_1_Dv_Block = F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 0));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 1));
    Ediag_1_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(52, 2));
    Ediag_1_Dv_Block.array() -= 1;
    Ediag_1_Dv_Block *= 0.5;
    Ediag_1_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Ediag_2_Dv_Block = F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 0));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 1));
    Ediag_2_Dv_Block += F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2).cwiseProduct(
        F_Transpose_CombinedBlock_col_ordered.block<4, 1>(56, 2));
    Ediag_2_Dv_Block.array() -= 1;
    Ediag_2_Dv_Block *= 0.5;
    Ediag_2_Dv_Block.array() *= m_GQWeight_det_J_0xi_Dv.array();

    ChVectorN<double, 4> Sdiag_0_Dv_Block =
        Dv(0, 0) * Ediag_0_Dv_Block + Dv(1, 0) * Ediag_1_Dv_Block + Dv(2, 0) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block =
        Dv(0, 1) * Ediag_0_Dv_Block + Dv(1, 1) * Ediag_1_Dv_Block + Dv(2, 1) * Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block =
        Dv(0, 2) * Ediag_0_Dv_Block + Dv(1, 2) * Ediag_1_Dv_Block + Dv(2, 2) * Ediag_2_Dv_Block;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 60> S_scaled_SD_precompute_col_ordered;
    S_scaled_SD_precompute_col_ordered.resize(9, 60);
    // ChMatrixNM<double, 9, 60> S_scaled_SD_precompute_col_ordered;

    for (auto i = 0; i < 9; i++) {
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0).noalias() =
        // SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) +=
        // SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0) +=
        // SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16).noalias() =
        // SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) +=
        // SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16) +=
        // SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32).noalias() =
        // SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) +=
        // SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16));
        // S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32) +=
        // SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 0).noalias() =
            SPK2_0_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0)) +
            SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16)) +
            SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 16).noalias() =
            SPK2_5_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0)) +
            SPK2_1_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16)) +
            SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 16>(i, 32).noalias() =
            SPK2_4_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 0)) +
            SPK2_3_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 16)) +
            SPK2_2_D0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 16>(i, 32));

        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 48).noalias() =
            Sdiag_0_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 48));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 52).noalias() =
            Sdiag_1_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 52));
        S_scaled_SD_precompute_col_ordered.block<1, 4>(i, 56).noalias() =
            Sdiag_2_Dv_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 4>(i, 56));
    }

    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 9, 9> Jacobian_CompactPart;
    // Jacobian_CompactPart.resize(9, 9);
    // ChMatrixNM<double, 9, 9> Jacobian_CompactPart = Mfactor * m_MassMatrix + Kfactor * m_SD_precompute_col_ordered *
    // S_scaled_SD_precompute_col_ordered.transpose();
    ChMatrixNM<double, 9, 9> Jacobian_CompactPart =
        m_SD_precompute_col_ordered * S_scaled_SD_precompute_col_ordered.transpose();
    Jacobian_CompactPart *= Kfactor;
    Jacobian_CompactPart += Mfactor * m_MassMatrix;
    // Jacobian_CompactPart.noalias() = Mfactor * m_MassMatrix + Kfactor * m_SD_precompute_col_ordered *
    // S_scaled_SD_precompute_col_ordered.transpose();

    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) += Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j + 1) += Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 2) += Jacobian_CompactPart(i, j);
        }
    }
}

void ChElementBeamANCF_TR08::ComputeInternalJacobians(ChMatrixNM<double, 27, 27>& JacobianMatrix,
                                                      double Kfactor,
                                                      double Rfactor) {
    // The integrated quantity represents the 27x27 Jacobian
    //      Kfactor * [K] + Rfactor * [R]

    ChVectorDynamic<double> FiOrignal(27);
    ChVectorDynamic<double> FiDelta(27);
    // ChMatrixNMc<double, 9, 3> e_bar;
    // ChMatrixNMc<double, 9, 3> e_bar_dot;

    // CalcCoordMatrix(e_bar);
    // CalcCoordDerivMatrix(e_bar_dot);

    double delta = 1e-6;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    // Runs faster if the internal force with or without damping calculations are not combined into the same function
    // using the common calculations with an if statement for the damping in the middle to calculate the different
    // P_transpose_scaled_Block components
    if (m_damping_enabled) {
        ChMatrixNM<double, 9, 6> ebar_ebardot;
        CalcCombinedCoordMatrix(ebar_ebardot);

        ComputeInternalForcesAtState(FiOrignal, ebar_ebardot);
        for (unsigned int i = 0; i < 27; i++) {
            ebar_ebardot(i / 3, i % 3) = ebar_ebardot(i / 3, i % 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar_ebardot);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            ebar_ebardot(i / 3, i % 3) = ebar_ebardot(i / 3, i % 3) - delta;

            ebar_ebardot(i / 3, (i % 3) + 3) = ebar_ebardot(i / 3, (i % 3) + 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar_ebardot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            ebar_ebardot(i / 3, (i % 3) + 3) = ebar_ebardot(i / 3, (i % 3) + 3) - delta;
        }
    } else {
        ChMatrixNMc<double, 9, 3> e_bar;
        CalcCoordMatrix(e_bar);

        ComputeInternalForcesAtStateNoDamping(FiOrignal, e_bar);
        for (unsigned int i = 0; i < 27; i++) {
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) + delta;
            ComputeInternalForcesAtStateNoDamping(FiDelta, e_bar);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) - delta;
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x27 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_TR08::Calc_Sxi(ChMatrixNM<double, 3, 27>& Sxi, double xi, double eta, double zeta) {
    ChVectorN<double, 9> Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    Sxi.setZero();

    //// MIKE Clean-up when slicing becomes available in Eigen 3.4
    for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
        Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
        Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
        Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
    }
}

// 9x1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
void ChElementBeamANCF_TR08::Calc_Sxi_compact(ChVectorN<double, 9>& Sxi_compact, double xi, double eta, double zeta) {
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

// Calculate the 27x3 Compact Shape Function Derivative Matrix modified by the inverse of the element Jacobian
//            [partial(s_1)/(partial xi)      partial(s_2)/(partial xi)     ...]T
// Sxi_D_0xi = [partial(s_1)/(partial eta)     partial(s_2)/(partial eta)    ...]  J_0xi^(-1)
//            [partial(s_1)/(partial zeta)    partial(s_2)/(partial zeta)   ...]
// See: J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and
//      thin three-dimensional beam elements in the absolute nodal coordinate formulation,
//      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.

void ChElementBeamANCF_TR08::Calc_Sxi_D(ChMatrixNMc<double, 9, 3>& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = xi - 0.5;
    Sxi_D(1, 0) = 0.25 * m_thicknessY * eta * (2.0 * xi - 1.0);
    Sxi_D(2, 0) = 0.25 * m_thicknessZ * zeta * (2.0 * xi - 1.0);
    Sxi_D(3, 0) = xi + 0.5;
    Sxi_D(4, 0) = 0.25 * m_thicknessY * eta * (2.0 * xi + 1.0);
    Sxi_D(5, 0) = 0.25 * m_thicknessZ * zeta * (2.0 * xi + 1.0);
    Sxi_D(6, 0) = -2.0 * xi;
    Sxi_D(7, 0) = -m_thicknessY * eta * xi;
    Sxi_D(8, 0) = -m_thicknessZ * zeta * xi;

    Sxi_D(0, 1) = 0.0;
    Sxi_D(1, 1) = 0.25 * m_thicknessY * (xi * xi - xi);
    Sxi_D(2, 1) = 0.0;
    Sxi_D(3, 1) = 0.0;
    Sxi_D(4, 1) = 0.25 * m_thicknessY * (xi * xi + xi);
    Sxi_D(5, 1) = 0.0;
    Sxi_D(6, 1) = 0.0;
    Sxi_D(7, 1) = 0.5 * m_thicknessY * (1 - xi * xi);
    Sxi_D(8, 1) = 0.0;

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.25 * m_thicknessZ * (xi * xi - xi);
    Sxi_D(3, 2) = 0.0;
    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.25 * m_thicknessZ * (xi * xi + xi);
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = 0.0;
    Sxi_D(8, 2) = 0.5 * m_thicknessZ * (1 - xi * xi);
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementBeamANCF_TR08::CalcCoordMatrix(ChMatrixNMc<double, 9, 3>& e) {
    e.row(0) = m_nodes[0]->GetPos().eigen();
    e.row(1) = m_nodes[0]->GetD().eigen();
    e.row(2) = m_nodes[0]->GetDD().eigen();

    e.row(3) = m_nodes[1]->GetPos().eigen();
    e.row(4) = m_nodes[1]->GetD().eigen();
    e.row(5) = m_nodes[1]->GetDD().eigen();

    e.row(6) = m_nodes[2]->GetPos().eigen();
    e.row(7) = m_nodes[2]->GetD().eigen();
    e.row(8) = m_nodes[2]->GetDD().eigen();
}

void ChElementBeamANCF_TR08::CalcCoordMatrix(ChMatrixNM<double, 9, 3>& e) {
    e.block<1, 3>(0, 0) = m_nodes[0]->GetPos().eigen();
    e.block<1, 3>(1, 0) = m_nodes[0]->GetD().eigen();
    e.block<1, 3>(2, 0) = m_nodes[0]->GetDD().eigen();

    e.block<1, 3>(3, 0) = m_nodes[1]->GetPos().eigen();
    e.block<1, 3>(4, 0) = m_nodes[1]->GetD().eigen();
    e.block<1, 3>(5, 0) = m_nodes[1]->GetDD().eigen();

    e.block<1, 3>(6, 0) = m_nodes[2]->GetPos().eigen();
    e.block<1, 3>(7, 0) = m_nodes[2]->GetD().eigen();
    e.block<1, 3>(8, 0) = m_nodes[2]->GetDD().eigen();
}

void ChElementBeamANCF_TR08::CalcCoordVector(ChVectorN<double, 27>& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();

    e.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(12, 3) = m_nodes[1]->GetD().eigen();
    e.segment(15, 3) = m_nodes[1]->GetDD().eigen();

    e.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(21, 3) = m_nodes[2]->GetD().eigen();
    e.segment(24, 3) = m_nodes[2]->GetDD().eigen();
}

void ChElementBeamANCF_TR08::CalcCoordDerivMatrix(ChMatrixNMc<double, 9, 3>& edot) {
    edot.row(0) = m_nodes[0]->GetPos_dt().eigen();
    edot.row(1) = m_nodes[0]->GetD_dt().eigen();
    edot.row(2) = m_nodes[0]->GetDD_dt().eigen();

    edot.row(3) = m_nodes[1]->GetPos_dt().eigen();
    edot.row(4) = m_nodes[1]->GetD_dt().eigen();
    edot.row(5) = m_nodes[1]->GetDD_dt().eigen();

    edot.row(6) = m_nodes[2]->GetPos_dt().eigen();
    edot.row(7) = m_nodes[2]->GetD_dt().eigen();
    edot.row(8) = m_nodes[2]->GetDD_dt().eigen();
}

void ChElementBeamANCF_TR08::CalcCoordDerivVector(ChVectorN<double, 27>& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();

    edot.segment(9, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(12, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetDD_dt().eigen();

    edot.segment(18, 3) = m_nodes[2]->GetPos_dt().eigen();
    edot.segment(21, 3) = m_nodes[2]->GetD_dt().eigen();
    edot.segment(24, 3) = m_nodes[2]->GetDD_dt().eigen();
}

void ChElementBeamANCF_TR08::CalcCombinedCoordMatrix(ChMatrixNM<double, 9, 6>& ebar_ebardot) {
    ebar_ebardot.block<1, 3>(0, 0) = m_nodes[0]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(1, 0) = m_nodes[0]->GetD().eigen();
    ebar_ebardot.block<1, 3>(1, 3) = m_nodes[0]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(2, 0) = m_nodes[0]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(2, 3) = m_nodes[0]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(3, 0) = m_nodes[1]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(3, 3) = m_nodes[1]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(4, 0) = m_nodes[1]->GetD().eigen();
    ebar_ebardot.block<1, 3>(4, 3) = m_nodes[1]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(5, 0) = m_nodes[1]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(5, 3) = m_nodes[1]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(6, 0) = m_nodes[2]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(6, 3) = m_nodes[2]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(7, 0) = m_nodes[2]->GetD().eigen();
    ebar_ebardot.block<1, 3>(7, 3) = m_nodes[2]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(8, 0) = m_nodes[2]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(8, 3) = m_nodes[2]->GetDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementBeamANCF_TR08::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    ChMatrixNMc<double, 9, 3> Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_e0_bar.transpose() * Sxi_D;
}

// Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementBeamANCF_TR08::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChMatrix33<double> J_0xi;

    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi = m_e0_bar.transpose() * Sxi_D;
    return (J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------
// ChVector<> ChElementBeamANCF_TR08::EvaluateBeamSectionStrains() {
//    // Element shape function
//    ShapeVector N;
//    this->ShapeFunctions(N, 0, 0, 0);
//
//    // Determinant of position vector gradient matrix: Initial configuration
//    ShapeVector Nx;
//    ShapeVector Ny;
//    ShapeVector Nz;
//    ChMatrixNM<double, 1, 3> Nx_d0;
//    ChMatrixNM<double, 1, 3> Ny_d0;
//    ChMatrixNM<double, 1, 3> Nz_d0;
//    double detJ0 = this->Calc_detJ0(0, 0, 0, Nx, Ny, Nz, Nx_d0, Ny_d0, Nz_d0);
//
//    // Transformation : Orthogonal transformation (A and J)
//    ChVector<double> G1xG2;  // Cross product of first and second column of
//    double G1dotG1;          // Dot product of first column of position vector gradient
//
//    G1xG2.x() = Nx_d0(0, 1) * Ny_d0(0, 2) - Nx_d0(0, 2) * Ny_d0(0, 1);
//    G1xG2.y() = Nx_d0(0, 2) * Ny_d0(0, 0) - Nx_d0(0, 0) * Ny_d0(0, 2);
//    G1xG2.z() = Nx_d0(0, 0) * Ny_d0(0, 1) - Nx_d0(0, 1) * Ny_d0(0, 0);
//    G1dotG1 = Nx_d0(0, 0) * Nx_d0(0, 0) + Nx_d0(0, 1) * Nx_d0(0, 1) + Nx_d0(0, 2) * Nx_d0(0, 2);
//
//    // Tangent Frame
//    ChVector<double> A1;
//    ChVector<double> A2;
//    ChVector<double> A3;
//    A1.x() = Nx_d0(0, 0);
//    A1.y() = Nx_d0(0, 1);
//    A1.z() = Nx_d0(0, 2);
//    A1 = A1 / sqrt(G1dotG1);
//    A3 = G1xG2.GetNormalized();
//    A2.Cross(A3, A1);
//
//    // Direction for orthotropic material
//    double theta = 0.0;  // Fiber angle
//    ChVector<double> AA1;
//    ChVector<double> AA2;
//    ChVector<double> AA3;
//    AA1 = A1;
//    AA2 = A2;
//    AA3 = A3;
//
//    /// Beta
//    ChMatrixNM<double, 3, 3> j0;
//    ChVector<double> j01;
//    ChVector<double> j02;
//    ChVector<double> j03;
//    // Calculates inverse of rd0 (j0) (position vector gradient: Initial Configuration)
//    j0(0, 0) = Ny_d0(0, 1) * Nz_d0(0, 2) - Nz_d0(0, 1) * Ny_d0(0, 2);
//    j0(0, 1) = Ny_d0(0, 2) * Nz_d0(0, 0) - Ny_d0(0, 0) * Nz_d0(0, 2);
//    j0(0, 2) = Ny_d0(0, 0) * Nz_d0(0, 1) - Nz_d0(0, 0) * Ny_d0(0, 1);
//    j0(1, 0) = Nz_d0(0, 1) * Nx_d0(0, 2) - Nx_d0(0, 1) * Nz_d0(0, 2);
//    j0(1, 1) = Nz_d0(0, 2) * Nx_d0(0, 0) - Nx_d0(0, 2) * Nz_d0(0, 0);
//    j0(1, 2) = Nz_d0(0, 0) * Nx_d0(0, 1) - Nz_d0(0, 1) * Nx_d0(0, 0);
//    j0(2, 0) = Nx_d0(0, 1) * Ny_d0(0, 2) - Ny_d0(0, 1) * Nx_d0(0, 2);
//    j0(2, 1) = Ny_d0(0, 0) * Nx_d0(0, 2) - Nx_d0(0, 0) * Ny_d0(0, 2);
//    j0(2, 2) = Nx_d0(0, 0) * Ny_d0(0, 1) - Ny_d0(0, 0) * Nx_d0(0, 1);
//    j0 /= detJ0;
//
//    j01[0] = j0(0, 0);
//    j02[0] = j0(1, 0);
//    j03[0] = j0(2, 0);
//    j01[1] = j0(0, 1);
//    j02[1] = j0(1, 1);
//    j03[1] = j0(2, 1);
//    j01[2] = j0(0, 2);
//    j02[2] = j0(1, 2);
//    j03[2] = j0(2, 2);
//
//    // Coefficients of contravariant transformation
//    ChVectorN<double, 9> beta;
//    beta(0) = Vdot(AA1, j01);
//    beta(1) = Vdot(AA2, j01);
//    beta(2) = Vdot(AA3, j01);
//    beta(3) = Vdot(AA1, j02);
//    beta(4) = Vdot(AA2, j02);
//    beta(5) = Vdot(AA3, j02);
//    beta(6) = Vdot(AA1, j03);
//    beta(7) = Vdot(AA2, j03);
//    beta(8) = Vdot(AA3, j03);
//
//    ChVectorN<double, 9> ddNx = m_ddT * Nx.transpose();
//    ChVectorN<double, 9> ddNy = m_ddT * Ny.transpose();
//    ChVectorN<double, 9> ddNz = m_ddT * Nz.transpose();
//
//    ChVectorN<double, 9> d0d0Nx = this->m_d0d0T * Nx.transpose();
//    ChVectorN<double, 9> d0d0Ny = this->m_d0d0T * Ny.transpose();
//    ChVectorN<double, 9> d0d0Nz = this->m_d0d0T * Nz.transpose();
//
//    // Strain component
//    ChVectorN<double, 6> strain_til;
//    strain_til(0) = 0.5 * (Nx.dot(ddNx) - Nx.dot(d0d0Nx));
//    strain_til(1) = 0.5 * (Ny.dot(ddNy) - Ny.dot(d0d0Ny));
//    strain_til(2) = Nx.dot(ddNy) - Nx.dot(d0d0Ny);
//    strain_til(3) = 0.5 * (Nz.dot(ddNz) - Nz.dot(d0d0Nz));
//    strain_til(4) = Nx.dot(ddNz) - Nx.dot(d0d0Nz);
//    strain_til(5) = Ny.dot(ddNz) - Ny.dot(d0d0Nz);
//
//    // For orthotropic material
//    ChVectorN<double, 6> strain;
//
//    strain(0) = strain_til(0) * beta(0) * beta(0) + strain_til(1) * beta(3) * beta(3) +
//                strain_til(2) * beta(0) * beta(3) + strain_til(3) * beta(6) * beta(6) +
//                strain_til(4) * beta(0) * beta(6) + strain_til(5) * beta(3) * beta(6);
//    strain(1) = strain_til(0) * beta(1) * beta(1) + strain_til(1) * beta(4) * beta(4) +
//                strain_til(2) * beta(1) * beta(4) + strain_til(3) * beta(7) * beta(7) +
//                strain_til(4) * beta(1) * beta(7) + strain_til(5) * beta(4) * beta(7);
//    strain(2) = strain_til(0) * 2.0 * beta(0) * beta(1) + strain_til(1) * 2.0 * beta(3) * beta(4) +
//                strain_til(2) * (beta(1) * beta(3) + beta(0) * beta(4)) + strain_til(3) * 2.0 * beta(6) * beta(7) +
//                strain_til(4) * (beta(1) * beta(6) + beta(0) * beta(7)) +
//                strain_til(5) * (beta(4) * beta(6) + beta(3) * beta(7));
//    strain(3) = strain_til(0) * beta(2) * beta(2) + strain_til(1) * beta(5) * beta(5) +
//                strain_til(2) * beta(2) * beta(5) + strain_til(3) * beta(8) * beta(8) +
//                strain_til(4) * beta(2) * beta(8) + strain_til(5) * beta(5) * beta(8);
//    strain(4) = strain_til(0) * 2.0 * beta(0) * beta(2) + strain_til(1) * 2.0 * beta(3) * beta(5) +
//                strain_til(2) * (beta(2) * beta(3) + beta(0) * beta(5)) + strain_til(3) * 2.0 * beta(6) * beta(8) +
//                strain_til(4) * (beta(2) * beta(6) + beta(0) * beta(8)) +
//                strain_til(5) * (beta(5) * beta(6) + beta(3) * beta(8));
//    strain(5) = strain_til(0) * 2.0 * beta(1) * beta(2) + strain_til(1) * 2.0 * beta(4) * beta(5) +
//                strain_til(2) * (beta(2) * beta(4) + beta(1) * beta(5)) + strain_til(3) * 2.0 * beta(7) * beta(8) +
//                strain_til(4) * (beta(2) * beta(7) + beta(1) * beta(8)) +
//                strain_til(5) * (beta(5) * beta(7) + beta(4) * beta(8));
//
//    return ChVector<>(strain(0), strain(1), strain(2));
//}
//
// void ChElementBeamANCF_TR08::EvaluateSectionDisplacement(const double u,
//                                                    const double v,
//                                                    ChVector<>& u_displ,
//                                                    ChVector<>& u_rotaz) {
//    // this is not a corotational element, so just do:
//    EvaluateSectionPoint(u, v, displ, u_displ);
//    u_rotaz = VNULL;  // no angles.. this is ANCF (or maybe return here the slope derivatives?)
//}

void ChElementBeamANCF_TR08::EvaluateSectionFrame(const double eta, ChVector<>& point, ChQuaternion<>& rot) {
    ChMatrixNMc<double, 9, 3> e_bar;
    ChVectorN<double, 9> Sxi_compact;
    ChMatrixNMc<double, 9, 3> Sxi_D;

    CalcCoordMatrix(e_bar);
    Calc_Sxi_compact(Sxi_compact, eta, 0, 0);
    Calc_Sxi_D(Sxi_D, eta, 0, 0);

    // r = Se
    point = e_bar.transpose() * Sxi_compact;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector<double> BeamAxisTangent = e_bar.transpose() * Sxi_D.col(0);
    ChVector<double> CrossSectionY = e_bar.transpose() * Sxi_D.col(1);

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33<> msect;
    msect.Set_A_Xdir(BeamAxisTangent, CrossSectionY);

    rot = msect.Get_A_quaternion();
}

// void ChElementBeamANCF_TR08::EvaluateSectionPoint(const double u,
//                                             const double v,
//                                             ChVector<>& point) {
//    ChVector<> u_displ;
//
//    ChMatrixNM<double, 1, 8> N;
//
//    double x = u;  // because ShapeFunctions() works in -1..1 range
//    double y = v;  // because ShapeFunctions() works in -1..1 range
//    double z = 0;
//
//    this->ShapeFunctions(N, x, y, z);
//
//    const ChVector<>& pA = m_nodes[0]->GetPos();
//    const ChVector<>& pB = m_nodes[1]->GetPos();
//    const ChVector<>& pC = m_nodes[2]->GetPos();
//    const ChVector<>& pD = m_nodes[3]->GetPos();
//
//    point.x() = N(0) * pA.x() + N(2) * pB.x() + N(4) * pC.x() + N(6) * pD.x();
//    point.y() = N(0) * pA.y() + N(2) * pB.y() + N(4) * pC.y() + N(6) * pD.y();
//    point.z() = N(0) * pA.z() + N(2) * pB.z() + N(4) * pC.z() + N(6) * pD.z();
//}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).
void ChElementBeamANCF_TR08::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).
void ChElementBeamANCF_TR08::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();

    mD.segment(block_offset + 9, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetDD_dt().eigen();

    mD.segment(block_offset + 18, 3) = m_nodes[2]->GetPos_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[2]->GetD_dt().eigen();
    mD.segment(block_offset + 24, 3) = m_nodes[2]->GetDD_dt().eigen();
}

/// Increment all DOFs using a delta.
void ChElementBeamANCF_TR08::LoadableStateIncrement(const unsigned int off_x,
                                                    ChState& x_new,
                                                    const ChState& x,
                                                    const unsigned int off_v,
                                                    const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 9, x_new, x, off_v + 9, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 18, x_new, x, off_v + 18, Dv);
}

// void ChElementBeamANCF_TR08::EvaluateSectionVelNorm(double U, ChVector<>& Result) {
//    ShapeVector N;
//    ShapeFunctions(N, U, 0, 0);
//    for (unsigned int ii = 0; ii < 3; ii++) {
//        Result += N(ii * 3) * m_nodes[ii]->GetPos_dt();
//        Result += N(ii * 3 + 1) * m_nodes[ii]->GetPos_dt();
//    }
//}

// Get the pointers to the contained ChVariables, appending to the mvars vector.
void ChElementBeamANCF_TR08::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U) coordinates of the centerline.
void ChElementBeamANCF_TR08::ComputeNF(
    const double U,              // parametric coordinate in surface
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    ComputeNF(U, 0, 0, Qi, detJ, F, state_x, state_w);
}

// Evaluate N'*F , where N is the shape function evaluated at (U,V,W) coordinates of the surface.
void ChElementBeamANCF_TR08::ComputeNF(
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
    ChMatrixNM<double, 3, 27> Sxi;
    Calc_Sxi(Sxi, U, 0, 0);
    Qi = Sxi.transpose() * F.segment(0, 3);

    // Compute the generalized force vector for the applied moment
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChMatrixNM<double, 3, 9> Sxi_D_transpose;
    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;
    ChVectorN<double, 9> G_A;
    ChVectorN<double, 9> G_B;
    ChVectorN<double, 9> G_C;
    ChVectorN<double, 3> M_scaled = 0.5 * F.segment(3, 3);

    CalcCoordMatrix(e_bar);
    Calc_Sxi_D(Sxi_D, U, V, W);

    J_Cxi.noalias() = e_bar.transpose() * Sxi_D;
    J_Cxi_Inv = J_Cxi.inverse();

    // Compute the unique pieces that make up the moment projection matrix "G"
    // See: Antonio M Recuero, Javier F Aceituno, Jose L Escalona, and Ahmed A Shabana.
    // A nonlinear approach for modeling rail flexibility using the absolute nodal coordinate
    // formulation. Nonlinear Dynamics, 83(1-2):463-481, 2016.
    Sxi_D_transpose = Sxi_D.transpose();
    G_A = Sxi_D_transpose.row(0) * J_Cxi_Inv(0, 0) + Sxi_D_transpose.row(1) * J_Cxi_Inv(1, 0) +
          Sxi_D_transpose.row(2) * J_Cxi_Inv(2, 0);
    G_B = Sxi_D_transpose.row(0) * J_Cxi_Inv(0, 1) + Sxi_D_transpose.row(1) * J_Cxi_Inv(1, 1) +
          Sxi_D_transpose.row(2) * J_Cxi_Inv(2, 1);
    G_C = Sxi_D_transpose.row(0) * J_Cxi_Inv(0, 2) + Sxi_D_transpose.row(1) * J_Cxi_Inv(1, 2) +
          Sxi_D_transpose.row(2) * J_Cxi_Inv(2, 2);

    // Compute G'M without actually forming the complete matrix "G" (since it has a sparsity pattern to it)
    //// MIKE Clean-up when slicing becomes available in Eigen 3.4
    for (unsigned int i = 0; i < 9; i++) {
        Qi(3 * i) += M_scaled(1) * G_C(i) - M_scaled(2) * G_B(i);
        Qi((3 * i) + 1) += M_scaled(2) * G_A(i) - M_scaled(0) * G_C(i);
        Qi((3 * i) + 2) += M_scaled(0) * G_B(i) - M_scaled(1) * G_A(i);
    }

    // Compute the element Jacobian between the current configuration and the normalized configuration
    // This is different than the element Jacobian between the reference configuration and the normalized
    //  configuration used in the internal force calculations
    detJ = J_Cxi.determinant();
}

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// Calculate average element density (needed for ChLoaderVolumeGravity).
double ChElementBeamANCF_TR08::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at (U) coordinates.
ChVector<> ChElementBeamANCF_TR08::ComputeTangent(const double U) {
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChVector<> r_xi;

    CalcCoordMatrix(e_bar);
    Calc_Sxi_D(Sxi_D, U, 0, 0);
    r_xi = e_bar.transpose() * Sxi_D.col(1);

    return r_xi.GetNormalized();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_TR08(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBeamANCF_TR08::GetStaticGQTables() {
    return &static_tables_TR08;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBeamANCF_TR08 methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBeamANCF_TR08::ChMaterialBeamANCF_TR08(double rho,        // material density
                                                 double E,          // Young's modulus
                                                 double nu,         // Poisson ratio
                                                 const double& k1,  // Shear correction factor along beam local y axis
                                                 const double& k2   // Shear correction factor along beam local z axis
                                                 )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G), k1, k2);
}

// Construct a (possibly) orthotropic material.
ChMaterialBeamANCF_TR08::ChMaterialBeamANCF_TR08(double rho,            // material density
                                                 const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
                                                 const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
                                                 const ChVector<>& G,   // shear moduli (G_xy, G_xz, G_yz)
                                                 const double& k1,  // Shear correction factor along beam local y axis
                                                 const double& k2   // Shear correction factor along beam local z axis
                                                 )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G, k1, k2);
}

// Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the
// Poisson effect
void ChMaterialBeamANCF_TR08::Calc_D0_Dv(const ChVector<>& E,
                                         const ChVector<>& nu,
                                         const ChVector<>& G,
                                         double k1,
                                         double k2) {
    // orthotropic material ref: http://homes.civil.aau.dk/lda/Continuum/material.pdf
    // except position of the shear terms is different to match the original ANCF reference paper

    double nu_12 = nu.x();
    double nu_13 = nu.y();
    double nu_23 = nu.z();
    double nu_21 = nu_12 * E.y() / E.x();
    double nu_31 = nu_13 * E.z() / E.x();
    double nu_32 = nu_23 * E.z() / E.y();
    double k = 1.0 - nu_23 * nu_32 - nu_12 * nu_21 - nu_13 * nu_31 - nu_12 * nu_23 * nu_31 - nu_21 * nu_32 * nu_13;

    // Component of Stiffness Tensor that does not contain the Poisson Effect
    m_D0(0) = E.x();
    m_D0(1) = E.y();
    m_D0(2) = E.z();
    m_D0(3) = G.z();
    m_D0(4) = G.y() * k1;
    m_D0(5) = G.x() * k2;

    // Remaining components of the Stiffness Tensor that contain the Poisson Effect
    m_Dv(0, 0) = E.x() * (1 - nu_23 * nu_32) / k - m_D0(0);
    m_Dv(1, 0) = E.y() * (nu_13 * nu_32 + nu_12) / k;
    m_Dv(2, 0) = E.z() * (nu_12 * nu_23 + nu_13) / k;

    m_Dv(0, 1) = E.x() * (nu_23 * nu_31 + nu_21) / k;
    m_Dv(1, 1) = E.y() * (1 - nu_13 * nu_31) / k - m_D0(1);
    m_Dv(2, 1) = E.z() * (nu_13 * nu_21 + nu_23) / k;

    m_Dv(0, 2) = E.x() * (nu_21 * nu_32 + nu_31) / k;
    m_Dv(1, 2) = E.y() * (nu_12 * nu_31 + nu_32) / k;
    m_Dv(2, 2) = E.z() * (1 - nu_12 * nu_21) / k - m_D0(2);
}

}  // end of namespace fea
}  // end of namespace chrono
