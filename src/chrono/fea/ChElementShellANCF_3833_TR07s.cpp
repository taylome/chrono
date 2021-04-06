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
// Higher oreder ANCF shell element with 8 nodes. Description of this element (3833)
// and its internal forces may be found in: Henrik Ebel, Marko K Matikainen, 
// Vesa-Ville Hurskainen, and Aki Mikkola. Analysis of high-order quadrilateral
// plate elements based on the absolute nodal coordinate formulation for three -
// dimensional elasticity.Advances in Mechanical Engineering, 9(6) : 1687814017705069, 2017.
// =============================================================================
// Internal Force Calculation Method is based on:  Gerstmayr, J., Shabana, A.A.:
// Efficient integration of the elastic forces and thin three-dimensional beam
// elements in the absolute nodal coordinate formulation.In: Proceedings of the
// Multibody Dynamics Eccomas thematic Conference, Madrid(2005)
// =============================================================================
// TR07S = a Gerstmayr style implementation of the element with pre-calculation
//     of the terms needed for the generalized internal force calculation with
//     an analytical Jacobian that is integrated across all GQ points at once
//
//  Mass Matrix = Constant, pre-calculated 24x24 matrix
//
//  Generalized Force due to gravity = Constant 72x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Gerstmayr method:
//     Dense Math: e_bar = 3x24 and S_bar = 24x1
//     Math is based on the method presented by Gerstmayr and Shabana
//     1 less than "Full Integration" Number of GQ Integration Points (6x6x2)
//     GQ integration is performed across all the GQ points at once
//     Pre-calculation of terms for the generalized internal force calculation
//
//  Jacobian of the Generalized Internal Force Vector = Analytical Jacobian that
//     is integrated across all GQ points at once
//     F and Strains are cached from the internal force calculation for reuse
//     during the Jacobian calculation
//
// =============================================================================
// Jacobian Symmetries Update & Upper Triangular Mass Matrix
// =============================================================================

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementShellANCF_3833_TR07S.h"
#include <cmath>

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementShellANCF_3833_TR07S::ChElementShellANCF_3833_TR07S()
    : m_gravity_on(false), m_lenX(0), m_lenY(0), m_thicknessZ(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(8);

    m_SD_precompute_col_ordered.resize(24, 192);
    m_F_Transpose_CombinedBlock_col_ordered.resize(192, 3);
    m_F_Transpose_CombinedBlockDamping_col_ordered.resize(192, 6);

    m_F_Transpose_CombinedBlock_col_ordered.setZero();
    m_F_Transpose_CombinedBlockDamping_col_ordered.setZero();
    m_SPK2_0_Block.setZero();
    m_SPK2_1_Block.setZero();
    m_SPK2_2_Block.setZero();
    m_SPK2_3_Block.setZero();
    m_SPK2_4_Block.setZero();
    m_SPK2_5_Block.setZero();
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementShellANCF_3833_TR07S::SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeB,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeC,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeD,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeE,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeF,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeG,
                                           std::shared_ptr<ChNodeFEAxyzDD> nodeH) {
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
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[2]->Variables());
    mvars.push_back(&m_nodes[2]->Variables_D());
    mvars.push_back(&m_nodes[2]->Variables_DD());
    mvars.push_back(&m_nodes[3]->Variables());
    mvars.push_back(&m_nodes[3]->Variables_D());
    mvars.push_back(&m_nodes[3]->Variables_DD());
    mvars.push_back(&m_nodes[4]->Variables());
    mvars.push_back(&m_nodes[4]->Variables_D());
    mvars.push_back(&m_nodes[4]->Variables_DD());
    mvars.push_back(&m_nodes[5]->Variables());
    mvars.push_back(&m_nodes[5]->Variables_D());
    mvars.push_back(&m_nodes[5]->Variables_DD());
    mvars.push_back(&m_nodes[6]->Variables());
    mvars.push_back(&m_nodes[6]->Variables_D());
    mvars.push_back(&m_nodes[6]->Variables_DD());
    mvars.push_back(&m_nodes[7]->Variables());
    mvars.push_back(&m_nodes[7]->Variables_D());
    mvars.push_back(&m_nodes[7]->Variables_DD());
	
    Kmatr.SetVariables(mvars);

    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementShellANCF_3833_TR07S::SetupInitial(ChSystem* system) {
    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementShellANCF_3833_TR07S::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementShellANCF_3833_TR07S::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(12, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    mD.segment(21, 3) = m_nodes[2]->GetD().eigen();
    mD.segment(24, 3) = m_nodes[2]->GetDD().eigen();
    mD.segment(27, 3) = m_nodes[3]->GetPos().eigen();
    mD.segment(30, 3) = m_nodes[3]->GetD().eigen();
    mD.segment(33, 3) = m_nodes[3]->GetDD().eigen();
    mD.segment(36, 3) = m_nodes[4]->GetPos().eigen();
    mD.segment(39, 3) = m_nodes[4]->GetD().eigen();
    mD.segment(42, 3) = m_nodes[4]->GetDD().eigen();
    mD.segment(45, 3) = m_nodes[5]->GetPos().eigen();
    mD.segment(48, 3) = m_nodes[5]->GetD().eigen();
    mD.segment(51, 3) = m_nodes[5]->GetDD().eigen();
    mD.segment(54, 3) = m_nodes[6]->GetPos().eigen();
    mD.segment(57, 3) = m_nodes[6]->GetD().eigen();
    mD.segment(60, 3) = m_nodes[6]->GetDD().eigen();
    mD.segment(63, 3) = m_nodes[7]->GetPos().eigen();
    mD.segment(66, 3) = m_nodes[7]->GetD().eigen();
    mD.segment(69, 3) = m_nodes[7]->GetDD().eigen();
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
void ChElementShellANCF_3833_TR07S::ComputeKRMmatricesGlobal(ChMatrixRef H,
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
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = i; j < 24; j++) {
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
void ChElementShellANCF_3833_TR07S::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Inflate the Mass Matrix since it is stored in compact form.
    // In MATLAB notation:
    // M(1:3:end,1:3:end) = m_MassMatrix;
    // M(2:3:end,2:3:end) = m_MassMatrix;
    // M(3:3:end,3:3:end) = m_MassMatrix;

    //Mass Matrix is Stored in Compact Upper Triangular Form
    //Expand it out into its Full Sparse Symmetric Form
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = i; j < 24; j++) {
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
void ChElementShellANCF_3833_TR07S::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, 5 GQ Points are needed in the xi, eta, and zeta directions
    //  for exact integration of the element's mass matrix, even if
    //  the reference configuration is not straight
	//Mass Matrix Integrand is of order: 9 in xi, order: 9 in eta, and order: 9 in zeta.
    // Since the major pieces of the generalized force due to gravity
    //  can also be used to calculate the mass matrix, these calculations
    //  are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = 4;        // 5 Point Gauss-Quadrature;

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MatrixNxN MassMatrixCompactSquare;
    MassMatrixCompactSquare.setZero();
    m_GravForce.setZero();

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_xi_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta_zeta][it_xi] * GQTable->Weight[GQ_idx_xi_eta_zeta][it_eta] *
                                   GQTable->Weight[GQ_idx_xi_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_xi_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta);  // determinate of the element Jacobian (volume ratio)
                Matrix3x3N Sxi;                                    // 3x72 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                VectorN Sxi_compact;  // 16x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight * rho * det_J_0xi) * Sxi.transpose() * g_acc.eigen();
                MassMatrixCompactSquare += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }

    //Store just the unique entries in the Mass Matrix in Compact Upper Triangular Form
    //since the full Mass Matrix is both sparse and symmetric
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = i; j < 24; j++) {
            m_MassMatrix(idx) = MassMatrixCompactSquare(i, j);
            idx++;
        }
    }
}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementShellANCF_3833_TR07S::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(3) + m_MassMatrix(6) + m_MassMatrix(9) + m_MassMatrix(12) + m_MassMatrix(15) + m_MassMatrix(18) + m_MassMatrix(21);
    m_nodes[1]->m_TotalMass += m_MassMatrix(3) + m_MassMatrix(69) + m_MassMatrix(72) + m_MassMatrix(75) + m_MassMatrix(78) + m_MassMatrix(81) + m_MassMatrix(84) + m_MassMatrix(87);
    m_nodes[2]->m_TotalMass += m_MassMatrix(6) + m_MassMatrix(72) + m_MassMatrix(129) + m_MassMatrix(132) + m_MassMatrix(135) + m_MassMatrix(138) + m_MassMatrix(141) + m_MassMatrix(144);
    m_nodes[3]->m_TotalMass += m_MassMatrix(9) + m_MassMatrix(75) + m_MassMatrix(132) + m_MassMatrix(180) + m_MassMatrix(183) + m_MassMatrix(186) + m_MassMatrix(189) + m_MassMatrix(192);
    m_nodes[4]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(78) + m_MassMatrix(135) + m_MassMatrix(183) + m_MassMatrix(222) + m_MassMatrix(225) + m_MassMatrix(228) + m_MassMatrix(231);
    m_nodes[5]->m_TotalMass += m_MassMatrix(15) + m_MassMatrix(81) + m_MassMatrix(138) + m_MassMatrix(186) + m_MassMatrix(225) + m_MassMatrix(255) + m_MassMatrix(258) + m_MassMatrix(261);
    m_nodes[6]->m_TotalMass += m_MassMatrix(18) + m_MassMatrix(84) + m_MassMatrix(141) + m_MassMatrix(189) + m_MassMatrix(228) + m_MassMatrix(258) + m_MassMatrix(279) + m_MassMatrix(282);
    m_nodes[7]->m_TotalMass += m_MassMatrix(21) + m_MassMatrix(87) + m_MassMatrix(144) + m_MassMatrix(192) + m_MassMatrix(231) + m_MassMatrix(261) + m_MassMatrix(282) + m_MassMatrix(294);
}

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementShellANCF_3833_TR07S::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta_zeta = 3;        // 4 Point Gauss-Quadrature;

    ChMatrixNM<double, 24, 3> SD_precompute_D;

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
                m_SD_precompute_col_ordered.col(index + 64) = SD_precompute_D.col(1);
                m_SD_precompute_col_ordered.col(index + 128) = SD_precompute_D.col(2);

                m_GQWeight_det_J_0xi_D(index) = -J_0xi.determinant() * GQ_weight;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementShellANCF_3833_TR07S::SetAlphaDamp(double a) {
    m_Alpha = a;
    m_2Alpha = 2 * a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementShellANCF_3833_TR07S::ComputeInternalForces(ChVectorDynamic<>& Fi) {
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

void ChElementShellANCF_3833_TR07S::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi, const MatrixNx6& ebar_ebardot) {
    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 8 in eta, and order : 8 in zeta.
    // This requires GQ 5 points along the xi direction and 5 points along the eta and zeta directions for "Full
    // Integration" However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in
    // roughly 1/3 of the calculations

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    m_F_Transpose_CombinedBlockDamping_col_ordered.noalias() = m_SD_precompute_col_ordered.transpose() * ebar_ebardot;

    VectorNIP  E0_Block = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0));
    E0_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1));
    E0_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2));
    E0_Block.array() -= 1;
    E0_Block *= 0.5;
    VectorNIP E_BlockDamping = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 5));
    E0_Block += m_Alpha * E_BlockDamping;
    E0_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP  E1_Block = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0));
    E1_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1));
    E1_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2));
    E1_Block.array() -= 1;
    E1_Block *= 0.5;
    E_BlockDamping.noalias() = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 5));
    E1_Block += m_Alpha * E_BlockDamping;
    E1_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E2_Block = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0));
    E2_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1));
    E2_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2));
    E2_Block.array() -= 1;
    E2_Block *= 0.5;
    E_BlockDamping.noalias() = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 5));
    E2_Block += m_Alpha * E_BlockDamping;
    E2_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP  E3_Block = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0));
    E3_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1));
    E3_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2));
    E_BlockDamping.noalias() = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 5));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 5));
    E3_Block += m_Alpha * E_BlockDamping;
    E3_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP  E4_Block = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0));
    E4_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1));
    E4_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2));
    E_BlockDamping.noalias() = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 5));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 5));
    E4_Block += m_Alpha * E_BlockDamping;
    E4_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP  E5_Block = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0));
    E5_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1));
    E5_Block += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2));
    E_BlockDamping.noalias() = m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 5));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 3));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 4));
    E_BlockDamping += m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 5));
    E5_Block += m_Alpha * E_BlockDamping;
    E5_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();
    m_SPK2_0_Block = D(0, 0)*E0_Block + D(0, 1)*E1_Block + D(0, 2)*E2_Block + D(0, 3)*E3_Block + D(0, 4)*E4_Block + D(0, 5)*E5_Block;
    m_SPK2_1_Block = D(1, 0)*E0_Block + D(1, 1)*E1_Block + D(1, 2)*E2_Block + D(1, 3)*E3_Block + D(1, 4)*E4_Block + D(1, 5)*E5_Block;
    m_SPK2_2_Block = D(2, 0)*E0_Block + D(2, 1)*E1_Block + D(2, 2)*E2_Block + D(2, 3)*E3_Block + D(2, 4)*E4_Block + D(2, 5)*E5_Block;
    m_SPK2_3_Block = D(3, 0)*E0_Block + D(3, 1)*E1_Block + D(3, 2)*E2_Block + D(3, 3)*E3_Block + D(3, 4)*E4_Block + D(3, 5)*E5_Block;
    m_SPK2_4_Block = D(4, 0)*E0_Block + D(4, 1)*E1_Block + D(4, 2)*E2_Block + D(4, 3)*E3_Block + D(4, 4)*E4_Block + D(4, 5)*E5_Block;
    m_SPK2_5_Block = D(5, 0)*E0_Block + D(5, 1)*E1_Block + D(5, 2)*E2_Block + D(5, 3)*E3_Block + D(5, 4)*E4_Block + D(5, 5)*E5_Block;

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> P_transpose_scaled_Block_col_ordered;
    //P_transpose_scaled_Block_col_ordered.resize(192, 3);
    ChMatrixNMc<double, 192, 3>
        P_transpose_scaled_Block_col_ordered;  // 1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled
                                               // across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 0) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(m_SPK2_0_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(m_SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 0) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(m_SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 1) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(m_SPK2_0_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(m_SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 1) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(m_SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 2) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(m_SPK2_0_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(m_SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 2) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(m_SPK2_4_Block);

    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 0) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(m_SPK2_5_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(m_SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 0) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 1) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(m_SPK2_5_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(m_SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 1) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 2) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(m_SPK2_5_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(m_SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 2) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(m_SPK2_3_Block);

    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 0) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).cwiseProduct(m_SPK2_4_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 0) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).cwiseProduct(m_SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 1) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).cwiseProduct(m_SPK2_4_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 1) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).cwiseProduct(m_SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 2) =
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).cwiseProduct(m_SPK2_4_Block) +
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 2) +=
        m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).cwiseProduct(m_SPK2_2_Block);

    // =============================================================================

    MatrixNx3 QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementShellANCF_3833_TR07S::ComputeInternalForcesAtStateNoDamping(ChVectorDynamic<>& Fi,
    const MatrixNx3& e_bar) {
    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full
    // Integration" However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in
    // roughly 1/3 of the calculations

    // Calculate F is one big block and then split up afterwards to improve efficiency (hopefully)
    m_F_Transpose_CombinedBlock_col_ordered = m_SD_precompute_col_ordered.transpose() * e_bar;

    VectorNIP E0_Block = m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0));
    E0_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1));
    E0_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2));
    E0_Block.array() -= 1;
    E0_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E0_Block *= 0.5;

    VectorNIP E1_Block = m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0));
    E1_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1));
    E1_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2));
    E1_Block.array() -= 1;
    E1_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E1_Block *= 0.5;

    VectorNIP E2_Block = m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0));
    E2_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1));
    E2_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2));
    E2_Block.array() -= 1;
    E2_Block.array() *= m_GQWeight_det_J_0xi_D.array();
    E2_Block *= 0.5;

    VectorNIP E3_Block = m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0));
    E3_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1));
    E3_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2));
    E3_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E4_Block = m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0));
    E4_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1));
    E4_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2));
    E4_Block.array() *= m_GQWeight_det_J_0xi_D.array();

    VectorNIP E5_Block = m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0));
    E5_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1));
    E5_Block += m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).cwiseProduct(
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2));
    E5_Block.array() *= m_GQWeight_det_J_0xi_D.array();



    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();
    m_SPK2_0_Block = D(0, 0)*E0_Block + D(0, 1)*E1_Block + D(0, 2)*E2_Block + D(0, 3)*E3_Block + D(0, 4)*E4_Block + D(0, 5)*E5_Block;
    m_SPK2_1_Block = D(1, 0)*E0_Block + D(1, 1)*E1_Block + D(1, 2)*E2_Block + D(1, 3)*E3_Block + D(1, 4)*E4_Block + D(1, 5)*E5_Block;
    m_SPK2_2_Block = D(2, 0)*E0_Block + D(2, 1)*E1_Block + D(2, 2)*E2_Block + D(2, 3)*E3_Block + D(2, 4)*E4_Block + D(2, 5)*E5_Block;
    m_SPK2_3_Block = D(3, 0)*E0_Block + D(3, 1)*E1_Block + D(3, 2)*E2_Block + D(3, 3)*E3_Block + D(3, 4)*E4_Block + D(3, 5)*E5_Block;
    m_SPK2_4_Block = D(4, 0)*E0_Block + D(4, 1)*E1_Block + D(4, 2)*E2_Block + D(4, 3)*E3_Block + D(4, 4)*E4_Block + D(4, 5)*E5_Block;
    m_SPK2_5_Block = D(5, 0)*E0_Block + D(5, 1)*E1_Block + D(5, 2)*E2_Block + D(5, 3)*E3_Block + D(5, 4)*E4_Block + D(5, 5)*E5_Block;

    ChMatrixNMc<double, 192, 3>
        P_transpose_scaled_Block_col_ordered;  // 1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled
                                               // across all Gauss-Quadrature points in column order in a big matrix

    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 0) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).cwiseProduct(m_SPK2_0_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).cwiseProduct(m_SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 0) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).cwiseProduct(m_SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 1) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).cwiseProduct(m_SPK2_0_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).cwiseProduct(m_SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 1) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).cwiseProduct(m_SPK2_4_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 2) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).cwiseProduct(m_SPK2_0_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).cwiseProduct(m_SPK2_5_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(0, 2) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).cwiseProduct(m_SPK2_4_Block);

    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 0) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).cwiseProduct(m_SPK2_5_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).cwiseProduct(m_SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 0) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 1) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).cwiseProduct(m_SPK2_5_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).cwiseProduct(m_SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 1) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 2) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).cwiseProduct(m_SPK2_5_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).cwiseProduct(m_SPK2_1_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(64, 2) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).cwiseProduct(m_SPK2_3_Block);

    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 0) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).cwiseProduct(m_SPK2_4_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 0) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).cwiseProduct(m_SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 1) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).cwiseProduct(m_SPK2_4_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 1) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).cwiseProduct(m_SPK2_2_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 2) =
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).cwiseProduct(m_SPK2_4_Block) +
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).cwiseProduct(m_SPK2_3_Block);
    P_transpose_scaled_Block_col_ordered.block<64, 1>(128, 2) +=
        m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).cwiseProduct(m_SPK2_2_Block);

    // =============================================================================

    MatrixNx3 QiCompact = m_SD_precompute_col_ordered * P_transpose_scaled_Block_col_ordered;
    Eigen::Map<Vector3N> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementShellANCF_3833_TR07S::ComputeInternalJacobians(Matrix3Nx3N& JacobianMatrix,
                                                           double Kfactor,
                                                           double Rfactor) {
    // The integrated quantity represents the 72x72 Jacobian
    //      Kfactor * [K] + Rfactor * [R]

    ChVectorDynamic<double> FiOrignal(72);
    ChVectorDynamic<double> FiDelta(72);

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
        for (unsigned int i = 0; i < 72; i++) {
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
        for (unsigned int i = 0; i < 72; i++) {
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) + delta;
            ComputeInternalForcesAtStateNoDamping(FiDelta, e_bar);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) - delta;
        }
    }
}


void ChElementShellANCF_3833_TR07S::ComputeInternalJacobianDamping(ChMatrixRef& H,
    double Kfactor,
    double Rfactor,
    double Mfactor) {
    // ChMatrixNM<double, 48, 180> partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> partial_epsilon_partial_e_Transpose;
    partial_epsilon_partial_e_Transpose.resize(72, 384);

    for (auto i = 0; i < 24; i++) {
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 0).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 1).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(128, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlockDamping_col_ordered.block<64, 1>(64, 2).transpose());
    }

    //ChMatrixNMc<double, 192, 3> scaled_F_Transpose_col_ordered =
    //    (Kfactor + m_Alpha * Rfactor) * m_F_Transpose_CombinedBlockDamping_col_ordered.block<192, 3>(0, 0) +
    //    (m_Alpha * Kfactor) * m_F_Transpose_CombinedBlockDamping_col_ordered.block<192, 3>(0, 3);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> scaled_F_Transpose_col_ordered;
    //scaled_F_Transpose_col_ordered.resize(192, 3);
    ChMatrixNMc<double, 192, 3> scaled_F_Transpose_col_ordered;
    scaled_F_Transpose_col_ordered = (Kfactor + m_Alpha * Rfactor) * m_F_Transpose_CombinedBlockDamping_col_ordered.block<192, 3>(0, 0) +
        (m_Alpha * Kfactor) * m_F_Transpose_CombinedBlockDamping_col_ordered.block<192, 3>(0, 3);

    for (auto i = 0; i < 3; i++) {
        scaled_F_Transpose_col_ordered.block<64, 1>(0, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<64, 1>(64, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<64, 1>(128, i).array() *= m_GQWeight_det_J_0xi_D.array();
    }

    // ChMatrixNM<double, 48, 432> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.resize(72, 384);


    for (auto i = 0; i < 24; i++) {
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 0).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 1).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 2).transpose());
    }

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    // ChMatrixNM<double, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    DScaled_Combined_partial_epsilon_partial_e_Transpose.resize(72, 384);

    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) = D(0, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(0, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(0, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(0, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(0, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(0, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) = D(1, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(1, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(1, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(1, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(1, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(1, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) = D(2, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(2, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(2, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(2, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(2, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(2, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) = D(3, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(3, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(3, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(3, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(3, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(3, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) = D(4, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(4, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(4, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(4, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(4, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(4, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320) = D(5, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(5, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(5, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(5, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(5, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(5, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);

    H = partial_epsilon_partial_e_Transpose * DScaled_Combined_partial_epsilon_partial_e_Transpose.transpose();

    //===========================================================================================

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_scaled_SD_precompute_col_ordered;
    //S_scaled_SD_precompute_col_ordered.resize(24, 192);
    ChMatrixNM<double, 24, 192> S_scaled_SD_precompute_col_ordered;

    for (auto i = 0; i < 24; i++) {
        S_scaled_SD_precompute_col_ordered.block<1, 64>(i, 0).noalias() =
            m_SPK2_0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 0)) +
            m_SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 64)) +
            m_SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 128));

        S_scaled_SD_precompute_col_ordered.block<1, 64>(i, 64).noalias() =
            m_SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 0)) +
            m_SPK2_1_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 64)) +
            m_SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 128));

        S_scaled_SD_precompute_col_ordered.block<1, 64>(i, 128).noalias() =
            m_SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 0)) +
            m_SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 64)) +
            m_SPK2_2_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 128));
    }

    ChVectorN<double, 300> ScaledMassMatrix = Mfactor * m_MassMatrix;

    //Calculate Just the Non-sparse Upper Triangular Entires and then expand out to full size
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = i; j < 24; j++) {
            double d = Kfactor * m_SD_precompute_col_ordered.block<1, 192>(i, 0) * S_scaled_SD_precompute_col_ordered.block<1, 192>(j, 0).transpose();
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

void ChElementShellANCF_3833_TR07S::ComputeInternalJacobianNoDamping(ChMatrixRef& H, double Kfactor, double Mfactor) {
    // ChMatrixNM<double, 48, 180> partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> partial_epsilon_partial_e_Transpose;
    partial_epsilon_partial_e_Transpose.resize(72, 384);

    for (auto i = 0; i < 24; i++) {
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 0).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 0).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 1).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 1).transpose());

        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(128, 2).transpose());
        partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(m_F_Transpose_CombinedBlock_col_ordered.block<64, 1>(64, 2).transpose());
    }

    //ChMatrixNMc<double, 192, 3> scaled_F_Transpose_col_ordered =
    //    (Kfactor + m_Alpha * Rfactor) * m_F_Transpose_CombinedBlockDamping_col_ordered.block<192, 3>(0, 0) +
    //    (m_Alpha * Kfactor) * m_F_Transpose_CombinedBlockDamping_col_ordered.block<192, 3>(0, 3);
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 192, 3> scaled_F_Transpose_col_ordered;
    //scaled_F_Transpose_col_ordered.resize(192, 3);
    ChMatrixNMc<double, 192, 3> scaled_F_Transpose_col_ordered;
    scaled_F_Transpose_col_ordered = Kfactor * m_F_Transpose_CombinedBlock_col_ordered;

    for (auto i = 0; i < 3; i++) {
        scaled_F_Transpose_col_ordered.block<64, 1>(0, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<64, 1>(64, i).array() *= m_GQWeight_det_J_0xi_D.array();
        scaled_F_Transpose_col_ordered.block<64, 1>(128, i).array() *= m_GQWeight_det_J_0xi_D.array();
    }

    // ChMatrixNM<double, 48, 432> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 72, 384> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Scaled_Combined_partial_epsilon_partial_e_Transpose;
    Scaled_Combined_partial_epsilon_partial_e_Transpose.resize(72, 384);

    for (auto i = 0; i < 24; i++) {
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 0).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>(3 * i, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 0).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 0).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 1).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 1, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 1).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 1).transpose());

        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 0).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 64).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 128).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 192).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 256).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 128).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(128, 2).transpose());
        Scaled_Combined_partial_epsilon_partial_e_Transpose.block<1, 64>((3 * i) + 2, 320).noalias() = m_SD_precompute_col_ordered.block<1, 64>(i, 64).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(0, 2).transpose()) +
            m_SD_precompute_col_ordered.block<1, 64>(i, 0).cwiseProduct(scaled_F_Transpose_col_ordered.block<64, 1>(64, 2).transpose());
    }

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    // ChMatrixNM<double, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 48, 432> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DScaled_Combined_partial_epsilon_partial_e_Transpose;
    DScaled_Combined_partial_epsilon_partial_e_Transpose.resize(72, 384);

    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) = D(0, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(0, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(0, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(0, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(0, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(0, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) = D(1, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(1, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(1, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(1, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(1, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(1, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) = D(2, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(2, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(2, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(2, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(2, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(2, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) = D(3, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(3, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(3, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(3, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(3, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(3, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) = D(4, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(4, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(4, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(4, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(4, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(4, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);
    DScaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320) = D(5, 0)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 0) + D(5, 1)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 64) + D(5, 2)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 128) + D(5, 3)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 192) + D(5, 4)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 256) + D(5, 5)*Scaled_Combined_partial_epsilon_partial_e_Transpose.block<72, 64>(0, 320);

    H = partial_epsilon_partial_e_Transpose * DScaled_Combined_partial_epsilon_partial_e_Transpose.transpose();

    //===========================================================================================

    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> S_scaled_SD_precompute_col_ordered;
    //S_scaled_SD_precompute_col_ordered.resize(24, 192);
    ChMatrixNM<double, 24, 192> S_scaled_SD_precompute_col_ordered;

    for (auto i = 0; i < 24; i++) {
        S_scaled_SD_precompute_col_ordered.block<1, 64>(i, 0).noalias() =
            m_SPK2_0_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 0)) +
            m_SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 64)) +
            m_SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 128));

        S_scaled_SD_precompute_col_ordered.block<1, 64>(i, 64).noalias() =
            m_SPK2_5_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 0)) +
            m_SPK2_1_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 64)) +
            m_SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 128));

        S_scaled_SD_precompute_col_ordered.block<1, 64>(i, 128).noalias() =
            m_SPK2_4_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 0)) +
            m_SPK2_3_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 64)) +
            m_SPK2_2_Block.transpose().cwiseProduct(m_SD_precompute_col_ordered.block<1, 64>(i, 128));
    }

    ChVectorN<double, 300> ScaledMassMatrix = Mfactor * m_MassMatrix;

    //Calculate Just the Non-sparse Upper Triangular Entires and then expand out to full size
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 24; i++) {
        for (unsigned int j = i; j < 24; j++) {
            double d = Kfactor * m_SD_precompute_col_ordered.block<1, 192>(i, 0) * S_scaled_SD_precompute_col_ordered.block<1, 192>(j, 0).transpose();
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

// 3x72 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi(Matrix3x3N& Sxi, double xi, double eta, double zeta) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    Sxi.setZero();

    for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
        Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
        Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
        Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
    }
}

// 24x1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {    
    Sxi_compact(0) = -0.25*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_compact(1) = -0.125*(m_thicknessZ)*(zeta)*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_compact(2) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_compact(3) = 0.25*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_compact(4) = 0.125*(m_thicknessZ)*(zeta)*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_compact(5) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_compact(6) = 0.25*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_compact(7) = 0.125*(m_thicknessZ)*(zeta)*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_compact(8) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_compact(9) = -0.25*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_compact(10) = -0.125*(m_thicknessZ)*(zeta)*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_compact(11) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_compact(12) = 0.5*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_compact(13) = 0.25*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_compact(14) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_compact(15) = -0.5*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_compact(16) = -0.25*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_compact(17) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_compact(18) = -0.5*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_compact(19) = -0.25*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_compact(20) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_compact(21) = 0.5*(eta - 1)*(eta + 1)*(xi - 1);
    Sxi_compact(22) = 0.25*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1)*(xi - 1);
    Sxi_compact(23) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 1)*(xi - 1);
}

// 3x72 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_xi(Matrix3x3N& Sxi_xi, double xi, double eta, double zeta) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta);
    Sxi_xi.setZero();

    for (unsigned int s = 0; s < Sxi_xi_compact.size(); s++) {
        Sxi_xi(0, 0 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(1, 1 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(2, 2 + (3 * s)) = Sxi_xi_compact(s);
    }
}

// 24x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = -0.25*(eta - 1)*(eta + 2*xi);
    Sxi_xi_compact(1) = -0.125*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 2*xi);
    Sxi_xi_compact(2) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 2*xi);
    Sxi_xi_compact(3) = 0.25*(eta - 1)*(eta - 2*xi);
    Sxi_xi_compact(4) = 0.125*(m_thicknessZ)*(zeta)*(eta - 1)*(eta - 2*xi);
    Sxi_xi_compact(5) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta - 2*xi);
    Sxi_xi_compact(6) = 0.25*(eta + 1)*(eta + 2*xi);
    Sxi_xi_compact(7) = 0.125*(m_thicknessZ)*(zeta)*(eta + 1)*(eta + 2*xi);
    Sxi_xi_compact(8) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta + 1)*(eta + 2*xi);
    Sxi_xi_compact(9) = -0.25*(eta + 1)*(eta - 2*xi);
    Sxi_xi_compact(10) = -0.125*(m_thicknessZ)*(zeta)*(eta + 1)*(eta - 2*xi);
    Sxi_xi_compact(11) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta + 1)*(eta - 2*xi);
    Sxi_xi_compact(12) = xi*(eta - 1);
    Sxi_xi_compact(13) = 0.5*(m_thicknessZ)*(xi)*(zeta)*(eta - 1);
    Sxi_xi_compact(14) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(xi)*(zeta)*(zeta)*(eta - 1);
    Sxi_xi_compact(15) = -0.5*(eta - 1)*(eta + 1);
    Sxi_xi_compact(16) = -0.25*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1);
    Sxi_xi_compact(17) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 1);
    Sxi_xi_compact(18) = -xi*(eta + 1);
    Sxi_xi_compact(19) = -0.5*(m_thicknessZ)*(xi)*(zeta)*(eta + 1);
    Sxi_xi_compact(20) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(xi)*(zeta)*(zeta)*(eta + 1);
    Sxi_xi_compact(21) = 0.5*(eta - 1)*(eta + 1);
    Sxi_xi_compact(22) = 0.25*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1);
    Sxi_xi_compact(23) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 1);
}

// 3x72 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_eta(Matrix3x3N& Sxi_eta, double xi, double eta, double zeta) {
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta);
    Sxi_eta.setZero();

    for (unsigned int s = 0; s < Sxi_eta_compact.size(); s++) {
        Sxi_eta(0, 0 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(1, 1 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(2, 2 + (3 * s)) = Sxi_eta_compact(s);
    }
}

// 24x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = -0.25*(xi - 1)*(2*eta + xi);
    Sxi_eta_compact(1) = -0.125*(m_thicknessZ)*(zeta)*(xi - 1)*(2*eta + xi);
    Sxi_eta_compact(2) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(2*eta + xi);
    Sxi_eta_compact(3) = 0.25*(xi + 1)*(2*eta - xi);
    Sxi_eta_compact(4) = 0.125*(m_thicknessZ)*(zeta)*(xi + 1)*(2*eta - xi);
    Sxi_eta_compact(5) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi + 1)*(2*eta - xi);
    Sxi_eta_compact(6) = 0.25*(xi + 1)*(2*eta + xi);
    Sxi_eta_compact(7) = 0.125*(m_thicknessZ)*(zeta)*(xi + 1)*(2*eta + xi);
    Sxi_eta_compact(8) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi + 1)*(2*eta + xi);
    Sxi_eta_compact(9) = -0.25*(xi - 1)*(2*eta - xi);
    Sxi_eta_compact(10) = -0.125*(m_thicknessZ)*(zeta)*(xi - 1)*(2*eta - xi);
    Sxi_eta_compact(11) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(2*eta - xi);
    Sxi_eta_compact(12) = 0.5*(xi - 1)*(xi + 1);
    Sxi_eta_compact(13) = 0.25*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_eta_compact(14) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_eta_compact(15) = -eta*(xi + 1);
    Sxi_eta_compact(16) = -0.5*(m_thicknessZ)*(eta)*(zeta)*(xi + 1);
    Sxi_eta_compact(17) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(eta)*(zeta)*(zeta)*(xi + 1);
    Sxi_eta_compact(18) = -0.5*(xi - 1)*(xi + 1);
    Sxi_eta_compact(19) = -0.25*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_eta_compact(20) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_eta_compact(21) = eta*(xi - 1);
    Sxi_eta_compact(22) = 0.5*(m_thicknessZ)*(eta)*(zeta)*(xi - 1);
    Sxi_eta_compact(23) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(eta)*(zeta)*(zeta)*(xi - 1);
}

// 3x72 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_zeta(Matrix3x3N& Sxi_zeta, double xi, double eta, double zeta) {
    VectorN Sxi_zeta_compact;
    Calc_Sxi_zeta_compact(Sxi_zeta_compact, xi, eta, zeta);
    Sxi_zeta.setZero();

    for (unsigned int s = 0; s < Sxi_zeta_compact.size(); s++) {
        Sxi_zeta(0, 0 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(1, 1 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(2, 2 + (3 * s)) = Sxi_zeta_compact(s);
    }
}

// 24x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]
void ChElementShellANCF_3833_TR07S::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta) {
    Sxi_zeta_compact(0) = 0;
    Sxi_zeta_compact(1) = -0.125*(m_thicknessZ)*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_zeta_compact(2) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_zeta_compact(3) = 0;
    Sxi_zeta_compact(4) = 0.125*(m_thicknessZ)*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_zeta_compact(5) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_zeta_compact(6) = 0;
    Sxi_zeta_compact(7) = 0.125*(m_thicknessZ)*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_zeta_compact(8) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_zeta_compact(9) = 0;
    Sxi_zeta_compact(10) = -0.125*(m_thicknessZ)*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_zeta_compact(11) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_zeta_compact(12) = 0;
    Sxi_zeta_compact(13) = 0.25*(m_thicknessZ)*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_zeta_compact(14) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_zeta_compact(15) = 0;
    Sxi_zeta_compact(16) = -0.25*(m_thicknessZ)*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_zeta_compact(17) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_zeta_compact(18) = 0;
    Sxi_zeta_compact(19) = -0.25*(m_thicknessZ)*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_zeta_compact(20) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_zeta_compact(21) = 0;
    Sxi_zeta_compact(22) = 0.25*(m_thicknessZ)*(eta - 1)*(eta + 1)*(xi - 1);
    Sxi_zeta_compact(23) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1)*(xi - 1);
}

void ChElementShellANCF_3833_TR07S::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = -0.25*(eta - 1)*(eta + 2 * xi);
    Sxi_D(1, 0) = -0.125*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 2 * xi);
    Sxi_D(2, 0) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 2 * xi);
    Sxi_D(3, 0) = 0.25*(eta - 1)*(eta - 2 * xi);
    Sxi_D(4, 0) = 0.125*(m_thicknessZ)*(zeta)*(eta - 1)*(eta - 2 * xi);
    Sxi_D(5, 0) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta - 2 * xi);
    Sxi_D(6, 0) = 0.25*(eta + 1)*(eta + 2 * xi);
    Sxi_D(7, 0) = 0.125*(m_thicknessZ)*(zeta)*(eta + 1)*(eta + 2 * xi);
    Sxi_D(8, 0) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta + 1)*(eta + 2 * xi);
    Sxi_D(9, 0) = -0.25*(eta + 1)*(eta - 2 * xi);
    Sxi_D(10, 0) = -0.125*(m_thicknessZ)*(zeta)*(eta + 1)*(eta - 2 * xi);
    Sxi_D(11, 0) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta + 1)*(eta - 2 * xi);
    Sxi_D(12, 0) = xi * (eta - 1);
    Sxi_D(13, 0) = 0.5*(m_thicknessZ)*(xi)*(zeta)*(eta - 1);
    Sxi_D(14, 0) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(xi)*(zeta)*(zeta)*(eta - 1);
    Sxi_D(15, 0) = -0.5*(eta - 1)*(eta + 1);
    Sxi_D(16, 0) = -0.25*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1);
    Sxi_D(17, 0) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 1);
    Sxi_D(18, 0) = -xi * (eta + 1);
    Sxi_D(19, 0) = -0.5*(m_thicknessZ)*(xi)*(zeta)*(eta + 1);
    Sxi_D(20, 0) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(xi)*(zeta)*(zeta)*(eta + 1);
    Sxi_D(21, 0) = 0.5*(eta - 1)*(eta + 1);
    Sxi_D(22, 0) = 0.25*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1);
    Sxi_D(23, 0) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(eta - 1)*(eta + 1);

    Sxi_D(0, 1) = -0.25*(xi - 1)*(2 * eta + xi);
    Sxi_D(1, 1) = -0.125*(m_thicknessZ)*(zeta)*(xi - 1)*(2 * eta + xi);
    Sxi_D(2, 1) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(2 * eta + xi);
    Sxi_D(3, 1) = 0.25*(xi + 1)*(2 * eta - xi);
    Sxi_D(4, 1) = 0.125*(m_thicknessZ)*(zeta)*(xi + 1)*(2 * eta - xi);
    Sxi_D(5, 1) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi + 1)*(2 * eta - xi);
    Sxi_D(6, 1) = 0.25*(xi + 1)*(2 * eta + xi);
    Sxi_D(7, 1) = 0.125*(m_thicknessZ)*(zeta)*(xi + 1)*(2 * eta + xi);
    Sxi_D(8, 1) = 0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi + 1)*(2 * eta + xi);
    Sxi_D(9, 1) = -0.25*(xi - 1)*(2 * eta - xi);
    Sxi_D(10, 1) = -0.125*(m_thicknessZ)*(zeta)*(xi - 1)*(2 * eta - xi);
    Sxi_D(11, 1) = -0.03125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(2 * eta - xi);
    Sxi_D(12, 1) = 0.5*(xi - 1)*(xi + 1);
    Sxi_D(13, 1) = 0.25*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_D(14, 1) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_D(15, 1) = -eta * (xi + 1);
    Sxi_D(16, 1) = -0.5*(m_thicknessZ)*(eta)*(zeta)*(xi + 1);
    Sxi_D(17, 1) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(eta)*(zeta)*(zeta)*(xi + 1);
    Sxi_D(18, 1) = -0.5*(xi - 1)*(xi + 1);
    Sxi_D(19, 1) = -0.25*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_D(20, 1) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(zeta)*(xi - 1)*(xi + 1);
    Sxi_D(21, 1) = eta * (xi - 1);
    Sxi_D(22, 1) = 0.5*(m_thicknessZ)*(eta)*(zeta)*(xi - 1);
    Sxi_D(23, 1) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(eta)*(zeta)*(zeta)*(xi - 1);

    Sxi_D(0, 2) = 0;
    Sxi_D(1, 2) = -0.125*(m_thicknessZ)*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_D(2, 2) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(eta - 1)*(eta + xi + 1);
    Sxi_D(3, 2) = 0;
    Sxi_D(4, 2) = 0.125*(m_thicknessZ)*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_D(5, 2) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi + 1)*(eta - 1)*(eta - xi + 1);
    Sxi_D(6, 2) = 0;
    Sxi_D(7, 2) = 0.125*(m_thicknessZ)*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_D(8, 2) = 0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi + 1)*(eta + 1)*(eta + xi - 1);
    Sxi_D(9, 2) = 0;
    Sxi_D(10, 2) = -0.125*(m_thicknessZ)*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_D(11, 2) = -0.0625*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(eta + 1)*(eta - xi - 1);
    Sxi_D(12, 2) = 0;
    Sxi_D(13, 2) = 0.25*(m_thicknessZ)*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_D(14, 2) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1)*(eta - 1);
    Sxi_D(15, 2) = 0;
    Sxi_D(16, 2) = -0.25*(m_thicknessZ)*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_D(17, 2) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1)*(xi + 1);
    Sxi_D(18, 2) = 0;
    Sxi_D(19, 2) = -0.25*(m_thicknessZ)*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_D(20, 2) = -0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(xi - 1)*(xi + 1)*(eta + 1);
    Sxi_D(21, 2) = 0;
    Sxi_D(22, 2) = 0.25*(m_thicknessZ)*(eta - 1)*(eta + 1)*(xi - 1);
    Sxi_D(23, 2) = 0.125*(m_thicknessZ)*(m_thicknessZ)*(zeta)*(eta - 1)*(eta + 1)*(xi - 1);
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementShellANCF_3833_TR07S::CalcCoordVector(Vector3N& e) {
	e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();

    e.segment(9, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(12, 3) = m_nodes[1]->GetD().eigen();
    e.segment(15, 3) = m_nodes[1]->GetDD().eigen();

    e.segment(18, 3) = m_nodes[2]->GetPos().eigen();
    e.segment(21, 3) = m_nodes[2]->GetD().eigen();
    e.segment(24, 3) = m_nodes[2]->GetDD().eigen();

    e.segment(27, 3) = m_nodes[3]->GetPos().eigen();
    e.segment(30, 3) = m_nodes[3]->GetD().eigen();
    e.segment(33, 3) = m_nodes[3]->GetDD().eigen();

    e.segment(36, 3) = m_nodes[4]->GetPos().eigen();
    e.segment(39, 3) = m_nodes[4]->GetD().eigen();
    e.segment(42, 3) = m_nodes[4]->GetDD().eigen();

    e.segment(45, 3) = m_nodes[5]->GetPos().eigen();
    e.segment(48, 3) = m_nodes[5]->GetD().eigen();
    e.segment(51, 3) = m_nodes[5]->GetDD().eigen();

    e.segment(54, 3) = m_nodes[6]->GetPos().eigen();
    e.segment(57, 3) = m_nodes[6]->GetD().eigen();
    e.segment(60, 3) = m_nodes[6]->GetDD().eigen();

    e.segment(63, 3) = m_nodes[7]->GetPos().eigen();
    e.segment(66, 3) = m_nodes[7]->GetD().eigen();
    e.segment(69, 3) = m_nodes[7]->GetDD().eigen();
	
}

void ChElementShellANCF_3833_TR07S::CalcCoordMatrix(Matrix3xN& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetD().eigen();
    ebar.col(2) = m_nodes[0]->GetDD().eigen();

    ebar.col(3) = m_nodes[1]->GetPos().eigen();
    ebar.col(4) = m_nodes[1]->GetD().eigen();
    ebar.col(5) = m_nodes[1]->GetDD().eigen();

    ebar.col(6) = m_nodes[2]->GetPos().eigen();
    ebar.col(7) = m_nodes[2]->GetD().eigen();
    ebar.col(8) = m_nodes[2]->GetDD().eigen();

    ebar.col(9) = m_nodes[3]->GetPos().eigen();
    ebar.col(10) = m_nodes[3]->GetD().eigen();
    ebar.col(11) = m_nodes[3]->GetDD().eigen();

    ebar.col(12) = m_nodes[4]->GetPos().eigen();
    ebar.col(13) = m_nodes[4]->GetD().eigen();
    ebar.col(14) = m_nodes[4]->GetDD().eigen();

    ebar.col(15) = m_nodes[5]->GetPos().eigen();
    ebar.col(16) = m_nodes[5]->GetD().eigen();
    ebar.col(17) = m_nodes[5]->GetDD().eigen();

    ebar.col(18) = m_nodes[6]->GetPos().eigen();
    ebar.col(19) = m_nodes[6]->GetD().eigen();
    ebar.col(20) = m_nodes[6]->GetDD().eigen();

    ebar.col(21) = m_nodes[7]->GetPos().eigen();
    ebar.col(22) = m_nodes[7]->GetD().eigen();
    ebar.col(23) = m_nodes[7]->GetDD().eigen();
}

void ChElementShellANCF_3833_TR07S::CalcCoordMatrix(MatrixNx3c& ebar) {
    ebar.row(0) = m_nodes[0]->GetPos().eigen();
    ebar.row(1) = m_nodes[0]->GetD().eigen();
    ebar.row(2) = m_nodes[0]->GetDD().eigen();

    ebar.row(3) = m_nodes[1]->GetPos().eigen();
    ebar.row(4) = m_nodes[1]->GetD().eigen();
    ebar.row(5) = m_nodes[1]->GetDD().eigen();

    ebar.row(6) = m_nodes[2]->GetPos().eigen();
    ebar.row(7) = m_nodes[2]->GetD().eigen();
    ebar.row(8) = m_nodes[2]->GetDD().eigen();

    ebar.row(9) = m_nodes[3]->GetPos().eigen();
    ebar.row(10) = m_nodes[3]->GetD().eigen();
    ebar.row(11) = m_nodes[3]->GetDD().eigen();

    ebar.row(12) = m_nodes[4]->GetPos().eigen();
    ebar.row(13) = m_nodes[4]->GetD().eigen();
    ebar.row(14) = m_nodes[4]->GetDD().eigen();

    ebar.row(15) = m_nodes[5]->GetPos().eigen();
    ebar.row(16) = m_nodes[5]->GetD().eigen();
    ebar.row(17) = m_nodes[5]->GetDD().eigen();

    ebar.row(18) = m_nodes[6]->GetPos().eigen();
    ebar.row(19) = m_nodes[6]->GetD().eigen();
    ebar.row(20) = m_nodes[6]->GetDD().eigen();

    ebar.row(21) = m_nodes[7]->GetPos().eigen();
    ebar.row(22) = m_nodes[7]->GetD().eigen();
    ebar.row(23) = m_nodes[7]->GetDD().eigen();
}

void ChElementShellANCF_3833_TR07S::CalcCoordMatrix(MatrixNx3& ebar) {
    ebar.row(0) = m_nodes[0]->GetPos().eigen();
    ebar.row(1) = m_nodes[0]->GetD().eigen();
    ebar.row(2) = m_nodes[0]->GetDD().eigen();

    ebar.row(3) = m_nodes[1]->GetPos().eigen();
    ebar.row(4) = m_nodes[1]->GetD().eigen();
    ebar.row(5) = m_nodes[1]->GetDD().eigen();

    ebar.row(6) = m_nodes[2]->GetPos().eigen();
    ebar.row(7) = m_nodes[2]->GetD().eigen();
    ebar.row(8) = m_nodes[2]->GetDD().eigen();

    ebar.row(9) = m_nodes[3]->GetPos().eigen();
    ebar.row(10) = m_nodes[3]->GetD().eigen();
    ebar.row(11) = m_nodes[3]->GetDD().eigen();

    ebar.row(12) = m_nodes[4]->GetPos().eigen();
    ebar.row(13) = m_nodes[4]->GetD().eigen();
    ebar.row(14) = m_nodes[4]->GetDD().eigen();

    ebar.row(15) = m_nodes[5]->GetPos().eigen();
    ebar.row(16) = m_nodes[5]->GetD().eigen();
    ebar.row(17) = m_nodes[5]->GetDD().eigen();

    ebar.row(18) = m_nodes[6]->GetPos().eigen();
    ebar.row(19) = m_nodes[6]->GetD().eigen();
    ebar.row(20) = m_nodes[6]->GetDD().eigen();

    ebar.row(21) = m_nodes[7]->GetPos().eigen();
    ebar.row(22) = m_nodes[7]->GetD().eigen();
    ebar.row(23) = m_nodes[7]->GetDD().eigen();
}

void ChElementShellANCF_3833_TR07S::CalcCoordDerivVector(Vector3N& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();

    edot.segment(9, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(12, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetDD_dt().eigen();

    edot.segment(18, 3) = m_nodes[2]->GetPos_dt().eigen();
    edot.segment(21, 3) = m_nodes[2]->GetD_dt().eigen();
    edot.segment(24, 3) = m_nodes[2]->GetDD_dt().eigen();

    edot.segment(27, 3) = m_nodes[3]->GetPos_dt().eigen();
    edot.segment(30, 3) = m_nodes[3]->GetD_dt().eigen();
    edot.segment(33, 3) = m_nodes[3]->GetDD_dt().eigen();

    edot.segment(36, 3) = m_nodes[4]->GetPos_dt().eigen();
    edot.segment(39, 3) = m_nodes[4]->GetD_dt().eigen();
    edot.segment(42, 3) = m_nodes[4]->GetDD_dt().eigen();

    edot.segment(45, 3) = m_nodes[5]->GetPos_dt().eigen();
    edot.segment(48, 3) = m_nodes[5]->GetD_dt().eigen();
    edot.segment(51, 3) = m_nodes[5]->GetDD_dt().eigen();

    edot.segment(54, 3) = m_nodes[6]->GetPos_dt().eigen();
    edot.segment(57, 3) = m_nodes[6]->GetD_dt().eigen();
    edot.segment(60, 3) = m_nodes[6]->GetDD_dt().eigen();

    edot.segment(63, 3) = m_nodes[7]->GetPos_dt().eigen();
    edot.segment(66, 3) = m_nodes[7]->GetD_dt().eigen();
    edot.segment(69, 3) = m_nodes[7]->GetDD_dt().eigen();
}

void ChElementShellANCF_3833_TR07S::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.col(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetDD_dt().eigen();

    ebardot.col(3) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.col(4) = m_nodes[1]->GetD_dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetDD_dt().eigen();

    ebardot.col(6) = m_nodes[2]->GetPos_dt().eigen();
    ebardot.col(7) = m_nodes[2]->GetD_dt().eigen();
    ebardot.col(8) = m_nodes[2]->GetDD_dt().eigen();

    ebardot.col(9) = m_nodes[3]->GetPos_dt().eigen();
    ebardot.col(10) = m_nodes[3]->GetD_dt().eigen();
    ebardot.col(11) = m_nodes[3]->GetDD_dt().eigen();

    ebardot.col(12) = m_nodes[4]->GetPos_dt().eigen();
    ebardot.col(13) = m_nodes[4]->GetD_dt().eigen();
    ebardot.col(14) = m_nodes[4]->GetDD_dt().eigen();

    ebardot.col(15) = m_nodes[5]->GetPos_dt().eigen();
    ebardot.col(16) = m_nodes[5]->GetD_dt().eigen();
    ebardot.col(17) = m_nodes[5]->GetDD_dt().eigen();

    ebardot.col(18) = m_nodes[6]->GetPos_dt().eigen();
    ebardot.col(19) = m_nodes[6]->GetD_dt().eigen();
    ebardot.col(20) = m_nodes[6]->GetDD_dt().eigen();

    ebardot.col(21) = m_nodes[7]->GetPos_dt().eigen();
    ebardot.col(22) = m_nodes[7]->GetD_dt().eigen();
    ebardot.col(23) = m_nodes[7]->GetDD_dt().eigen();
}

void ChElementShellANCF_3833_TR07S::CalcCoordDerivMatrix(MatrixNx3c& ebardot) {
    ebardot.row(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.row(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.row(2) = m_nodes[0]->GetDD_dt().eigen();

    ebardot.row(3) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.row(4) = m_nodes[1]->GetD_dt().eigen();
    ebardot.row(5) = m_nodes[1]->GetDD_dt().eigen();

    ebardot.row(6) = m_nodes[2]->GetPos_dt().eigen();
    ebardot.row(7) = m_nodes[2]->GetD_dt().eigen();
    ebardot.row(8) = m_nodes[2]->GetDD_dt().eigen();

    ebardot.row(9) = m_nodes[3]->GetPos_dt().eigen();
    ebardot.row(10) = m_nodes[3]->GetD_dt().eigen();
    ebardot.row(11) = m_nodes[3]->GetDD_dt().eigen();

    ebardot.row(12) = m_nodes[4]->GetPos_dt().eigen();
    ebardot.row(13) = m_nodes[4]->GetD_dt().eigen();
    ebardot.row(14) = m_nodes[4]->GetDD_dt().eigen();

    ebardot.row(15) = m_nodes[5]->GetPos_dt().eigen();
    ebardot.row(16) = m_nodes[5]->GetD_dt().eigen();
    ebardot.row(17) = m_nodes[5]->GetDD_dt().eigen();

    ebardot.row(18) = m_nodes[6]->GetPos_dt().eigen();
    ebardot.row(19) = m_nodes[6]->GetD_dt().eigen();
    ebardot.row(20) = m_nodes[6]->GetDD_dt().eigen();

    ebardot.row(21) = m_nodes[7]->GetPos_dt().eigen();
    ebardot.row(22) = m_nodes[7]->GetD_dt().eigen();
    ebardot.row(23) = m_nodes[7]->GetDD_dt().eigen();
}

void ChElementShellANCF_3833_TR07S::CalcCombinedCoordMatrix(MatrixNx6& ebar_ebardot) {
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

    ebar_ebardot.block<1, 3>(9, 0) = m_nodes[3]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(9, 3) = m_nodes[3]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(10, 0) = m_nodes[3]->GetD().eigen();
    ebar_ebardot.block<1, 3>(10, 3) = m_nodes[3]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(11, 0) = m_nodes[3]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(11, 3) = m_nodes[3]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(12, 0) = m_nodes[4]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(12, 3) = m_nodes[4]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(13, 0) = m_nodes[4]->GetD().eigen();
    ebar_ebardot.block<1, 3>(13, 3) = m_nodes[4]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(14, 0) = m_nodes[4]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(14, 3) = m_nodes[4]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(15, 0) = m_nodes[5]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(15, 3) = m_nodes[5]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(16, 0) = m_nodes[5]->GetD().eigen();
    ebar_ebardot.block<1, 3>(16, 3) = m_nodes[5]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(17, 0) = m_nodes[5]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(17, 3) = m_nodes[5]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(18, 0) = m_nodes[6]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(18, 3) = m_nodes[6]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(19, 0) = m_nodes[6]->GetD().eigen();
    ebar_ebardot.block<1, 3>(19, 3) = m_nodes[6]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(20, 0) = m_nodes[6]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(20, 3) = m_nodes[6]->GetDD_dt().eigen();

    ebar_ebardot.block<1, 3>(21, 0) = m_nodes[7]->GetPos().eigen();
    ebar_ebardot.block<1, 3>(21, 3) = m_nodes[7]->GetPos_dt().eigen();
    ebar_ebardot.block<1, 3>(22, 0) = m_nodes[7]->GetD().eigen();
    ebar_ebardot.block<1, 3>(22, 3) = m_nodes[7]->GetD_dt().eigen();
    ebar_ebardot.block<1, 3>(23, 0) = m_nodes[7]->GetDD().eigen();
    ebar_ebardot.block<1, 3>(23, 3) = m_nodes[7]->GetDD_dt().eigen();

}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementShellANCF_3833_TR07S::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementShellANCF_3833_TR07S::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------

void ChElementShellANCF_3833_TR07S::EvaluateSectionFrame(const double xi, const double eta, ChVector<>& point, ChQuaternion<>& rot) {
    Matrix3x3N Sxi;
    Matrix3x3N Sxi_xi;
    Matrix3x3N Sxi_eta;
    Calc_Sxi(Sxi, xi, eta, 0);
    Calc_Sxi_xi(Sxi_xi, xi, eta, 0);
    Calc_Sxi_eta(Sxi_eta, xi, eta, 0);

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

void ChElementShellANCF_3833_TR07S::EvaluateSectionPoint(const double u, const double v, ChVector<>& point) {
    Matrix3x3N Sxi;
    Calc_Sxi(Sxi, u, v, 0);

    Vector3N e;
    CalcCoordVector(e);

    // r = Se
    point = Sxi * e;
}

void ChElementShellANCF_3833_TR07S::EvaluateSectionVelNorm(double U, double V, ChVector<>& Result) {
    Matrix3x3N Sxi_zeta;
    Vector3N edot;
    ChVector<> r_zeta;

    Calc_Sxi_zeta(Sxi_zeta, U, V, 0);
    CalcCoordDerivVector(edot);
    Result = Sxi_zeta * edot;
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).
void ChElementShellANCF_3833_TR07S::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
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
void ChElementShellANCF_3833_TR07S::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
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
void ChElementShellANCF_3833_TR07S::LoadableStateIncrement(const unsigned int off_x,
                                                         ChState& x_new,
                                                         const ChState& x,
                                                         const unsigned int off_v,
                                                         const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 24, x_new, x, off_v + 24, Dv);
    m_nodes[3]->NodeIntStateIncrement(off_x + 36, x_new, x, off_v + 36, Dv);
}

// Get the pointers to the contained ChVariables, appending to the mvars vector.
void ChElementShellANCF_3833_TR07S::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U) coordinates of the centerline.
void ChElementShellANCF_3833_TR07S::ComputeNF(
    const double U,              // parametric coordinate in surface
    const double V,              // parametric coordinate in surface
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force
    Matrix3x3N Sxi;
    Calc_Sxi(Sxi, U, V, 0);
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

    Calc_Sxi_xi(Sxi_xi, U, V, 0);
    Calc_Sxi_eta(Sxi_eta, U, V, 0);
    Calc_Sxi_zeta(Sxi_zeta, U, V, 0);

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

// Evaluate N'*F , where N is the shape function evaluated at (U,V,W) coordinates of the surface.
void ChElementShellANCF_3833_TR07S::ComputeNF(
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
double ChElementShellANCF_3833_TR07S::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at (U) coordinates.
ChVector<> ChElementShellANCF_3833_TR07S::ComputeNormal(const double U, const double V) {
    Matrix3x3N Sxi_zeta;
    Vector3N e;
    ChVector<> r_zeta;

    Calc_Sxi_zeta(Sxi_zeta, U, V, 0);
    CalcCoordVector(e);
    r_zeta = Sxi_zeta * e;

    return r_zeta.GetNormalized();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3833_TR07S(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementShellANCF_3833_TR07S::GetStaticGQTables() {
    return &static_tables_3833_TR07S;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialShellANCF_3833_TR07S methods
// ============================================================================

// Construct an isotropic material.
ChMaterialShellANCF_3833_TR07S::ChMaterialShellANCF_3833_TR07S(
    double rho,        // material density
    double E,          // Young's modulus
    double nu          // Poisson ratio
    )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G));
}

// Construct a (possibly) orthotropic material.
ChMaterialShellANCF_3833_TR07S::ChMaterialShellANCF_3833_TR07S(
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
void ChMaterialShellANCF_3833_TR07S::Calc_D0_Dv(const ChVector<>& E,
                                              const ChVector<>& nu,
                                              const ChVector<>& G) {
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
