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
// Internal Force Calculation Method is based on:  Liu, Cheng, Qiang Tian, and
// Haiyan Hu. "Dynamics of a large scale rigid–flexible multibody system composed
// of composite laminated plates." Multibody System Dynamics 26, no. 3 (2011): 283-305.
// =============================================================================
// TR09 = a Liu style implementation of the element with pre-calculation
//     of the matrices needed for both the internal force (O1) and Jacobian Calculation (O2)
//
//  Mass Matrix = Constant, pre-calculated 8x8 matrix
//
//  Generalized Force due to gravity = Constant 96x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Liu method:
//     Math is based on the method presented by Liu, Tian, and Hu
//     "Full Integration" Number of GQ Integration Points (7x7x7)
//
//  Jacobian of the Generalized Internal Force Vector = Analytical Jacobian that
//     is based on the method presented by Liu, Tian, and Hu.
//     Internal force calculation results are cached for reuse during the Jacobian
//     calculations
//     O2 is precalculated and stored rather than calculated from O1
//
// =============================================================================

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementBrickANCF_3843_TR09.h"
#include <cmath>

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBrickANCF_3843_TR09::ChElementBrickANCF_3843_TR09()
    : m_gravity_on(false), m_lenX(0), m_lenY(0), m_lenZ(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(8);

    m_O1.resize(1024, 1024);
    m_O2.resize(1024, 1024);
    m_K3Compact.resize(32, 32);
    m_K13Compact.resize(32, 32);

    m_K13Compact.setZero();
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR09::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
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
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementBrickANCF_3843_TR09::SetupInitial(ChSystem* system) {
    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementBrickANCF_3843_TR09::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementBrickANCF_3843_TR09::GetStateBlock(ChVectorDynamic<>& mD) {
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
void ChElementBrickANCF_3843_TR09::ComputeKRMmatricesGlobal(ChMatrixRef H,
    double Kfactor,
    double Rfactor,
    double Mfactor) {
    assert((H.rows() == 96) && (H.cols() == 96));

    MatrixNx3c e_bar;
    MatrixNx3c e_bar_dot;

    CalcCoordMatrix(e_bar);
    CalcCoordDerivMatrix(e_bar_dot);

    // Liu Jacobian Method with Damping modifications
    Matrix3xN temp =
        (Kfactor + m_Alpha * Rfactor) * e_bar.transpose() + (m_Alpha * Kfactor) * e_bar_dot.transpose();
    ChMatrixNM<double, 1, 32> tempRow0 = temp.block<1, 32>(0, 0);
    ChMatrixNM<double, 1, 32> tempRow1 = temp.block<1, 32>(1, 0);
    ChMatrixNM<double, 1, 32> tempRow2 = temp.block<1, 32>(2, 0);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> PI2;
    PI2.resize(9, 1024);

    for (unsigned int v = 0; v < 32; v++) {
        PI2.block<3, 32>(0, 32 * v) = e_bar.block<1, 3>(v, 0).transpose() * tempRow0;
        PI2.block<3, 32>(3, 32 * v) = e_bar.block<1, 3>(v, 0).transpose() * tempRow1;
        PI2.block<3, 32>(6, 32 * v) = e_bar.block<1, 3>(v, 0).transpose() * tempRow2;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>  K2 = -PI2 * m_O2;

    for (unsigned int k = 0; k < 32; k++) {
        for (unsigned int f = 0; f < 32; f++) {
            H.block<3, 1>(3 * k, 3 * f) = K2.block<3, 1>(0, 32 * f + k);
            H.block<3, 1>(3 * k, 3 * f + 1) = K2.block<3, 1>(3, 32 * f + k);
            H.block<3, 1>(3 * k, 3 * f + 2) = K2.block<3, 1>(6, 32 * f + k);
        }
    }

    MatrixNxN KM_Compact = Mfactor * m_MassMatrix;
    KM_Compact -= Kfactor * m_K13Compact;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = 0; j < 32; j++) {
            H(3 * i, 3 * j) += KM_Compact(i, j);
            H(3 * i + 1, 3 * j + 1) += KM_Compact(i, j);
            H(3 * i + 2, 3 * j + 2) += KM_Compact(i, j);
        }
    }
}

// Return the mass matrix.
void ChElementBrickANCF_3843_TR09::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Inflate the Mass Matrix since it is stored in compact form.
    // In MATLAB notation:
    // M(1:3:end,1:3:end) = m_MassMatrix;
    // M(2:3:end,2:3:end) = m_MassMatrix;
    // M(3:3:end,3:3:end) = m_MassMatrix;
    for (unsigned int i = 0; i < 32; i++) {
        for (unsigned int j = 0; j < 32; j++) {
            M(3 * i, 3 * j) = m_MassMatrix(i, j);
            M(3 * i + 1, 3 * j + 1) = m_MassMatrix(i, j);
            M(3 * i + 2, 3 * j + 2) = m_MassMatrix(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------
void ChElementBrickANCF_3843_TR09::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
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
    m_MassMatrix.setZero();
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
                m_MassMatrix += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }
}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementBrickANCF_3843_TR09::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0, 0) + m_MassMatrix(0, 12) + m_MassMatrix(0, 24) + m_MassMatrix(0, 36) + m_MassMatrix(0, 48) + m_MassMatrix(0, 60) + m_MassMatrix(0, 72) + m_MassMatrix(0, 84);
    m_nodes[1]->m_TotalMass += m_MassMatrix(12, 0) + m_MassMatrix(12, 12) + m_MassMatrix(12, 24) + m_MassMatrix(12, 36) + m_MassMatrix(12, 48) + m_MassMatrix(12, 60) + m_MassMatrix(12, 72) + m_MassMatrix(12, 84);
    m_nodes[2]->m_TotalMass += m_MassMatrix(24, 0) + m_MassMatrix(24, 12) + m_MassMatrix(24, 24) + m_MassMatrix(24, 36) + m_MassMatrix(24, 48) + m_MassMatrix(24, 60) + m_MassMatrix(24, 72) + m_MassMatrix(24, 84);
    m_nodes[3]->m_TotalMass += m_MassMatrix(36, 0) + m_MassMatrix(36, 12) + m_MassMatrix(36, 24) + m_MassMatrix(36, 36) + m_MassMatrix(36, 48) + m_MassMatrix(36, 60) + m_MassMatrix(36, 72) + m_MassMatrix(36, 84);
    m_nodes[4]->m_TotalMass += m_MassMatrix(48, 0) + m_MassMatrix(48, 12) + m_MassMatrix(48, 24) + m_MassMatrix(48, 36) + m_MassMatrix(48, 48) + m_MassMatrix(48, 60) + m_MassMatrix(48, 72) + m_MassMatrix(48, 84);
    m_nodes[5]->m_TotalMass += m_MassMatrix(60, 0) + m_MassMatrix(60, 12) + m_MassMatrix(60, 24) + m_MassMatrix(60, 36) + m_MassMatrix(60, 48) + m_MassMatrix(60, 60) + m_MassMatrix(60, 72) + m_MassMatrix(60, 84);
    m_nodes[6]->m_TotalMass += m_MassMatrix(72, 0) + m_MassMatrix(72, 12) + m_MassMatrix(72, 24) + m_MassMatrix(72, 36) + m_MassMatrix(72, 48) + m_MassMatrix(72, 60) + m_MassMatrix(72, 72) + m_MassMatrix(72, 84);
    m_nodes[7]->m_TotalMass += m_MassMatrix(84, 0) + m_MassMatrix(84, 12) + m_MassMatrix(84, 24) + m_MassMatrix(84, 36) + m_MassMatrix(84, 48) + m_MassMatrix(84, 60) + m_MassMatrix(84, 72) + m_MassMatrix(84, 84);
}

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementBrickANCF_3843_TR09::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta = 6;    // 7 Point Gauss-Quadrature;
    unsigned int GQ_idx_zeta = 6;      // 7 Point Gauss-Quadrature;


    m_K3Compact.setZero();
    m_O1.setZero();

    const ChMatrixNM<double, 6, 6>& D = GetMaterial()->Get_D();

    ChMatrixNM<double, 3, 3> D11;
    ChMatrixNM<double, 3, 3> D22;
    ChMatrixNM<double, 3, 3> D33;
    D11(0, 0) = D(0, 0);
    D11(1, 1) = D(1, 0);
    D11(2, 2) = D(2, 0);
    D11(1, 0) = D(5, 0);
    D11(0, 1) = D(5, 0);
    D11(2, 0) = D(4, 0);
    D11(0, 2) = D(4, 0);
    D11(2, 1) = D(3, 0);
    D11(1, 2) = D(3, 0);

    D22(0, 0) = D(0, 1);
    D22(1, 1) = D(1, 1);
    D22(2, 2) = D(2, 1);
    D22(1, 0) = D(5, 1);
    D22(0, 1) = D(5, 1);
    D22(2, 0) = D(4, 1);
    D22(0, 2) = D(4, 1);
    D22(2, 1) = D(3, 1);
    D22(1, 2) = D(3, 1);

    D33(0, 0) = D(0, 2);
    D33(1, 1) = D(1, 2);
    D33(2, 2) = D(2, 2);
    D33(1, 0) = D(5, 2);
    D33(0, 1) = D(5, 2);
    D33(2, 0) = D(4, 2);
    D33(0, 2) = D(4, 2);
    D33(2, 1) = D(3, 2);
    D33(1, 2) = D(3, 2);

    ChMatrixNM<double, 9, 9> D_block;
    D_block << D(0, 0), D(0, 5), D(0, 4), D(0, 5), D(0, 1), D(0, 3), D(0, 4), D(0, 3), D(0, 2), D(5, 0), D(5, 5),
        D(5, 4), D(5, 5), D(5, 1), D(5, 3), D(5, 4), D(5, 3), D(5, 2), D(4, 0), D(4, 5), D(4, 4), D(4, 5), D(4, 1),
        D(4, 3), D(4, 4), D(4, 3), D(4, 2), D(5, 0), D(5, 5), D(5, 4), D(5, 5), D(5, 1), D(5, 3), D(5, 4), D(5, 3),
        D(5, 2), D(1, 0), D(1, 5), D(1, 4), D(1, 5), D(1, 1), D(1, 3), D(1, 4), D(1, 3), D(1, 2), D(3, 0), D(3, 5),
        D(3, 4), D(3, 5), D(3, 1), D(3, 3), D(3, 4), D(3, 3), D(3, 2), D(4, 0), D(4, 5), D(4, 4), D(4, 5), D(4, 1),
        D(4, 3), D(4, 4), D(4, 3), D(4, 2), D(3, 0), D(3, 5), D(3, 4), D(3, 5), D(3, 1), D(3, 3), D(3, 4), D(3, 3),
        D(3, 2), D(2, 0), D(2, 5), D(2, 4), D(2, 5), D(2, 1), D(2, 3), D(2, 4), D(2, 3), D(2, 2);

    // Pre-calculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determinate of the element Jacobian for later Calculating the
    // portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta][it_xi] * GQTable->Weight[GQ_idx_xi_eta][it_eta] *
                    GQTable->Weight[GQ_idx_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_zeta][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_zeta].size() +
                    it_xi * GQTable->Lroots[GQ_idx_zeta].size() * GQTable->Lroots[GQ_idx_xi_eta].size();
                ChMatrix33<double>
                    J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
                MatrixNx3c Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                MatrixNx3c Sxi_D_0xi = Sxi_D * J_0xi.inverse();
                double GQWeight_det_J_0xi = -J_0xi.determinant() * GQ_weight;

                m_K3Compact += GQWeight_det_J_0xi * 0.5 * (Sxi_D_0xi*D11*Sxi_D_0xi.transpose() + Sxi_D_0xi *
                    D22*Sxi_D_0xi.transpose() + Sxi_D_0xi * D33*Sxi_D_0xi.transpose());

                MatrixNxN scale;
                for (unsigned int n = 0; n < 3; n++) {
                    for (unsigned int c = 0; c < 3; c++) {
                        scale = Sxi_D_0xi * D_block.block<3, 3>(3 * n, 3 * c) * Sxi_D_0xi.transpose();
                        scale *= GQWeight_det_J_0xi;

                        MatrixNxN Sxi_D_0xi_n_Sxi_D_0xi_c_transpose =
                            Sxi_D_0xi.block<32, 1>(0, n) * Sxi_D_0xi.block<32, 1>(0, c).transpose();
                        for (unsigned int f = 0; f < 32; f++) {
                            for (unsigned int t = 0; t < 32; t++) {
                                m_O1.block<32, 32>(32 * t, 32 * f) += scale(t, f) * Sxi_D_0xi_n_Sxi_D_0xi_c_transpose;
                            }
                        }
                    }
                }
            }
        }
    }

    for (unsigned int f = 0; f < 32; f++) {
        for (unsigned int t = 0; t < 32; t++) {
            m_O2.block<32, 32>(32 * t, 32 * f) = m_O1.block<32, 32>(32 * t, 32 * f).transpose();
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementBrickANCF_3843_TR09::SetAlphaDamp(double a) {
    m_Alpha = a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementBrickANCF_3843_TR09::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == m_GravForce.size());
    MatrixNx3c ebar;
    MatrixNx3c ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);
    ComputeInternalForcesAtState(Fi, ebar, ebardot);

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementBrickANCF_3843_TR09::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
    const MatrixNx3c& ebar,
    const MatrixNx3c& ebardot) {
    MatrixNxN PI1_matrix = 0.5 * ebar * ebar.transpose();
    if (m_damping_enabled) {
        PI1_matrix += m_Alpha * ebardot * ebar.transpose();
    }
    MatrixNxN K1_matrix;
    Eigen::Map<ChVectorN<double, 1024>> PI1(PI1_matrix.data(), PI1_matrix.size());
    Eigen::Map<ChVectorN<double, 1024>> K1_vec(K1_matrix.data(), K1_matrix.size());
    K1_vec.noalias() = m_O1 * PI1;

    m_K13Compact.noalias() = K1_matrix - m_K3Compact;

    MatrixNx3 QiCompactLiu = m_K13Compact * ebar;
    Eigen::Map<Vector3N> QiReshapedLiu(QiCompactLiu.data(), QiCompactLiu.size());

    Fi = QiReshapedLiu;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR09::ComputeInternalJacobians(Matrix3Nx3N& JacobianMatrix,
                                                            double Kfactor,
                                                            double Rfactor) {
    // The integrated quantity represents the 96x96 Jacobian
    //      Kfactor * [K] + Rfactor * [R]
    // Note that the matrices with current nodal coordinates and velocities are
    // already available in m_d and m_d_dt (as set in ComputeInternalForces).
    // Similarly, the ANS strain and strain derivatives are already available in
    // m_strainANS and m_strainANS_D (as calculated in ComputeInternalForces).

    ChVectorDynamic<double> FiOrignal(96);
    ChVectorDynamic<double> FiDelta(96);
    MatrixNx3c ebar;
    MatrixNx3c ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);

    double delta = 1e-6;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForcesAtState(FiOrignal, ebar, ebardot);
    for (unsigned int i = 0; i < 96; i++) {
        ebar(i / 3, i % 3) = ebar(i / 3, i % 3) + delta;
        ComputeInternalForcesAtState(FiDelta, ebar, ebardot);
        JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
        ebar(i / 3, i % 3) = ebar(i / 3, i % 3) - delta;

        if (m_damping_enabled) {
            ebardot(i / 3, i % 3) = ebardot(i / 3, i % 3) + delta;
            ComputeInternalForcesAtState(FiDelta, ebar, ebardot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            ebardot(i / 3, i % 3) = ebardot(i / 3, i % 3) - delta;
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x96 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBrickANCF_3843_TR09::Calc_Sxi(Matrix3x3N& Sxi, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_xi(Matrix3x3N& Sxi_xi, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_eta(Matrix3x3N& Sxi_eta, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_zeta(Matrix3x3N& Sxi_zeta, double xi, double eta, double zeta) {
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
void ChElementBrickANCF_3843_TR09::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact,
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

void ChElementBrickANCF_3843_TR09::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
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

void ChElementBrickANCF_3843_TR09::CalcCoordVector(Vector3N& e) {
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

void ChElementBrickANCF_3843_TR09::CalcCoordMatrix(Matrix3xN& ebar) {
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

void ChElementBrickANCF_3843_TR09::CalcCoordMatrix(MatrixNx3c& ebar) {
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

void ChElementBrickANCF_3843_TR09::CalcCoordDerivVector(Vector3N& edot) {
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

void ChElementBrickANCF_3843_TR09::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
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

void ChElementBrickANCF_3843_TR09::CalcCoordDerivMatrix(MatrixNx3c& ebardot) {
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

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementBrickANCF_3843_TR09::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementBrickANCF_3843_TR09::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------

void ChElementBrickANCF_3843_TR09::EvaluateSectionFrame(const double xi,
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

void ChElementBrickANCF_3843_TR09::EvaluateSectionPoint(const double u, const double v, const double w, ChVector<>& point) {
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
void ChElementBrickANCF_3843_TR09::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
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
void ChElementBrickANCF_3843_TR09::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
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
void ChElementBrickANCF_3843_TR09::LoadableStateIncrement(const unsigned int off_x,
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
void ChElementBrickANCF_3843_TR09::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U,V,W) coordinates of the surface.
void ChElementBrickANCF_3843_TR09::ComputeNF(
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
double ChElementBrickANCF_3843_TR09::GetDensity() {
    return GetMaterial()->Get_rho();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_3843_TR09(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBrickANCF_3843_TR09::GetStaticGQTables() {
    return &static_tables_3843_TR09;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBrickANCF_3843_TR09 methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBrickANCF_3843_TR09::ChMaterialBrickANCF_3843_TR09(double rho,  // material density
                                                             double E,    // Young's modulus
                                                             double nu    // Poisson ratio
                                                             )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G));
}

// Construct a (possibly) orthotropic material.
ChMaterialBrickANCF_3843_TR09::ChMaterialBrickANCF_3843_TR09(
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
void ChMaterialBrickANCF_3843_TR09::Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G) {
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
