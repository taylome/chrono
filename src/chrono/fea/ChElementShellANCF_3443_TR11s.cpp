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
// Fully Parameterized ANCF shell element with 4 nodes. Description of this element
// and its internal forces may be found in: Henrik Ebel, Marko K Matikainen,
// Vesa-Ville Hurskainen, and Aki Mikkola. Analysis of high-order quadrilateral
// plate elements based on the absolute nodal coordinate formulation for three -
// dimensional elasticity.Advances in Mechanical Engineering, 9(6) : 1687814017705069, 2017.
// =============================================================================
// Internal Force Calculation Method is based on:  D Garcia-Vallejo, J Mayo,
// J L Escalona, and J Dominguez. Efficient evaluation of the elastic forces and
// the jacobian in the absolute nodal coordinate formulation. Nonlinear
// Dynamics, 35(4) : 313-329, 2004.
// =============================================================================
// TR11S = a Garcia-Vallejo style implementation of the element with pre-calculation
//     of the matrices needed for both the internal force and Jacobian Calculation
//
//  Mass Matrix = Constant, pre-calculated 8x8 matrix
//
//  Generalized Force due to gravity = Constant 24x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Garcia-Vallejo method:
//     Math is based on the method presented by Garcia-Vallejo et al.
//     "Full Integration" Number of GQ Integration Points (5x3x3)
//
//  Jacobian of the Generalized Internal Force Vector = Analytical Jacobian that
//     is based on the method presented by Garcia-Vallejo et al.
//     Internal force calculation results are cached for reuse during the Jacobian
//     calculations
//
// =============================================================================

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementShellANCF_3443_TR11s.h"
#include <cmath>

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementShellANCF_3443_TR11S::ChElementShellANCF_3443_TR11S()
    : m_gravity_on(false), m_lenX(0), m_lenY(0), m_thicknessZ(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(4);

    m_Ccompact.resize(2176, 16);  // N^3xN
    m_K1.resize(48, 48);          // 3Nx3N
    m_K2.resize(48, 48);          // 3Nx3N

    m_Ccompact.setZero();
    m_K1.setZero();
    m_K2.setZero();
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementShellANCF_3443_TR11S::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                            std::shared_ptr<ChNodeFEAxyzDDD> nodeB,
                                            std::shared_ptr<ChNodeFEAxyzDDD> nodeC,
                                            std::shared_ptr<ChNodeFEAxyzDDD> nodeD) {
    assert(nodeA);
    assert(nodeB);
    assert(nodeC);
    assert(nodeD);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;
    m_nodes[2] = nodeC;
    m_nodes[3] = nodeD;

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

    Kmatr.SetVariables(mvars);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementShellANCF_3443_TR11S::SetupInitial(ChSystem* system) {
    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementShellANCF_3443_TR11S::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementShellANCF_3443_TR11S::GetStateBlock(ChVectorDynamic<>& mD) {
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
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
void ChElementShellANCF_3443_TR11S::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                            double Kfactor,
                                                            double Rfactor,
                                                            double Mfactor) {
    assert((H.rows() == 48) && (H.cols() == 48));

#if true  // Analytical Jacobian
    ChMatrixNMc<double, 16, 3> e_bar;
    ChMatrixNMc<double, 16, 3> e_bar_dot;
    ChMatrixNMc<double, 16, 3> e_bar_plus_e_bar_dot;

    CalcCoordMatrix(e_bar);
    CalcCoordDerivMatrix(e_bar_dot);
    e_bar_plus_e_bar_dot = e_bar + m_2Alpha * e_bar_dot;

    Kfactor *= -1;
    Rfactor *= -1;

    H = Kfactor * (m_K1 + m_K2) + Rfactor * m_2Alpha * m_K2;

    unsigned int index = 0;
    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int s = i; s < 16; s++) {
            ChMatrixNM<double, 16, 16> Ccompact = m_Ccompact.block<16, 16>(16 * index, 0);
            index++;

            for (unsigned int k = 0; k < 16; k++) {
                double d = e_bar_plus_e_bar_dot.row(s) * e_bar.transpose() * Ccompact.row(k).transpose();
                ChMatrixNM<double, 3, 3> Block = e_bar.transpose() * Ccompact.col(k) * e_bar_plus_e_bar_dot.row(s);
                Block(0, 0) += d;
                Block(1, 1) += d;
                Block(2, 2) += d;
                H.block<3, 3>(3 * i, 3 * k) -= Kfactor * Block;
            }

            if (i != s) {
                for (unsigned int k = 0; k < 16; k++) {
                    double d = e_bar_plus_e_bar_dot.row(i) * e_bar.transpose() * Ccompact.col(k);
                    ChMatrixNM<double, 3, 3> Block = e_bar.transpose() * Ccompact.row(k).transpose() * e_bar_plus_e_bar_dot.row(i);
                    Block(0, 0) += d;
                    Block(1, 1) += d;
                    Block(2, 2) += d;
                    H.block<3, 3>(3 * s, 3 * k) -= Kfactor * Block;
                }
            }
        }
    }

#else  // Numeric Jacobian
    // Calculate the linear combination Kfactor*[K] + Rfactor*[R]
    ChMatrixNM<double, 48, 48> JacobianMatrix;
    ComputeInternalJacobians(JacobianMatrix, Kfactor, Rfactor);

    // Load Jac + Mfactor*[M] into H
    H = JacobianMatrix;
#endif

    unsigned int idx = 0;
    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int j = i; j < 16; j++) {
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
}

// Return the mass matrix.
void ChElementShellANCF_3443_TR11S::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Inflate the Mass Matrix since it is stored in compact form.
    // In MATLAB notation:
    // M(1:3:end,1:3:end) = m_MassMatrix;
    // M(2:3:end,2:3:end) = m_MassMatrix;
    // M(3:3:end,3:3:end) = m_MassMatrix;
    unsigned int idx = 0;
    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int j = i; j < 16; j++) {
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
void ChElementShellANCF_3443_TR11S::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, 7 GQ Points are needed in the xi & eta directions
    //  and 3 GQ Points are needed in the eta & zeta directions
    //  for exact integration of the element's mass matrix, even if
    //  the reference configuration is not straight
    // Mass Matrix Integrand is of order: 12 in xi, order: 12 in eta, and order: 4 in zeta.
    // Since the major pieces of the generalized force due to gravity
    //  can also be used to calculate the mass matrix, these calculations
    //  are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta = 6;  // 7 Point Gauss-Quadrature;
    unsigned int GQ_idx_zeta = 2;    // 3 Point Gauss-Quadrature;

    double rho = GetMaterial()->Get_rho();  // Density of the material for the element

    // Set these to zeros since they will be incremented as the vector/matrix is calculated
    MatrixNxN MassMatrixCompactSquare;
    MassMatrixCompactSquare.setZero();
    m_GravForce.setZero();

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta][it_xi] * GQTable->Weight[GQ_idx_xi_eta][it_eta] *
                                   GQTable->Weight[GQ_idx_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta);  // determinate of the element Jacobian (volume ratio)
                Matrix3x3N Sxi;                                    // 3x48 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                VectorN Sxi_compact;  // 16x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight * rho * det_J_0xi) * Sxi.transpose() * g_acc.eigen();
                MassMatrixCompactSquare += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }

    unsigned int idx = 0;
    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int j = i; j < 16; j++) {
            m_MassMatrix(idx) = MassMatrixCompactSquare(i, j);
            idx++;
        }
    }
}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementShellANCF_3443_TR11S::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0) + m_MassMatrix(4) + m_MassMatrix(8) + m_MassMatrix(12);
    m_nodes[1]->m_TotalMass += m_MassMatrix(4) + m_MassMatrix(58) + m_MassMatrix(62) + m_MassMatrix(66);
    m_nodes[2]->m_TotalMass += m_MassMatrix(8) + m_MassMatrix(62) + m_MassMatrix(100) + m_MassMatrix(104);
    m_nodes[3]->m_TotalMass += m_MassMatrix(12) + m_MassMatrix(66) + m_MassMatrix(104) + m_MassMatrix(126);
}

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementShellANCF_3443_TR11S::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi_eta = 6;  // 7 Point Gauss-Quadrature;
    unsigned int GQ_idx_zeta = 2;    // 3 Point Gauss-Quadrature;

    ChMatrixNM<double, 16, 16> K1compact;
    K1compact.setZero();
    m_Ccompact.setZero();
    m_K1.setZero();

    ChMatrixNM<double, 6, 6> D = GetMaterial()->Get_D();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> epsilion_matrices;
    epsilion_matrices.resize(96, 16);  // 6*NxN

    // Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight
    // reference configuration & GQ Weights times the determiate of the element Jacobian for later Calculating the
    // portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi_eta].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_xi_eta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi_eta][it_xi] * GQTable->Weight[GQ_idx_xi_eta][it_eta] *
                                   GQTable->Weight[GQ_idx_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi_eta][it_xi];
                double eta = GQTable->Lroots[GQ_idx_xi_eta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_zeta][it_zeta];

                ChMatrix33<double>
                    J_0xi;  // Element Jacobian between the reference configuration and normalized configuration
                ChMatrixNMc<double, 16, 3> Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;
                Sxi_D = Sxi_D * J_0xi.inverse();

                epsilion_matrices.block<16, 16>(0, 0) = 0.5 * Sxi_D.col(0) * Sxi_D.col(0).transpose();
                epsilion_matrices.block<16, 16>(16, 0) = 0.5 * Sxi_D.col(1) * Sxi_D.col(1).transpose();
                epsilion_matrices.block<16, 16>(32, 0) = 0.5 * Sxi_D.col(2) * Sxi_D.col(2).transpose();
                epsilion_matrices.block<16, 16>(48, 0) =
                    0.5 * Sxi_D.col(1) * Sxi_D.col(2).transpose() + 0.5 * Sxi_D.col(2) * Sxi_D.col(1).transpose();
                epsilion_matrices.block<16, 16>(64, 0) =
                    0.5 * Sxi_D.col(0) * Sxi_D.col(2).transpose() + 0.5 * Sxi_D.col(2) * Sxi_D.col(0).transpose();
                epsilion_matrices.block<16, 16>(80, 0) =
                    0.5 * Sxi_D.col(0) * Sxi_D.col(1).transpose() + 0.5 * Sxi_D.col(1) * Sxi_D.col(0).transpose();
                // depsilion_matrices_de = 2*epsilion_matrices

                // Just need to compute the upper triangular C compact block matrices since symmetry will be used
                // in the internal force calculations
                unsigned int index = 0;
                for (unsigned int i = 0; i < 16; i++) {
                    for (unsigned int j = i; j < 16; j++) {
                        for (unsigned int k = 0; k < 6; k++) {
                            for (unsigned int l = 0; l < 6; l++) {
                                m_Ccompact.block<16, 16>(16 * index, 0) +=
                                    J_0xi.determinant() * GQ_weight * 2 *
                                    epsilion_matrices.block<1, 16>(i + 16 * k, 0).transpose() * D(k, l) *
                                    epsilion_matrices.block<16, 1>(16 * l, j).transpose();
                            }
                        }
                        index++;
                    }
                }

                for (unsigned int k = 0; k < 6; k++) {
                    for (unsigned int l = 0; l < 3; l++) {
                        K1compact +=
                            J_0xi.determinant() * GQ_weight * D(k, l) * epsilion_matrices.block<16, 16>(16 * k, 0);
                    }
                }
            }
        }
    }

    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int j = 0; j < 16; j++) {
            m_K1(3 * i, 3 * j) = K1compact(i, j);
            m_K1(3 * i + 1, 3 * j + 1) = K1compact(i, j);
            m_K1(3 * i + 2, 3 * j + 2) = K1compact(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementShellANCF_3443_TR11S::SetAlphaDamp(double a) {
    m_Alpha = a;
    m_2Alpha = 2 * a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementShellANCF_3443_TR11S::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    assert(Fi.size() == m_GravForce.size());
    MatrixNx3c ebar;
    Vector3N edot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivVector(edot);
    ComputeInternalForcesAtState(Fi, ebar, edot);

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementShellANCF_3443_TR11S::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
                                                                const MatrixNx3c& ebar,
                                                                const Vector3N& edot) {
    ChMatrixNM<double, 3, 3> Block;
    unsigned int index = 0;
    for (unsigned int i = 0; i < 16; i++) {
        for (unsigned int j = i; j < 16; j++) {
            Block = -ebar.transpose() * m_Ccompact.block<16, 16>(16 * index, 0) * ebar;
            m_K2.block<3, 3>(3 * i, 3 * j) = Block;
            if (i != j) {
                m_K2.block<3, 3>(3 * j, 3 * i) = Block.transpose();
            }
            index++;
        }
    }

    MatrixNx3 ebar_rowmajor = ebar;
    Eigen::Map<Vector3N> e(ebar_rowmajor.data(), ebar_rowmajor.size());

    Fi = (m_K1 + m_K2) * e + m_2Alpha * m_K2 * edot;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_TR11S::ComputeInternalJacobians(Matrix3Nx3N& JacobianMatrix,
                                                            double Kfactor,
                                                            double Rfactor) {
    // The integrated quantity represents the 48x48 Jacobian
    //      Kfactor * [K] + Rfactor * [R]
    // Note that the matrices with current nodal coordinates and velocities are
    // already available in m_d and m_d_dt (as set in ComputeInternalForces).
    // Similarly, the ANS strain and strain derivatives are already available in
    // m_strainANS and m_strainANS_D (as calculated in ComputeInternalForces).

    ChVectorDynamic<double> FiOrignal(48);
    ChVectorDynamic<double> FiDelta(48);
    MatrixNx3c ebar;
    Vector3N edot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivVector(edot);

    double delta = 1e-6;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForcesAtState(FiOrignal, ebar, edot);
    for (unsigned int i = 0; i < 48; i++) {
        ebar(i / 3, i % 3) = ebar(i / 3, i % 3) + delta;
        ComputeInternalForcesAtState(FiDelta, ebar, edot);
        JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
        ebar(i / 3, i % 3) = ebar(i / 3, i % 3) - delta;

        if (m_damping_enabled) {
            edot(i) += delta;
            ComputeInternalForcesAtState(FiDelta, ebar, edot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            edot(i) -= delta;
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x48 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi(Matrix3x3N& Sxi, double xi, double eta, double zeta) {
    VectorN Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    Sxi.setZero();

    for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
        Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
        Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
        Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
    }
}

// 16x1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) = -0.125 * (xi - 1) * (eta - 1) * (eta * eta + eta + xi * xi + xi - 2);
    Sxi_compact(1) = -0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta - 1);
    Sxi_compact(2) = -0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi - 1);
    Sxi_compact(3) = 0.125 * m_thicknessZ * zeta * (xi - 1) * (eta - 1);

    Sxi_compact(4) = 0.125 * (xi + 1) * (eta - 1) * (eta * eta + eta + xi * xi - xi - 2);
    Sxi_compact(5) = -0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta - 1);
    Sxi_compact(6) = 0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1) * (xi + 1);
    Sxi_compact(7) = -0.125 * m_thicknessZ * zeta * (xi + 1) * (eta - 1);

    Sxi_compact(8) = -0.125 * (xi + 1) * (eta + 1) * (eta * eta - eta + xi * xi - xi - 2);
    Sxi_compact(9) = 0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1) * (eta + 1);
    Sxi_compact(10) = 0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi + 1);
    Sxi_compact(11) = 0.125 * m_thicknessZ * zeta * (xi + 1) * (eta + 1);

    Sxi_compact(12) = 0.125 * (xi - 1) * (eta + 1) * (eta * eta - eta + xi * xi + xi - 2);
    Sxi_compact(13) = 0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1) * (eta + 1);
    Sxi_compact(14) = -0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1) * (xi - 1);
    Sxi_compact(15) = -0.125 * m_thicknessZ * zeta * (xi - 1) * (eta + 1);
}

// 3x48 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_xi(Matrix3x3N& Sxi_xi, double xi, double eta, double zeta) {
    VectorN Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta);
    Sxi_xi.setZero();

    for (unsigned int s = 0; s < Sxi_xi_compact.size(); s++) {
        Sxi_xi(0, 0 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(1, 1 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(2, 2 + (3 * s)) = Sxi_xi_compact(s);
    }
}

// 16x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta) {
    Sxi_xi_compact(0) = -0.125 * (eta - 1) * (eta * eta + eta + 3 * xi * xi - 3);
    Sxi_xi_compact(1) = -0.0625 * m_lenX * (3 * xi + 1) * (xi - 1) * (eta - 1);
    Sxi_xi_compact(2) = -0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1);
    Sxi_xi_compact(3) = 0.125 * m_thicknessZ * zeta * (eta - 1);

    Sxi_xi_compact(4) = 0.125 * (eta - 1) * (eta * eta + eta + 3 * xi * xi - 3);
    Sxi_xi_compact(5) = -0.0625 * m_lenX * (xi + 1) * (3 * xi - 1) * (eta - 1);
    Sxi_xi_compact(6) = 0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1);
    Sxi_xi_compact(7) = -0.125 * m_thicknessZ * zeta * (eta - 1);

    Sxi_xi_compact(8) = -0.125 * (eta + 1) * (eta * eta - eta + 3 * xi * xi - 3);
    Sxi_xi_compact(9) = 0.0625 * m_lenX * (xi + 1) * (3 * xi - 1) * (eta + 1);
    Sxi_xi_compact(10) = 0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1);
    Sxi_xi_compact(11) = 0.125 * m_thicknessZ * zeta * (eta + 1);

    Sxi_xi_compact(12) = 0.125 * (eta + 1) * (eta * eta - eta + 3 * xi * xi - 3);
    Sxi_xi_compact(13) = 0.0625 * m_lenX * (3 * xi + 1) * (xi - 1) * (eta + 1);
    Sxi_xi_compact(14) = -0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1);
    Sxi_xi_compact(15) = -0.125 * m_thicknessZ * zeta * (eta + 1);
}

// 3x48 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_eta(Matrix3x3N& Sxi_eta, double xi, double eta, double zeta) {
    VectorN Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta);
    Sxi_eta.setZero();

    for (unsigned int s = 0; s < Sxi_eta_compact.size(); s++) {
        Sxi_eta(0, 0 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(1, 1 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(2, 2 + (3 * s)) = Sxi_eta_compact(s);
    }
}

// 16x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta) {
    Sxi_eta_compact(0) = -0.125 * (xi - 1) * (3 * eta * eta + xi * xi + xi - 3);
    Sxi_eta_compact(1) = -0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1);
    Sxi_eta_compact(2) = -0.0625 * m_lenY * (3 * eta + 1) * (eta - 1) * (xi - 1);
    Sxi_eta_compact(3) = 0.125 * m_thicknessZ * zeta * (xi - 1);

    Sxi_eta_compact(4) = 0.125 * (xi + 1) * (3 * eta * eta + xi * xi - xi - 3);
    Sxi_eta_compact(5) = -0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1);
    Sxi_eta_compact(6) = 0.0625 * m_lenY * (3 * eta + 1) * (eta - 1) * (xi + 1);
    Sxi_eta_compact(7) = -0.125 * m_thicknessZ * zeta * (xi + 1);

    Sxi_eta_compact(8) = -0.125 * (xi + 1) * (3 * eta * eta + xi * xi - xi - 3);
    Sxi_eta_compact(9) = 0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1);
    Sxi_eta_compact(10) = 0.0625 * m_lenY * (eta + 1) * (3 * eta - 1) * (xi + 1);
    Sxi_eta_compact(11) = 0.125 * m_thicknessZ * zeta * (xi + 1);

    Sxi_eta_compact(12) = 0.125 * (xi - 1) * (3 * eta * eta + xi * xi + xi - 3);
    Sxi_eta_compact(13) = 0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1);
    Sxi_eta_compact(14) = -0.0625 * m_lenY * (eta + 1) * (3 * eta - 1) * (xi - 1);
    Sxi_eta_compact(15) = -0.125 * m_thicknessZ * zeta * (xi - 1);
}

// 3x48 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_zeta(Matrix3x3N& Sxi_zeta, double xi, double eta, double zeta) {
    VectorN Sxi_zeta_compact;
    Calc_Sxi_zeta_compact(Sxi_zeta_compact, xi, eta, zeta);
    Sxi_zeta.setZero();

    for (unsigned int s = 0; s < Sxi_zeta_compact.size(); s++) {
        Sxi_zeta(0, 0 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(1, 1 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(2, 2 + (3 * s)) = Sxi_zeta_compact(s);
    }
}

// 16x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1; s2; s3; ...]
void ChElementShellANCF_3443_TR11S::Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact,
                                                         double xi,
                                                         double eta,
                                                         double zeta) {
    Sxi_zeta_compact(0) = 0.0;
    Sxi_zeta_compact(1) = 0.0;
    Sxi_zeta_compact(2) = 0.0;
    Sxi_zeta_compact(3) = 0.125 * m_thicknessZ * (xi - 1) * (eta - 1);

    Sxi_zeta_compact(4) = 0.0;
    Sxi_zeta_compact(5) = 0.0;
    Sxi_zeta_compact(6) = 0.0;
    Sxi_zeta_compact(7) = -0.125 * m_thicknessZ * (xi + 1) * (eta - 1);

    Sxi_zeta_compact(8) = 0.0;
    Sxi_zeta_compact(9) = 0.0;
    Sxi_zeta_compact(10) = 0.0;
    Sxi_zeta_compact(11) = 0.125 * m_thicknessZ * (xi + 1) * (eta + 1);

    Sxi_zeta_compact(12) = 0.0;
    Sxi_zeta_compact(13) = 0.0;
    Sxi_zeta_compact(14) = 0.0;
    Sxi_zeta_compact(15) = -0.125 * m_thicknessZ * (xi - 1) * (eta + 1);
}

void ChElementShellANCF_3443_TR11S::Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = -0.125 * (eta - 1) * (eta * eta + eta + 3 * xi * xi - 3);
    Sxi_D(1, 0) = -0.0625 * m_lenX * (3 * xi + 1) * (xi - 1) * (eta - 1);
    Sxi_D(2, 0) = -0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1);
    Sxi_D(3, 0) = 0.125 * m_thicknessZ * zeta * (eta - 1);

    Sxi_D(4, 0) = 0.125 * (eta - 1) * (eta * eta + eta + 3 * xi * xi - 3);
    Sxi_D(5, 0) = -0.0625 * m_lenX * (xi + 1) * (3 * xi - 1) * (eta - 1);
    Sxi_D(6, 0) = 0.0625 * m_lenY * (eta + 1) * (eta - 1) * (eta - 1);
    Sxi_D(7, 0) = -0.125 * m_thicknessZ * zeta * (eta - 1);

    Sxi_D(8, 0) = -0.125 * (eta + 1) * (eta * eta - eta + 3 * xi * xi - 3);
    Sxi_D(9, 0) = 0.0625 * m_lenX * (xi + 1) * (3 * xi - 1) * (eta + 1);
    Sxi_D(10, 0) = 0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1);
    Sxi_D(11, 0) = 0.125 * m_thicknessZ * zeta * (eta + 1);

    Sxi_D(12, 0) = 0.125 * (eta + 1) * (eta * eta - eta + 3 * xi * xi - 3);
    Sxi_D(13, 0) = 0.0625 * m_lenX * (3 * xi + 1) * (xi - 1) * (eta + 1);
    Sxi_D(14, 0) = -0.0625 * m_lenY * (eta - 1) * (eta + 1) * (eta + 1);
    Sxi_D(15, 0) = -0.125 * m_thicknessZ * zeta * (eta + 1);

    Sxi_D(0, 1) = -0.125 * (xi - 1) * (3 * eta * eta + xi * xi + xi - 3);
    Sxi_D(1, 1) = -0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1);
    Sxi_D(2, 1) = -0.0625 * m_lenY * (3 * eta + 1) * (eta - 1) * (xi - 1);
    Sxi_D(3, 1) = 0.125 * m_thicknessZ * zeta * (xi - 1);

    Sxi_D(4, 1) = 0.125 * (xi + 1) * (3 * eta * eta + xi * xi - xi - 3);
    Sxi_D(5, 1) = -0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1);
    Sxi_D(6, 1) = 0.0625 * m_lenY * (3 * eta + 1) * (eta - 1) * (xi + 1);
    Sxi_D(7, 1) = -0.125 * m_thicknessZ * zeta * (xi + 1);

    Sxi_D(8, 1) = -0.125 * (xi + 1) * (3 * eta * eta + xi * xi - xi - 3);
    Sxi_D(9, 1) = 0.0625 * m_lenX * (xi - 1) * (xi + 1) * (xi + 1);
    Sxi_D(10, 1) = 0.0625 * m_lenY * (eta + 1) * (3 * eta - 1) * (xi + 1);
    Sxi_D(11, 1) = 0.125 * m_thicknessZ * zeta * (xi + 1);

    Sxi_D(12, 1) = 0.125 * (xi - 1) * (3 * eta * eta + xi * xi + xi - 3);
    Sxi_D(13, 1) = 0.0625 * m_lenX * (xi + 1) * (xi - 1) * (xi - 1);
    Sxi_D(14, 1) = -0.0625 * m_lenY * (eta + 1) * (3 * eta - 1) * (xi - 1);
    Sxi_D(15, 1) = -0.125 * m_thicknessZ * zeta * (xi - 1);

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.0;
    Sxi_D(3, 2) = 0.125 * m_thicknessZ * (xi - 1) * (eta - 1);

    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.0;
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = -0.125 * m_thicknessZ * (xi + 1) * (eta - 1);

    Sxi_D(8, 2) = 0.0;
    Sxi_D(9, 2) = 0.0;
    Sxi_D(10, 2) = 0.0;
    Sxi_D(11, 2) = 0.125 * m_thicknessZ * (xi + 1) * (eta + 1);

    Sxi_D(12, 2) = 0.0;
    Sxi_D(13, 2) = 0.0;
    Sxi_D(14, 2) = 0.0;
    Sxi_D(15, 2) = -0.125 * m_thicknessZ * (xi - 1) * (eta + 1);
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_TR11S::CalcCoordVector(Vector3N& e) {
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
}

void ChElementShellANCF_3443_TR11S::CalcCoordMatrix(Matrix3xN& ebar) {
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
}

void ChElementShellANCF_3443_TR11S::CalcCoordMatrix(MatrixNx3c& ebar) {
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
}

void ChElementShellANCF_3443_TR11S::CalcCoordDerivVector(Vector3N& edot) {
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
}

void ChElementShellANCF_3443_TR11S::CalcCoordDerivMatrix(Matrix3xN& ebardot) {
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
}

void ChElementShellANCF_3443_TR11S::CalcCoordDerivMatrix(MatrixNx3c& ebardot) {
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
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementShellANCF_3443_TR11S::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    MatrixNx3c Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementShellANCF_3443_TR11S::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------

void ChElementShellANCF_3443_TR11S::EvaluateSectionFrame(const double xi,
                                                        const double eta,
                                                        ChVector<>& point,
                                                        ChQuaternion<>& rot) {
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

void ChElementShellANCF_3443_TR11S::EvaluateSectionPoint(const double u, const double v, ChVector<>& point) {
    Matrix3x3N Sxi;
    Calc_Sxi(Sxi, u, v, 0);

    Vector3N e;
    CalcCoordVector(e);

    // r = Se
    point = Sxi * e;
}

void ChElementShellANCF_3443_TR11S::EvaluateSectionVelNorm(double U, double V, ChVector<>& Result) {
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
void ChElementShellANCF_3443_TR11S::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
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
}

// Gets all the DOFs packed in a single vector (velocity part).
void ChElementShellANCF_3443_TR11S::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
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
}

/// Increment all DOFs using a delta.
void ChElementShellANCF_3443_TR11S::LoadableStateIncrement(const unsigned int off_x,
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
void ChElementShellANCF_3443_TR11S::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U) coordinates of the centerline.
void ChElementShellANCF_3443_TR11S::ComputeNF(
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
void ChElementShellANCF_3443_TR11S::ComputeNF(
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
double ChElementShellANCF_3443_TR11S::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at (U) coordinates.
ChVector<> ChElementShellANCF_3443_TR11S::ComputeNormal(const double U, const double V) {
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
ChQuadratureTables static_tables_3443_TR11S(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementShellANCF_3443_TR11S::GetStaticGQTables() {
    return &static_tables_3443_TR11S;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialShellANCF_3443_TR11S methods
// ============================================================================

// Construct an isotropic material.
ChMaterialShellANCF_3443_TR11S::ChMaterialShellANCF_3443_TR11S(double rho,  // material density
                                                             double E,    // Young's modulus
                                                             double nu    // Poisson ratio
                                                             )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G));
}

// Construct a (possibly) orthotropic material.
ChMaterialShellANCF_3443_TR11S::ChMaterialShellANCF_3443_TR11S(
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
void ChMaterialShellANCF_3443_TR11S::Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G) {
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
