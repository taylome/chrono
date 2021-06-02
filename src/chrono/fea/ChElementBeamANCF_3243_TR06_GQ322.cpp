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
// Fully Parameterized ANCF beam element with 2 nodes. Description of this element
// and its internal forces may be found in
// =============================================================================
// Internal Force Calculation Method is based on:  Gerstmayr, J., Shabana, A.A.:
// Efficient integration of the elastic forces and thin three-dimensional beam
// elements in the absolute nodal coordinate formulation.In: Proceedings of the
// Multibody Dynamics Eccomas thematic Conference, Madrid(2005)
// =============================================================================
// TR06_GQ322 = a Gerstmayr style implementation of the element with pre-calculation
//     of the terms needed for the generalized internal force calculation with
//     an analytical Jacobian that is integrated across GQ points one at a time
//
//  Mass Matrix = Constant, pre-calculated 8x8 matrix
//
//  Generalized Force due to gravity = Constant 24x1 Vector
//     (assumption that gravity is constant too)
//
//  Generalized Internal Force Vector = Calculated using the Gerstmayr method:
//     Dense Math: e_bar = 3x8 and S_bar = 8x1
//     Math is a translation from the method presented by Gerstmayr and Shabana
//     Reduced Number of GQ Integration Points (3x2x2)
//     GQ integration is performed one GQ point at a time
//     Pre-calculation of terms for the generalized internal force calculation
//
//  Jacobian of the Generalized Internal Force Vector = Analytical Jacobian that
//     is integrated across GQ points one at a time
//
// =============================================================================

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementBeamANCF_3243_TR06_GQ322.h"
#include <cmath>

namespace chrono {
namespace fea {

// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBeamANCF_3243_TR06_GQ322::ChElementBeamANCF_3243_TR06_GQ322()
    : m_gravity_on(false), m_thicknessY(0), m_thicknessZ(0), m_lenX(0), m_Alpha(0), m_damping_enabled(false) {
    m_nodes.resize(2);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBeamANCF_3243_TR06_GQ322::SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA,
                                           std::shared_ptr<ChNodeFEAxyzDDD> nodeB) {
    assert(nodeA);
    assert(nodeB);

    m_nodes[0] = nodeA;
    m_nodes[1] = nodeB;

    std::vector<ChVariables*> mvars;
    mvars.push_back(&m_nodes[0]->Variables());
    mvars.push_back(&m_nodes[0]->Variables_D());
    mvars.push_back(&m_nodes[0]->Variables_DD());
    mvars.push_back(&m_nodes[0]->Variables_DDD());
    mvars.push_back(&m_nodes[1]->Variables());
    mvars.push_back(&m_nodes[1]->Variables_D());
    mvars.push_back(&m_nodes[1]->Variables_DD());
    mvars.push_back(&m_nodes[1]->Variables_DDD());

    Kmatr.SetVariables(mvars);
}

// -----------------------------------------------------------------------------
// Interface to ChElementBase base class
// -----------------------------------------------------------------------------

// Initial element setup.
void ChElementBeamANCF_3243_TR06_GQ322::SetupInitial(ChSystem* system) {
    // Initial positions and slopes of the element nodes
    // These values define the reference configuration of the element
    CalcCoordMatrix(m_ebar0);

    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());

    // Compute Pre-computed matrices and vectors for the generalized internal force calcualtions
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementBeamANCF_3243_TR06_GQ322::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementBeamANCF_3243_TR06_GQ322::GetStateBlock(ChVectorDynamic<>& mD) {
    mD.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(9, 3) = m_nodes[0]->GetDDD().eigen();
    mD.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(21, 3) = m_nodes[1]->GetDDD().eigen();
}

// Calculate the global matrix H as a linear combination of K, R, and M:
//   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R]
void ChElementBeamANCF_3243_TR06_GQ322::ComputeKRMmatricesGlobal(ChMatrixRef H,
                                                           double Kfactor,
                                                           double Rfactor,
                                                           double Mfactor) {
    assert((H.rows() == 24) && (H.cols() == 24));

#if true  // Analytical Jacobian
    if (m_damping_enabled) {  // If linear Kelvin-Voigt viscoelastic material model is enabled
        ComputeInternalJacobianDamping(H, -Kfactor, -Rfactor, Mfactor);
    } else {
        ComputeInternalJacobianNoDamping(H, -Kfactor, Mfactor);
    }
#else  // Numeric Jacobian
    // Calculate the linear combination Kfactor*[K] + Rfactor*[R]
    ChMatrixNM<double, 24, 24> JacobianMatrix;
    ComputeInternalJacobians(JacobianMatrix, Kfactor, Rfactor);

    // Load Jac + Mfactor*[M] into H
    H = JacobianMatrix + Mfactor * m_MassMatrix;
#endif
}

// Return the mass matrix.
void ChElementBeamANCF_3243_TR06_GQ322::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    // Inflate the Mass Matrix since it is stored in compact form.
    // In MATLAB notation:
    // M(1:3:end,1:3:end) = m_MassMatrix;
    // M(2:3:end,2:3:end) = m_MassMatrix;
    // M(3:3:end,3:3:end) = m_MassMatrix;
    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++) {
            M(3 * i, 3 * j) = m_MassMatrix(i, j);
            M(3 * i + 1, 3 * j + 1) = m_MassMatrix(i, j);
            M(3 * i + 2, 3 * j + 2) = m_MassMatrix(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Mass Matrix & Generalized Force Due to Gravity Calculation
// -----------------------------------------------------------------------------
void ChElementBeamANCF_3243_TR06_GQ322::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
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
                ChMatrixNM<double, 3, 24> Sxi;                     // 3x24 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                ChVectorN<double, 8> Sxi_compact;  // 8x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight * rho * det_J_0xi) * Sxi.transpose() * g_acc.eigen();
                m_MassMatrix += (GQ_weight * rho * det_J_0xi) * Sxi_compact * Sxi_compact.transpose();
            }
        }
    }
}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementBeamANCF_3243_TR06_GQ322::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0, 0) + m_MassMatrix(0, 12);
    m_nodes[1]->m_TotalMass += m_MassMatrix(12, 12) + m_MassMatrix(12, 0);
}

// Precalculate constant matrices and scalars for the internal force calculations
void ChElementBeamANCF_3243_TR06_GQ322::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 2;        // 3 Point Gauss-Quadrature;
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
                ChMatrixNMc<double, 8, 3> Sxi_D;  // Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_ebar0 * Sxi_D;

                m_SD_precompute_D0.block(0, 3 * index, 8, 3) = Sxi_D * J_0xi.inverse();
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
        ChMatrixNMc<double, 8, 3> Sxi_D;  // Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, 0, 0);
        J_0xi.noalias() = m_ebar0 * Sxi_D;

        m_SD_precompute_Dv.block(0, 3 * it_xi, 8, 3) = Sxi_D * J_0xi.inverse();
        m_GQWeight_det_J_0xi_Dv(it_xi) = -J_0xi.determinant() * GQ_weight;
    }
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementBeamANCF_3243_TR06_GQ322::SetAlphaDamp(double a) {
    m_Alpha = a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementBeamANCF_3243_TR06_GQ322::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    ChMatrixNM<double, 3, 8> ebar;
    ChMatrixNM<double, 3, 8> ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);
    ComputeInternalForcesAtState(Fi, ebar, ebardot);

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementBeamANCF_3243_TR06_GQ322::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi,
                                                               const ChMatrixNM<double, 3, 8>& ebar,
                                                               const ChMatrixNM<double, 3, 8>& ebardot) {
    ChMatrixNM<double, 8, 3> QiCompact;
    QiCompact.setZero();

    // Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();
    for (unsigned int GQpnt = 0; GQpnt < 12; GQpnt++) {
        ChMatrixNMc<double, 8, 3> Sbar_xi_D = m_SD_precompute_D0.block<8, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
        ChVectorN<double, 6> epsilon_combined;
        epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
        epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
        epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);
        epsilon_combined(3) = F.col(1).dot(F.col(2));
        epsilon_combined(4) = F.col(0).dot(F.col(2));
        epsilon_combined(5) = F.col(0).dot(F.col(1));

        if (m_damping_enabled) {
            // Calculate the time derivative of the Deformation Gradient at the current point
            ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

            // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
            // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
            epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
            epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
            epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
            epsilon_combined(3) += m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
            epsilon_combined(4) += m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
            epsilon_combined(5) += m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));
        }
        epsilon_combined = epsilon_combined.cwiseProduct(D0) * m_GQWeight_det_J_0xi_D0(GQpnt);

        ChMatrixNM<double, 3, 3> SPK2;  // 2nd Piola Kirchhoff Stress tensor
        SPK2(0, 0) = epsilon_combined(0);
        SPK2(1, 1) = epsilon_combined(1);
        SPK2(2, 2) = epsilon_combined(2);
        SPK2(1, 2) = epsilon_combined(3);
        SPK2(2, 1) = epsilon_combined(3);
        SPK2(0, 2) = epsilon_combined(4);
        SPK2(2, 0) = epsilon_combined(4);
        SPK2(0, 1) = epsilon_combined(5);
        SPK2(1, 0) = epsilon_combined(5);

        // Calculate the transpose of the (1st Piola Kirchhoff Stress tensor = F*SPK2) scaled by the negative of the
        // determinate of the element Jacobian.  Note that SPK2 is symmetric.
        ChMatrixNM<double, 3, 3> P_transpose_scaled = SPK2 * F.transpose();
        QiCompact += Sbar_xi_D * P_transpose_scaled;
    }

    // Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the
    // beam axis
    if (GetStrainFormulation() == ChElementBeamANCF_3243_TR06_GQ322::StrainFormulation::CMPoisson) {
        const ChMatrix33<double>& Dv = GetMaterial()->Get_Dv();

        for (unsigned int GQpnt = 0; GQpnt < 3; GQpnt++) {
            ChMatrixNMc<double, 8, 3> Sbar_xi_D = m_SD_precompute_Dv.block<8, 3>(0, 3 * GQpnt);

            // Calculate the Deformation Gradient at the current point
            ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

            // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation (due to the material,
            // only diagonal terms are needed)
            ChVectorN<double, 3> epsilon_combined;
            epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
            epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
            epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);

            if (m_damping_enabled) {
                // Calculate the time derivative of the Deformation Gradient at the current point
                ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

                // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
                // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
                // (due to the material, only need the first 3 rows are needed)
                epsilon_combined(0) += m_Alpha * F.col(0).dot(Fdot.col(0));
                epsilon_combined(1) += m_Alpha * F.col(1).dot(Fdot.col(1));
                epsilon_combined(2) += m_Alpha * F.col(2).dot(Fdot.col(2));
            }
            epsilon_combined = Dv * (epsilon_combined * m_GQWeight_det_J_0xi_Dv(GQpnt));

            // Calculate the transpose of the (1st Piola Kirchhoff Stress tensor = F*SPK2) scaled by the negative of the
            // determinate of the element Jacobian.  Note that SPK2 is symmetric.
            ChMatrixNM<double, 3, 3> P_transpose_scaled = epsilon_combined.asDiagonal() * F.transpose();
            QiCompact += Sbar_xi_D * P_transpose_scaled;
        }
    }

    Eigen::Map<ChVectorN<double, 24>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3243_TR06_GQ322::ComputeInternalJacobians(ChMatrixNM<double, 24, 24>& JacobianMatrix,
                                                           double Kfactor,
                                                           double Rfactor) {
    // The integrated quantity represents the 24x24 Jacobian
    //      Kfactor * [K] + Rfactor * [R]
    // Note that the matrices with current nodal coordinates and velocities are
    // already available in m_d and m_d_dt (as set in ComputeInternalForces).
    // Similarly, the ANS strain and strain derivatives are already available in
    // m_strainANS and m_strainANS_D (as calculated in ComputeInternalForces).

    ChVectorDynamic<double> FiOrignal(24);
    ChVectorDynamic<double> FiDelta(24);

    ChMatrixNM<double, 3, 8> ebar;
    ChMatrixNM<double, 3, 8> ebardot;

    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);

    double delta = 1e-6;

    // Compute the Jacobian via numerical differentiation of the generalized internal force vector
    // Since the generalized force vector due to gravity is a constant, it doesn't affect this
    // Jacobian calculation
    ComputeInternalForcesAtState(FiOrignal, ebar, ebardot);
    for (unsigned int i = 0; i < 24; i++) {
        ebar(i % 3, i / 3) += delta;
        ComputeInternalForcesAtState(FiDelta, ebar, ebardot);
        JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
        ebar(i % 3, i / 3) -= delta;

        if (m_damping_enabled) {
            ebardot(i % 3, i / 3) += delta;
            ComputeInternalForcesAtState(FiDelta, ebar, ebardot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            ebardot(i % 3, i / 3) -= delta;
        }
    }
}

void ChElementBeamANCF_3243_TR06_GQ322::ComputeInternalJacobianDamping(ChMatrixRef& H,
                                                                 double Kfactor,
                                                                 double Rfactor,
                                                                 double Mfactor) {
    H.setZero();

    ChMatrixNM<double, 8, 8> Jacobian_CompactPart = Mfactor * m_MassMatrix;

    ChMatrixNM<double, 8, 3> PartialEpsilon0Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon0(PartialEpsilon0Compact.data(), PartialEpsilon0Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon1Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon1(PartialEpsilon1Compact.data(), PartialEpsilon1Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon2Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon2(PartialEpsilon2Compact.data(), PartialEpsilon2Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon3Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon3(PartialEpsilon3Compact.data(), PartialEpsilon3Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon4Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon4(PartialEpsilon4Compact.data(), PartialEpsilon4Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon5Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon5(PartialEpsilon5Compact.data(), PartialEpsilon5Compact.size());

    ChMatrixNM<double, 3, 8> ebar;
    ChMatrixNM<double, 3, 8> ebardot;
    CalcCoordMatrix(ebar);
    CalcCoordDerivMatrix(ebardot);

    // Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();
    for (unsigned int GQpnt = 0; GQpnt < 12; GQpnt++) {
        ChMatrixNMc<double, 8, 3> Sbar_xi_D = m_SD_precompute_D0.block<8, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        // Calculate the time derivative of the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

        // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
        // and calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model

        ChVectorN<double, 6> epsilon_combined;
        epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1) + m_Alpha * F.col(0).dot(Fdot.col(0));
        epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1) + m_Alpha * F.col(1).dot(Fdot.col(1));
        epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1) + m_Alpha * F.col(2).dot(Fdot.col(2));
        epsilon_combined(3) =
            F.col(1).dot(F.col(2)) + m_Alpha * (F.col(1).dot(Fdot.col(2)) + Fdot.col(1).dot(F.col(2)));
        epsilon_combined(4) =
            F.col(0).dot(F.col(2)) + m_Alpha * (F.col(0).dot(Fdot.col(2)) + Fdot.col(0).dot(F.col(2)));
        epsilon_combined(5) =
            F.col(0).dot(F.col(1)) + m_Alpha * (F.col(0).dot(Fdot.col(1)) + Fdot.col(0).dot(F.col(1)));

        // Multiply by D0 to get the 2nd Piola Kirchoff Stress Tensor in Voigt notation (this is no longer strain)
        epsilon_combined = epsilon_combined.cwiseProduct(D0) * m_GQWeight_det_J_0xi_D0(GQpnt);

        ChMatrixNM<double, 3, 3> SPK2;  // 2nd Piola Kirchhoff Stress tensor
        SPK2(0, 0) = epsilon_combined(0);
        SPK2(1, 1) = epsilon_combined(1);
        SPK2(2, 2) = epsilon_combined(2);
        SPK2(1, 2) = epsilon_combined(3);
        SPK2(2, 1) = epsilon_combined(3);
        SPK2(0, 2) = epsilon_combined(4);
        SPK2(2, 0) = epsilon_combined(4);
        SPK2(0, 1) = epsilon_combined(5);
        SPK2(1, 0) = epsilon_combined(5);

        Jacobian_CompactPart += Kfactor * Sbar_xi_D * SPK2 * Sbar_xi_D.transpose();

        PartialEpsilon0Compact = Sbar_xi_D.col(0) * F.col(0).transpose();
        PartialEpsilon1Compact = Sbar_xi_D.col(1) * F.col(1).transpose();
        PartialEpsilon2Compact = Sbar_xi_D.col(2) * F.col(2).transpose();
        PartialEpsilon3Compact = Sbar_xi_D.col(2) * F.col(1).transpose() + Sbar_xi_D.col(1) * F.col(2).transpose();
        PartialEpsilon4Compact = Sbar_xi_D.col(2) * F.col(0).transpose() + Sbar_xi_D.col(0) * F.col(2).transpose();
        PartialEpsilon5Compact = Sbar_xi_D.col(1) * F.col(0).transpose() + Sbar_xi_D.col(0) * F.col(1).transpose();

        ChMatrixNM<double, 6, 24> PartialEpsilon;
        PartialEpsilon.row(0) = PartialEpsilon0;
        PartialEpsilon.row(1) = PartialEpsilon1;
        PartialEpsilon.row(2) = PartialEpsilon2;
        PartialEpsilon.row(3) = PartialEpsilon3;
        PartialEpsilon.row(4) = PartialEpsilon4;
        PartialEpsilon.row(5) = PartialEpsilon5;

        PartialEpsilon0Compact =
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(0) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon0Compact +
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(0) * Kfactor * m_Alpha) * Sbar_xi_D.col(0) * Fdot.col(0).transpose();
        PartialEpsilon1Compact =
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(1) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon1Compact +
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(1) * Kfactor * m_Alpha) * Sbar_xi_D.col(1) * Fdot.col(1).transpose();
        PartialEpsilon2Compact =
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(2) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon2Compact +
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(2) * Kfactor * m_Alpha) * Sbar_xi_D.col(2) * Fdot.col(2).transpose();
        PartialEpsilon3Compact =
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(3) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon3Compact +
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(3) * Kfactor * m_Alpha) *
                (Sbar_xi_D.col(2) * Fdot.col(1).transpose() + Sbar_xi_D.col(1) * Fdot.col(2).transpose());
        PartialEpsilon4Compact =
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(4) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon4Compact +
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(4) * Kfactor * m_Alpha) *
                (Sbar_xi_D.col(2) * Fdot.col(0).transpose() + Sbar_xi_D.col(0) * Fdot.col(2).transpose());
        PartialEpsilon5Compact =
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(5) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon5Compact +
            (m_GQWeight_det_J_0xi_D0(GQpnt) * D0(5) * Kfactor * m_Alpha) *
                (Sbar_xi_D.col(1) * Fdot.col(0).transpose() + Sbar_xi_D.col(0) * Fdot.col(1).transpose());

        ChMatrixNM<double, 6, 24> PartialEpsilonCombined;
        PartialEpsilonCombined.row(0) = PartialEpsilon0;
        PartialEpsilonCombined.row(1) = PartialEpsilon1;
        PartialEpsilonCombined.row(2) = PartialEpsilon2;
        PartialEpsilonCombined.row(3) = PartialEpsilon3;
        PartialEpsilonCombined.row(4) = PartialEpsilon4;
        PartialEpsilonCombined.row(5) = PartialEpsilon5;

        H += PartialEpsilon.transpose() * PartialEpsilonCombined;
    }

    // Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the
    // beam axis
    if (GetStrainFormulation() == ChElementBeamANCF_3243_TR06_GQ322::StrainFormulation::CMPoisson) {
        const ChMatrix33<double>& Dv = GetMaterial()->Get_Dv();

        for (unsigned int GQpnt = 0; GQpnt < 3; GQpnt++) {
            ChMatrixNMc<double, 8, 3> Sbar_xi_D = m_SD_precompute_Dv.block<8, 3>(0, 3 * GQpnt);

            // Calculate the Deformation Gradient at the current point
            ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

            // Calculate the time derivative of the Deformation Gradient at the current point
            ChMatrixNMc<double, 3, 3> Fdot = ebardot * Sbar_xi_D;

            // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation (due to the material,
            // only diagonal terms are needed) and
            // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
            // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
            ChVectorN<double, 3> epsilon_combined;
            epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1) + m_Alpha * F.col(0).dot(Fdot.col(0));
            epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1) + m_Alpha * F.col(1).dot(Fdot.col(1));
            epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1) + m_Alpha * F.col(2).dot(Fdot.col(2));

            // Multiply by Dv to get the 2nd Piola Kirchoff Stress Tensor (this is no longer strain)
            epsilon_combined = Dv * (Kfactor * m_GQWeight_det_J_0xi_Dv(GQpnt) * epsilon_combined);

            Jacobian_CompactPart += Sbar_xi_D * epsilon_combined.asDiagonal() * Sbar_xi_D.transpose();

            PartialEpsilon0Compact = Sbar_xi_D.col(0) * F.col(0).transpose();
            PartialEpsilon1Compact = Sbar_xi_D.col(1) * F.col(1).transpose();
            PartialEpsilon2Compact = Sbar_xi_D.col(2) * F.col(2).transpose();

            ChMatrixNM<double, 3, 24> PartialEpsilon;
            PartialEpsilon.row(0) = PartialEpsilon0;
            PartialEpsilon.row(1) = PartialEpsilon1;
            PartialEpsilon.row(2) = PartialEpsilon2;

            PartialEpsilon0Compact =
                (m_GQWeight_det_J_0xi_Dv(GQpnt) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon0Compact +
                (m_GQWeight_det_J_0xi_Dv(GQpnt) * Kfactor * m_Alpha) * Sbar_xi_D.col(0) * Fdot.col(0).transpose();
            PartialEpsilon1Compact =
                (m_GQWeight_det_J_0xi_Dv(GQpnt) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon1Compact +
                (m_GQWeight_det_J_0xi_Dv(GQpnt) * Kfactor * m_Alpha) * Sbar_xi_D.col(1) * Fdot.col(1).transpose();
            PartialEpsilon2Compact =
                (m_GQWeight_det_J_0xi_Dv(GQpnt) * (Kfactor + m_Alpha * Rfactor)) * PartialEpsilon2Compact +
                (m_GQWeight_det_J_0xi_Dv(GQpnt) * Kfactor * m_Alpha) * Sbar_xi_D.col(2) * Fdot.col(2).transpose();

            ChMatrixNM<double, 3, 24> PartialEpsilonCombined;
            PartialEpsilonCombined.row(0) = PartialEpsilon0;
            PartialEpsilonCombined.row(1) = PartialEpsilon1;
            PartialEpsilonCombined.row(2) = PartialEpsilon2;

            H += PartialEpsilon.transpose() * Dv * PartialEpsilonCombined;
        }
    }

    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++) {
            H(3 * i, 3 * j) += Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j + 1) += Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 2) += Jacobian_CompactPart(i, j);
        }
    }
}

void ChElementBeamANCF_3243_TR06_GQ322::ComputeInternalJacobianNoDamping(ChMatrixRef& H, double Kfactor, double Mfactor) {
    H.setZero();

    ChMatrixNM<double, 8, 8> Jacobian_CompactPart = Mfactor * m_MassMatrix;

    ChMatrixNM<double, 8, 3> PartialEpsilon0Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon0(PartialEpsilon0Compact.data(), PartialEpsilon0Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon1Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon1(PartialEpsilon1Compact.data(), PartialEpsilon1Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon2Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon2(PartialEpsilon2Compact.data(), PartialEpsilon2Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon3Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon3(PartialEpsilon3Compact.data(), PartialEpsilon3Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon4Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon4(PartialEpsilon4Compact.data(), PartialEpsilon4Compact.size());
    ChMatrixNM<double, 8, 3> PartialEpsilon5Compact;
    Eigen::Map<ChVectorN<double, 24>> PartialEpsilon5(PartialEpsilon5Compact.data(), PartialEpsilon5Compact.size());

    ChMatrixNM<double, 3, 8> ebar;
    CalcCoordMatrix(ebar);

    // Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    const ChVectorN<double, 6>& D0 = GetMaterial()->Get_D0();
    for (unsigned int GQpnt = 0; GQpnt < 12; GQpnt++) {
        ChMatrixNMc<double, 8, 3> Sbar_xi_D = m_SD_precompute_D0.block<8, 3>(0, 3 * GQpnt);

        // Calculate the Deformation Gradient at the current point
        ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

        // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation
        // and calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
        // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model

        ChVectorN<double, 6> epsilon_combined;
        epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
        epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
        epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);
        epsilon_combined(3) = F.col(1).dot(F.col(2));
        epsilon_combined(4) = F.col(0).dot(F.col(2));
        epsilon_combined(5) = F.col(0).dot(F.col(1));

        // Multiply by D0 to get the 2nd Piola Kirchoff Stress Tensor in Voigt notation (this is no longer strain)
        epsilon_combined = epsilon_combined.cwiseProduct(D0) * m_GQWeight_det_J_0xi_D0(GQpnt);

        ChMatrixNM<double, 3, 3> SPK2;  // 2nd Piola Kirchhoff Stress tensor
        SPK2(0, 0) = epsilon_combined(0);
        SPK2(1, 1) = epsilon_combined(1);
        SPK2(2, 2) = epsilon_combined(2);
        SPK2(1, 2) = epsilon_combined(3);
        SPK2(2, 1) = epsilon_combined(3);
        SPK2(0, 2) = epsilon_combined(4);
        SPK2(2, 0) = epsilon_combined(4);
        SPK2(0, 1) = epsilon_combined(5);
        SPK2(1, 0) = epsilon_combined(5);

        Jacobian_CompactPart += Kfactor * Sbar_xi_D * SPK2 * Sbar_xi_D.transpose();

        PartialEpsilon0Compact = Sbar_xi_D.col(0) * F.col(0).transpose();
        PartialEpsilon1Compact = Sbar_xi_D.col(1) * F.col(1).transpose();
        PartialEpsilon2Compact = Sbar_xi_D.col(2) * F.col(2).transpose();
        PartialEpsilon3Compact = Sbar_xi_D.col(2) * F.col(1).transpose() + Sbar_xi_D.col(1) * F.col(2).transpose();
        PartialEpsilon4Compact = Sbar_xi_D.col(2) * F.col(0).transpose() + Sbar_xi_D.col(0) * F.col(2).transpose();
        PartialEpsilon5Compact = Sbar_xi_D.col(1) * F.col(0).transpose() + Sbar_xi_D.col(0) * F.col(1).transpose();

        ChMatrixNM<double, 6, 24> PartialEpsilon;
        PartialEpsilon.row(0) = PartialEpsilon0;
        PartialEpsilon.row(1) = PartialEpsilon1;
        PartialEpsilon.row(2) = PartialEpsilon2;
        PartialEpsilon.row(3) = PartialEpsilon3;
        PartialEpsilon.row(4) = PartialEpsilon4;
        PartialEpsilon.row(5) = PartialEpsilon5;

        PartialEpsilon0Compact *= (Kfactor * m_GQWeight_det_J_0xi_D0(GQpnt) * D0(0));
        PartialEpsilon1Compact *= (Kfactor * m_GQWeight_det_J_0xi_D0(GQpnt) * D0(1));
        PartialEpsilon2Compact *= (Kfactor * m_GQWeight_det_J_0xi_D0(GQpnt) * D0(2));
        PartialEpsilon3Compact *= (Kfactor * m_GQWeight_det_J_0xi_D0(GQpnt) * D0(3));
        PartialEpsilon4Compact *= (Kfactor * m_GQWeight_det_J_0xi_D0(GQpnt) * D0(4));
        PartialEpsilon5Compact *= (Kfactor * m_GQWeight_det_J_0xi_D0(GQpnt) * D0(5));

        ChMatrixNM<double, 6, 24> PartialEpsilonCombined;
        PartialEpsilonCombined.row(0) = PartialEpsilon0;
        PartialEpsilonCombined.row(1) = PartialEpsilon1;
        PartialEpsilonCombined.row(2) = PartialEpsilon2;
        PartialEpsilonCombined.row(3) = PartialEpsilon3;
        PartialEpsilonCombined.row(4) = PartialEpsilon4;
        PartialEpsilonCombined.row(5) = PartialEpsilon5;

        H += PartialEpsilon.transpose() * PartialEpsilonCombined;
    }

    // Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the
    // beam axis
    if (GetStrainFormulation() == ChElementBeamANCF_3243_TR06_GQ322::StrainFormulation::CMPoisson) {
        const ChMatrix33<double>& Dv = GetMaterial()->Get_Dv();

        for (unsigned int GQpnt = 0; GQpnt < 3; GQpnt++) {
            ChMatrixNMc<double, 8, 3> Sbar_xi_D = m_SD_precompute_Dv.block<8, 3>(0, 3 * GQpnt);

            // Calculate the Deformation Gradient at the current point
            ChMatrixNMc<double, 3, 3> F = ebar * Sbar_xi_D;

            // Calculate the Green-Lagrange strain tensor at the current point in Voigt notation (due to the material,
            // only diagonal terms are needed) and
            // Calculate the time derivative of the Green-Lagrange strain tensor in Voigt notation
            // and combine it with epsilon assuming a Linear Kelvin-Voigt Viscoelastic material model
            ChVectorN<double, 3> epsilon_combined;
            epsilon_combined(0) = 0.5 * (F.col(0).dot(F.col(0)) - 1);
            epsilon_combined(1) = 0.5 * (F.col(1).dot(F.col(1)) - 1);
            epsilon_combined(2) = 0.5 * (F.col(2).dot(F.col(2)) - 1);

            // Multiply by Dv to get the 2nd Piola Kirchoff Stress Tensor (this is no longer strain)
            epsilon_combined = Dv * (Kfactor * epsilon_combined * m_GQWeight_det_J_0xi_Dv(GQpnt));

            Jacobian_CompactPart += Sbar_xi_D * epsilon_combined.asDiagonal() * Sbar_xi_D.transpose();

            PartialEpsilon0Compact = Sbar_xi_D.col(0) * F.col(0).transpose();
            PartialEpsilon1Compact = Sbar_xi_D.col(1) * F.col(1).transpose();
            PartialEpsilon2Compact = Sbar_xi_D.col(2) * F.col(2).transpose();

            ChMatrixNM<double, 3, 24> PartialEpsilon;
            PartialEpsilon.row(0) = PartialEpsilon0;
            PartialEpsilon.row(1) = PartialEpsilon1;
            PartialEpsilon.row(2) = PartialEpsilon2;

            H += PartialEpsilon.transpose() * ((Kfactor * m_GQWeight_det_J_0xi_Dv(GQpnt)) * Dv) * PartialEpsilon;
        }
    }

    for (unsigned int i = 0; i < 8; i++) {
        for (unsigned int j = 0; j < 8; j++) {
            H(3 * i, 3 * j) += Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j + 1) += Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 2) += Jacobian_CompactPart(i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x24 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi(ChMatrixNM<double, 3, 24>& Sxi, double xi, double eta, double zeta) {
    ChVectorN<double, 8> Sxi_compact;
    Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);
    Sxi.setZero();

    for (unsigned int s = 0; s < Sxi_compact.size(); s++) {
        Sxi(0, 0 + (3 * s)) = Sxi_compact(s);
        Sxi(1, 1 + (3 * s)) = Sxi_compact(s);
        Sxi(2, 2 + (3 * s)) = Sxi_compact(s);
    }
}

// 8x1 Vector Form of the Normalized Shape Functions
// [s1; s2; s3; ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_compact(ChVectorN<double, 8>& Sxi_compact,
                                                   double xi,
                                                   double eta,
                                                   double zeta) {
    Sxi_compact(0) = 0.25 * (xi * xi * xi - 3 * xi + 2);
    Sxi_compact(1) = 0.125 * m_lenX * (xi * xi * xi - xi * xi - xi + 1);
    Sxi_compact(2) = 0.25 * m_thicknessY * eta * (1 - xi);
    Sxi_compact(3) = 0.25 * m_thicknessZ * zeta * (1 - xi);
    Sxi_compact(4) = 0.25 * (-xi * xi * xi + 3 * xi + 2);
    Sxi_compact(5) = 0.125 * m_lenX * (xi * xi * xi + xi * xi - xi - 1);
    Sxi_compact(6) = 0.25 * m_thicknessY * eta * (1 + xi);
    Sxi_compact(7) = 0.25 * m_thicknessZ * zeta * (1 + xi);
}

// 3x24 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_xi(ChMatrixNM<double, 3, 24>& Sxi_xi, double xi, double eta, double zeta) {
    ChVectorN<double, 8> Sxi_xi_compact;
    Calc_Sxi_xi_compact(Sxi_xi_compact, xi, eta, zeta);
    Sxi_xi.setZero();

    for (unsigned int s = 0; s < Sxi_xi_compact.size(); s++) {
        Sxi_xi(0, 0 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(1, 1 + (3 * s)) = Sxi_xi_compact(s);
        Sxi_xi(2, 2 + (3 * s)) = Sxi_xi_compact(s);
    }
}

// 8x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to xi
// [s1; s2; s3; ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_xi_compact(ChVectorN<double, 8>& Sxi_xi_compact,
                                                      double xi,
                                                      double eta,
                                                      double zeta) {
    Sxi_xi_compact(0) = 0.75 * (xi * xi - 1);
    Sxi_xi_compact(1) = 0.125 * m_lenX * (3 * xi * xi - 2 * xi - 1);
    Sxi_xi_compact(2) = -0.25 * m_thicknessY * eta;
    Sxi_xi_compact(3) = -0.25 * m_thicknessZ * zeta;
    Sxi_xi_compact(4) = 0.75 * (-xi * xi + 1);
    Sxi_xi_compact(5) = 0.125 * m_lenX * (3 * xi * xi + 2 * xi - 1);
    Sxi_xi_compact(6) = 0.25 * m_thicknessY * eta;
    Sxi_xi_compact(7) = 0.25 * m_thicknessZ * zeta;
}

// 3x24 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_eta(ChMatrixNM<double, 3, 24>& Sxi_eta, double xi, double eta, double zeta) {
    ChVectorN<double, 8> Sxi_eta_compact;
    Calc_Sxi_eta_compact(Sxi_eta_compact, xi, eta, zeta);
    Sxi_eta.setZero();

    for (unsigned int s = 0; s < Sxi_eta_compact.size(); s++) {
        Sxi_eta(0, 0 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(1, 1 + (3 * s)) = Sxi_eta_compact(s);
        Sxi_eta(2, 2 + (3 * s)) = Sxi_eta_compact(s);
    }
}

// 8x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_eta_compact(ChVectorN<double, 8>& Sxi_eta_compact,
                                                       double xi,
                                                       double eta,
                                                       double zeta) {
    Sxi_eta_compact(0) = 0.0;
    Sxi_eta_compact(1) = 0.0;
    Sxi_eta_compact(2) = 0.25 * m_thicknessY * (-xi + 1);
    Sxi_eta_compact(3) = 0.0;
    Sxi_eta_compact(4) = 0.0;
    Sxi_eta_compact(5) = 0.0;
    Sxi_eta_compact(6) = 0.25 * m_thicknessY * (xi + 1);
    Sxi_eta_compact(7) = 0.0;
}

// 3x24 Sparse Form of the partial derivatives of Normalized Shape Functions with respect to zeta
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_zeta(ChMatrixNM<double, 3, 24>& Sxi_zeta,
                                                double xi,
                                                double eta,
                                                double zeta) {
    ChVectorN<double, 8> Sxi_zeta_compact;
    Calc_Sxi_zeta_compact(Sxi_zeta_compact, xi, eta, zeta);
    Sxi_zeta.setZero();

    for (unsigned int s = 0; s < Sxi_zeta_compact.size(); s++) {
        Sxi_zeta(0, 0 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(1, 1 + (3 * s)) = Sxi_zeta_compact(s);
        Sxi_zeta(2, 2 + (3 * s)) = Sxi_zeta_compact(s);
    }
}

// 8x1 Vector Form of the partial derivatives of Normalized Shape Functions with respect to eta
// [s1; s2; s3; ...]
void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_zeta_compact(ChVectorN<double, 8>& Sxi_zeta_compact,
                                                        double xi,
                                                        double eta,
                                                        double zeta) {
    Sxi_zeta_compact(0) = 0.0;
    Sxi_zeta_compact(1) = 0.0;
    Sxi_zeta_compact(2) = 0.0;
    Sxi_zeta_compact(3) = 0.25 * m_thicknessZ * (-xi + 1);
    Sxi_zeta_compact(4) = 0.0;
    Sxi_zeta_compact(5) = 0.0;
    Sxi_zeta_compact(6) = 0.0;
    Sxi_zeta_compact(7) = 0.25 * m_thicknessZ * (xi + 1);
}

void ChElementBeamANCF_3243_TR06_GQ322::Calc_Sxi_D(ChMatrixNMc<double, 8, 3>& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = 0.75 * (xi * xi - 1);
    Sxi_D(1, 0) = 0.125 * m_lenX * (3 * xi * xi - 2 * xi - 1);
    Sxi_D(2, 0) = -0.25 * m_thicknessY * eta;
    Sxi_D(3, 0) = -0.25 * m_thicknessZ * zeta;
    Sxi_D(4, 0) = 0.75 * (-xi * xi + 1);
    Sxi_D(5, 0) = 0.125 * m_lenX * (3 * xi * xi + 2 * xi - 1);
    Sxi_D(6, 0) = 0.25 * m_thicknessY * eta;
    Sxi_D(7, 0) = 0.25 * m_thicknessZ * zeta;

    Sxi_D(0, 1) = 0.0;
    Sxi_D(1, 1) = 0.0;
    Sxi_D(2, 1) = 0.25 * m_thicknessY * (-xi + 1);
    Sxi_D(3, 1) = 0.0;
    Sxi_D(4, 1) = 0.0;
    Sxi_D(5, 1) = 0.0;
    Sxi_D(6, 1) = 0.25 * m_thicknessY * (xi + 1);
    Sxi_D(7, 1) = 0.0;

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.0;
    Sxi_D(3, 2) = 0.25 * m_thicknessZ * (-xi + 1);
    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.0;
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = 0.25 * m_thicknessZ * (xi + 1);
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementBeamANCF_3243_TR06_GQ322::CalcCoordVector(ChVectorN<double, 24>& e) {
    e.segment(0, 3) = m_nodes[0]->GetPos().eigen();
    e.segment(3, 3) = m_nodes[0]->GetD().eigen();
    e.segment(6, 3) = m_nodes[0]->GetDD().eigen();
    e.segment(9, 3) = m_nodes[0]->GetDDD().eigen();

    e.segment(12, 3) = m_nodes[1]->GetPos().eigen();
    e.segment(15, 3) = m_nodes[1]->GetD().eigen();
    e.segment(18, 3) = m_nodes[1]->GetDD().eigen();
    e.segment(21, 3) = m_nodes[1]->GetDDD().eigen();
}

void ChElementBeamANCF_3243_TR06_GQ322::CalcCoordMatrix(ChMatrixNM<double, 3, 8>& ebar) {
    ebar.col(0) = m_nodes[0]->GetPos().eigen();
    ebar.col(1) = m_nodes[0]->GetD().eigen();
    ebar.col(2) = m_nodes[0]->GetDD().eigen();
    ebar.col(3) = m_nodes[0]->GetDDD().eigen();

    ebar.col(4) = m_nodes[1]->GetPos().eigen();
    ebar.col(5) = m_nodes[1]->GetD().eigen();
    ebar.col(6) = m_nodes[1]->GetDD().eigen();
    ebar.col(7) = m_nodes[1]->GetDDD().eigen();
}

void ChElementBeamANCF_3243_TR06_GQ322::CalcCoordDerivVector(ChVectorN<double, 24>& edot) {
    edot.segment(0, 3) = m_nodes[0]->GetPos_dt().eigen();
    edot.segment(3, 3) = m_nodes[0]->GetD_dt().eigen();
    edot.segment(6, 3) = m_nodes[0]->GetDD_dt().eigen();
    edot.segment(9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    edot.segment(12, 3) = m_nodes[1]->GetPos_dt().eigen();
    edot.segment(15, 3) = m_nodes[1]->GetD_dt().eigen();
    edot.segment(18, 3) = m_nodes[1]->GetDD_dt().eigen();
    edot.segment(21, 3) = m_nodes[1]->GetDDD_dt().eigen();
}

void ChElementBeamANCF_3243_TR06_GQ322::CalcCoordDerivMatrix(ChMatrixNM<double, 3, 8>& ebardot) {
    ebardot.col(0) = m_nodes[0]->GetPos_dt().eigen();
    ebardot.col(1) = m_nodes[0]->GetD_dt().eigen();
    ebardot.col(2) = m_nodes[0]->GetDD_dt().eigen();
    ebardot.col(3) = m_nodes[0]->GetDDD_dt().eigen();

    ebardot.col(4) = m_nodes[1]->GetPos_dt().eigen();
    ebardot.col(5) = m_nodes[1]->GetD_dt().eigen();
    ebardot.col(6) = m_nodes[1]->GetDD_dt().eigen();
    ebardot.col(7) = m_nodes[1]->GetDDD_dt().eigen();
}

// Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementBeamANCF_3243_TR06_GQ322::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    ChMatrixNMc<double, 8, 3> Sxi_D;

    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_ebar0 * Sxi_D;
}

// Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementBeamANCF_3243_TR06_GQ322::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrix33<double> J_0xi;
    Calc_J_0xi(J_0xi, xi, eta, zeta);

    return (J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------
// ChVector<> ChElementBeamANCF_3243_TR06_GQ322::EvaluateBeamSectionStrains() {
//    // Element shape function
//    ChMatrixNM<double, 1, 9> N;
//    this->ShapeFunctions(N, 0, 0, 0);
//
//    // Determinant of position vector gradient matrix: Initial configuration
//    ChMatrixNM<double, 1, 9> Nx;
//    ChMatrixNM<double, 1, 9> Ny;
//    ChMatrixNM<double, 1, 9> Nz;
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
// void ChElementBeamANCF_3243_TR06_GQ322::EvaluateSectionDisplacement(const double u,
//                                                    const double v,
//                                                    ChVector<>& u_displ,
//                                                    ChVector<>& u_rotaz) {
//    // this is not a corotational element, so just do:
//    EvaluateSectionPoint(u, v, displ, u_displ);
//    u_rotaz = VNULL;  // no angles.. this is ANCF (or maybe return here the slope derivatives?)
//}
//
// void ChElementBeamANCF_3243_TR06_GQ322::EvaluateSectionFrame(const double u,
//                                             const double v,
//                                             ChVector<>& point,
//                                             ChQuaternion<>& rot) {
//    // this is not a corotational element, so just do:
//    EvaluateSectionPoint(u, v, displ, point);
//    rot = QUNIT;  // or maybe use gram-schmidt to get csys of section from slopes?
//}
//
// void ChElementBeamANCF_3243_TR06_GQ322::EvaluateSectionPoint(const double u,
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

void ChElementBeamANCF_3243_TR06_GQ322::EvaluateSectionFrame(const double xi, ChVector<>& point, ChQuaternion<>& rot) {
    ChMatrixNM<double, 3, 24> Sxi;
    ChMatrixNM<double, 3, 24> Sxi_xi;
    ChMatrixNM<double, 3, 24> Sxi_eta;
    Calc_Sxi(Sxi, xi, 0, 0);
    Calc_Sxi_xi(Sxi_xi, xi, 0, 0);
    Calc_Sxi_eta(Sxi_eta, xi, 0, 0);

    ChVectorN<double, 24> e;
    CalcCoordVector(e);

    // r = Se
    point = Sxi * e;

    // Since ANCF does not use rotations, calculate an approximate
    // rotation based off the position vector gradients
    ChVector<double> BeamAxisTangent = Sxi_xi * e;
    ChVector<double> CrossSectionY = Sxi_eta * e;

    // Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33<> msect;
    msect.Set_A_Xdir(BeamAxisTangent, CrossSectionY);

    rot = msect.Get_A_quaternion();
}

// -----------------------------------------------------------------------------
// Functions for ChLoadable interface
// -----------------------------------------------------------------------------

// Gets all the DOFs packed in a single vector (position part).
void ChElementBeamANCF_3243_TR06_GQ322::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD().eigen();
}

// Gets all the DOFs packed in a single vector (velocity part).
void ChElementBeamANCF_3243_TR06_GQ322::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
    mD.segment(block_offset + 0, 3) = m_nodes[0]->GetPos_dt().eigen();
    mD.segment(block_offset + 3, 3) = m_nodes[0]->GetD_dt().eigen();
    mD.segment(block_offset + 6, 3) = m_nodes[0]->GetDD_dt().eigen();
    mD.segment(block_offset + 9, 3) = m_nodes[0]->GetDDD_dt().eigen();

    mD.segment(block_offset + 12, 3) = m_nodes[1]->GetPos_dt().eigen();
    mD.segment(block_offset + 15, 3) = m_nodes[1]->GetD_dt().eigen();
    mD.segment(block_offset + 18, 3) = m_nodes[1]->GetDD_dt().eigen();
    mD.segment(block_offset + 21, 3) = m_nodes[1]->GetDDD_dt().eigen();
}

/// Increment all DOFs using a delta.
void ChElementBeamANCF_3243_TR06_GQ322::LoadableStateIncrement(const unsigned int off_x,
                                                         ChState& x_new,
                                                         const ChState& x,
                                                         const unsigned int off_v,
                                                         const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 12, x_new, x, off_v + 12, Dv);
}

// void ChElementBeamANCF_3243_TR06_GQ322::EvaluateSectionVelNorm(double U, ChVector<>& Result) {
//    ChMatrixNM<double, 1, 9> N;
//    ShapeFunctions(N, U, 0, 0);
//    for (unsigned int ii = 0; ii < 3; ii++) {
//        Result += N(ii * 3) * m_nodes[ii]->GetPos_dt();
//        Result += N(ii * 3 + 1) * m_nodes[ii]->GetPos_dt();
//    }
//}

// Get the pointers to the contained ChVariables, appending to the mvars vector.
void ChElementBeamANCF_3243_TR06_GQ322::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
        mvars.push_back(&m_nodes[i]->Variables_DDD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U) coordinates of the centerline.
void ChElementBeamANCF_3243_TR06_GQ322::ComputeNF(
    const double U,              // parametric coordinate in surface
    ChVectorDynamic<>& Qi,       // Return result of Q = N'*F  here
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is =n. field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    // Compute the generalized force vector for the applied force
    ChMatrixNM<double, 3, 24> Sxi;
    Calc_Sxi(Sxi, U, 0, 0);
    Qi = Sxi.transpose() * F.segment(0, 3);

    // Compute the generalized force vector for the applied moment
    ChMatrixNM<double, 3, 24> Sxi_xi;
    ChMatrixNM<double, 3, 24> Sxi_eta;
    ChMatrixNM<double, 3, 24> Sxi_zeta;
    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;
    ChMatrixNM<double, 3, 24> G;
    ChVectorN<double, 24> e;

    CalcCoordVector(e);

    Calc_Sxi_xi(Sxi_xi, U, 0, 0);
    Calc_Sxi_eta(Sxi_eta, U, 0, 0);
    Calc_Sxi_zeta(Sxi_zeta, U, 0, 0);

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
void ChElementBeamANCF_3243_TR06_GQ322::ComputeNF(
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
    ChMatrixNM<double, 3, 24> Sxi;
    Calc_Sxi(Sxi, U, V, W);
    Qi = Sxi.transpose() * F.segment(0, 3);

    // Compute the generalized force vector for the applied moment
    ChMatrixNM<double, 3, 24> Sxi_xi;
    ChMatrixNM<double, 3, 24> Sxi_eta;
    ChMatrixNM<double, 3, 24> Sxi_zeta;
    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;
    ChMatrixNM<double, 3, 24> G;
    ChVectorN<double, 24> e;

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
double ChElementBeamANCF_3243_TR06_GQ322::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at (U) coordinates.
ChVector<> ChElementBeamANCF_3243_TR06_GQ322::ComputeTangent(const double U) {
    ChMatrixNM<double, 3, 24> Sxi_xi;
    ChVectorN<double, 24> e;
    ChVector<> r_xi;

    Calc_Sxi_xi(Sxi_xi, U, 0, 0);
    CalcCoordVector(e);
    r_xi = Sxi_xi * e;

    return r_xi.GetNormalized();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_2_TR06_GQ322(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBeamANCF_3243_TR06_GQ322::GetStaticGQTables() {
    return &static_tables_2_TR06_GQ322;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBeamANCF_3243_TR06_GQ322 methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBeamANCF_3243_TR06_GQ322::ChMaterialBeamANCF_3243_TR06_GQ322(
    double rho,        // material density
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
ChMaterialBeamANCF_3243_TR06_GQ322::ChMaterialBeamANCF_3243_TR06_GQ322(
    double rho,            // material density
    const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
    const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
    const ChVector<>& G,   // shear moduli (G_xy, G_xz, G_yz)
    const double& k1,      // Shear correction factor along beam local y axis
    const double& k2       // Shear correction factor along beam local z axis
    )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G, k1, k2);
}

// Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the
// Poisson effect
void ChMaterialBeamANCF_3243_TR06_GQ322::Calc_D0_Dv(const ChVector<>& E,
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
