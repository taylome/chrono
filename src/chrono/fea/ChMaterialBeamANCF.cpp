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
// Material class for ANCF beam elements using the Enhanced Continuum Mechanics based method
//
// A description of the Enhanced Continuum Mechanics based method can be found in: K. Nachbagauer, P. Gruber, and J.
// Gerstmayr. Structural and Continuum Mechanics Approaches for a 3D Shear Deformable ANCF Beam Finite Element :
// Application to Static and Linearized Dynamic Examples.J.Comput.Nonlinear Dynam, 8 (2) : 021004, 2012.
// =============================================================================

#include "chrono/fea/ChMaterialBeamANCF.h"

namespace chrono {
namespace fea {

// Construct an isotropic material.
ChMaterialBeamANCF::ChMaterialBeamANCF(double rho,        // material density
                                       double E,          // Young's modulus
                                       double nu,         // Poisson ratio
                                       double k1,  // Shear correction factor along beam local y axis
                                       double k2   // Shear correction factor along beam local z axis
                                       )
    : m_rho(rho) {
    double G = 0.5 * E / (1 + nu);
    Calc_D0_Dv(ChVector<>(E), ChVector<>(nu), ChVector<>(G), k1, k2);
}

// Construct a (possibly) orthotropic material.
ChMaterialBeamANCF::ChMaterialBeamANCF(double rho,            // material density
                                       const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
                                       const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
                                       const ChVector<>& G,   // shear moduli (G_xy, G_xz, G_yz)
                                       double k1,      // Shear correction factor along beam local y axis
                                       double k2       // Shear correction factor along beam local z axis
                                       )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G, k1, k2);
}

// Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the
// Poisson effect when utilizing the Enhanced Continuum Mechanics based method
void ChMaterialBeamANCF::Calc_D0_Dv(const ChVector<>& E,
                                    const ChVector<>& nu,
                                    const ChVector<>& G,
                                    double k1,
                                    double k2) {
    // orthotropic material ref: http://homes.civil.aau.dk/lda/Continuum/material.pdf
    // except position of the shear terms is different to match the original ANCF reference paper
    //
    // Assumed Voigt Notation: epsilon = [E11,E22,E33,2*E23,2*E13,2*E12]

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

// Return the complete elasticity tensor in 6x6 matrix form by combining the two parts of the elasticity tensor used for
// the Enhanced Continuum Mechanics based method
void ChMaterialBeamANCF::Get_D(ChMatrixNM<double, 6, 6>& D) {
    D.setZero();
    D.diagonal() = m_D0;
    D.block<3, 3>(0, 0) += m_Dv;
}

}  // end of namespace fea
}  // end of namespace chrono
