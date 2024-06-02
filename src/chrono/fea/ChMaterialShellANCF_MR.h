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
// Material for ANCF Shell Element for a Two term Mooney-Rivlin Hyperelastic Material Law with penalty term for incompressibility with the option for a single coefficient nonlinear KV Damping
// =============================================================================

#ifndef CHMATERIALSHELLANCF_MR_H
#define CHMATERIALSHELLANCF_MR_H

#include "chrono/fea/ChElementGeneric.h"

namespace chrono {
namespace fea {

/// @addtogroup fea_elements
/// @{

/// Definition of materials to be used for ANCF Shell elements.
class ChApi ChMaterialShellANCF_MR {
  public:
    /// Construct the material.
      ChMaterialShellANCF_MR(double rho,  // material Density
          double c10,  ///< Material Fit Parameter c_10
          double c01,    ///< Material Fit Parameter c_01
          double k,    ///< Bulk Modulus
          double mu      ///< Viscosity Coefficient
      ) : m_rho(rho), m_c10(c10), m_c01(c01), m_k(k), m_mu(mu){};

    /// Return the material density.
    double GetDensity() const { return m_rho; }
    double Get_c10() const { return m_c10; }
    double Get_c01() const { return m_c01; }
    double Get_k() const { return m_k; }
    double Get_mu() const { return m_mu; }
    void Set_mu(double mu) { m_mu = mu; }
    
  private:
    double m_rho;    ///< density
    double m_c10;   ///< Material Fit Parameter mu_10
    double m_c01;   ///< Material Fit Parameter mu_01
    double m_k;      ///< Bulk Modulus
    double m_mu;      ///< Viscosity Coefficient

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @} fea_elements

}  // end of namespace fea
}  // end of namespace chrono

#endif
