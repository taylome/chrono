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

#ifndef CHELEMENTBEAMANCFMT30_H
#define CHELEMENTBEAMANCFMT30_H

#include <vector>

#include "chrono/core/ChQuadrature.h"
#include "chrono/fea/ChElementBeam.h"
#include "chrono/fea/ChNodeFEAxyzDD.h"

namespace chrono {
namespace fea {

/// @addtogroup fea_elements
/// @{

/// Material definition.
/// This class implements material properties for an ANCF Beam.
class ChApi ChMaterialBeamANCF_MT30 {
  public:
    /// Construct an isotropic material.
    ChMaterialBeamANCF_MT30(double rho,        ///< material density
                       double E,          ///< Young's modulus
                       double nu,         ///< Poisson ratio
                       const double& k1,  ///< Shear correction factor along beam local y axis
                       const double& k2   ///< Shear correction factor along beam local z axis
    );

    /// Construct a (possibly) orthotropic material.
    ChMaterialBeamANCF_MT30(double rho,            ///< material density
                       const ChVector<>& E,   ///< elasticity moduli (E_x, E_y, E_z)
                       const ChVector<>& nu,  ///< Poisson ratios (nu_xy, nu_xz, nu_yz)
                       const ChVector<>& G,   ///< shear moduli (G_xy, G_xz, G_yz)
                       const double& k1,      ///< Shear correction factor along beam local y axis
                       const double& k2       ///< Shear correction factor along beam local z axis
    );

    /// Return the material density.
    double Get_rho() const { return m_rho; }

    const ChVectorN<double, 6>& Get_D0() const { return m_D0; }
    const ChMatrixNM<double, 3, 3>& Get_Dv() const { return m_Dv; }

  private:
    ///Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the Poisson effect
    ///k1 and k2 are Timoshenko shear correction factors.
    void Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G, double k1, double k2);

    double m_rho;                         ///< density
    ChVectorN<double, 6> m_D0;            ///< matrix of elastic coefficients (split of diagonal terms for integration across the entire element)
    ChMatrix33<double> m_Dv;        ///< matrix of elastic coefficients (remainder of split, upper 3x3 terms for integration only on the beam axis)

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ----------------------------------------------------------------------------
/// ANCF beam element with 3 nodes.
/// This class implements a continuum-based elastic force formulation.
///
/// The node numbering, as follows:
/// <pre>
///               v
///               ^
///               |
/// A o-----+-----o-----+-----o B -> u
///              /C
///             w
/// </pre>
/// where C is the third and central node.

class ChApi ChElementBeamANCF_MT30 : public ChElementBeam, public ChLoadableU, public ChLoadableUVW {
  public:
    using ShapeVector = ChMatrixNM<double, 1, 9>;

    /// Dense matrix with *fixed size* (known at compile time).
    /// A ChMatrixNM is templated by the type of its coefficients and by the matrix dimensions (number of rows and columns).
    template <typename T, int M, int N>
    using ChMatrixNMc = Eigen::Matrix<T, M, N, Eigen::ColMajor>;

    ChElementBeamANCF_MT30();
    ~ChElementBeamANCF_MT30() {}

    /// Get the number of nodes used by this element.
    virtual int GetNnodes() override { return 3; }

    /// Get the number of coordinates in the field used by the referenced nodes.
    virtual int GetNdofs() override { return 3 * 9; }

    /// Get the number of coordinates from the n-th node used by this element.
    virtual int GetNodeNdofs(int n) override { return 9; }

    /// Specify the nodes of this element.
    void SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,   //
                  std::shared_ptr<ChNodeFEAxyzDD> nodeB,   //
                  std::shared_ptr<ChNodeFEAxyzDD> nodeC);  //

    /// Specify the element dimensions.
    void SetDimensions(double lenX, double thicknessY, double thicknessZ) {
        m_lenX = lenX;
        m_thicknessY = thicknessY;
        m_thicknessZ = thicknessZ;
    }

    /// Specify the element material.
    void SetMaterial(std::shared_ptr<ChMaterialBeamANCF_MT30> beam_mat) { m_material = beam_mat; }

    /// Access the n-th node of this element.
    virtual std::shared_ptr<ChNodeFEAbase> GetNodeN(int n) override { return m_nodes[n]; }

    /// Get a handle to the first node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeA() const { return m_nodes[0]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeB() const { return m_nodes[1]; }

    /// Get a handle to the third node of this element.
    std::shared_ptr<ChNodeFEAxyzDD> GetNodeC() const { return m_nodes[2]; }

    /// Return the material.
    std::shared_ptr<ChMaterialBeamANCF_MT30> GetMaterial() const { return m_material; }

    /// Turn gravity on/off.
    void SetGravityOn(bool val) { m_gravity_on = val; }

    /// Set the structural damping.
    void SetAlphaDamp(double a);

    /// Get the element length in the X direction.
    double GetLengthX() const { return m_lenX; }

    /// Get the total thickness of the beam element.
    double GetThicknessY() { return m_thicknessY; }

    /// Get the total thickness of the beam element.
    double GetThicknessZ() { return m_thicknessZ; }

    /// Poisson effect selection.
    enum StrainFormulation {
        CMPoisson,   ///< Continuum-Mechanics formulation, including Poisson effects
        CMNoPoisson  ///< Continuum-Mechanics formulation, disregarding Poisson effects
    };

  public:
    // Interface to ChElementBase base class
    // -------------------------------------

    // Fill the D vector (column matrix) with the current field values at the nodes of the element, with proper
    // ordering. If the D vector has not the size of this->GetNdofs(), it will be resized.
    //  {x_a y_a z_a Dx_a Dx_a Dx_a x_b y_b z_b Dx_b Dy_b Dz_b}
    virtual void GetStateBlock(ChVectorDynamic<>& mD) override;

    // Set H as a linear combination of M, K, and R.
    //   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R],
    // where [M] is the mass matrix, [K] is the stiffness matrix, and [R] is the damping matrix.
    virtual void ComputeKRMmatricesGlobal(ChMatrixRef H,
                                          double Kfactor,
                                          double Rfactor = 0,
                                          double Mfactor = 0) override;

    // Set M as the global mass matrix.
    virtual void ComputeMmatrixGlobal(ChMatrixRef M) override;

    /// Add contribution of element inertia to total nodal masses
    virtual void ComputeNodalMass() override;

    /// Computes the internal forces.
    /// (E.g. the actual position of nodes is not in relaxed reference position) and set values in the Fi vector.
    virtual void ComputeInternalForces(ChVectorDynamic<>& Fi) override;

    /// Update the state of this element.
    virtual void Update() override;


    // Internal computations - Exposed for unit testing
    // ------------------------------------------------

    //Calculate the generalized internal force for the element given the provided current state coordinates
    void ComputeInternalForcesAtState(ChVectorDynamic<>& Fi, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    //Return the pre-computed generalized force due to gravity
    void Get_GravityFrc(ChVectorN<double, 27>& Gi) { Gi = m_GravForce; }


    // Strain Formulation methods
    // --------------------------

    /// Set the strain formulation.
    void SetStrainFormulation(StrainFormulation model) { m_strain_form = model; }

    /// Get the strain formulation.
    StrainFormulation GetStrainFormulation() const { return m_strain_form; }

    // Interface to ChElementBeam base class
    // --------------------------------------

    // void EvaluateSectionPoint(const double u, const ChMatrix<>& displ, ChVector<>& point); // Not needed?

    // Dummy method definitions.
    virtual void EvaluateSectionStrain(const double, chrono::ChVector<double>&) override {}

    virtual void EvaluateSectionForceTorque(const double,
                                            chrono::ChVector<double>&,
                                            chrono::ChVector<double>&) override {}

    /// Gets the xyz displacement of a point on the beam line,
    /// and the rotation RxRyRz of section plane, at abscissa 'eta'.
    /// Note, eta=-1 at node1, eta=+1 at node2.
    /// Note, 'displ' is the displ.state of 2 nodes, ex. get it as GetStateBlock()
    /// Results are not corotated.
    virtual void EvaluateSectionDisplacement(const double eta, ChVector<>& u_displ, ChVector<>& u_rotaz) override {}

    /// Gets the absolute xyz position of a point on the beam line,
    /// and the absolute rotation of section plane, at abscissa 'eta'.
    /// Note, eta=-1 at node1, eta=+1 at node2.
    /// Note, 'displ' is the displ.state of 2 nodes, ex. get it as GetStateBlock()
    /// Results are corotated (expressed in world reference)
    virtual void EvaluateSectionFrame(const double eta, ChVector<>& point, ChQuaternion<>& rot) override;


    // Functions for ChLoadable interface
    // ----------------------------------

    /// Gets the number of DOFs affected by this element (position part).
    virtual int LoadableGet_ndof_x() override { return 3 * 9; }

    /// Gets the number of DOFs affected by this element (velocity part).
    virtual int LoadableGet_ndof_w() override { return 3 * 9; }

    /// Gets all the DOFs packed in a single vector (position part).
    virtual void LoadableGetStateBlock_x(int block_offset, ChState& mD) override;

    /// Gets all the DOFs packed in a single vector (velocity part).
    virtual void LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) override;

    /// Increment all DOFs using a delta.
    virtual void LoadableStateIncrement(const unsigned int off_x,
                                        ChState& x_new,
                                        const ChState& x,
                                        const unsigned int off_v,
                                        const ChStateDelta& Dv) override;

    /// Number of coordinates in the interpolated field, ex=3 for a
    /// tetrahedron finite element or a cable, = 1 for a thermal problem, etc.
    virtual int Get_field_ncoords() override { return 9; }

    /// Tell the number of DOFs blocks (ex. =1 for a body, =4 for a tetrahedron, etc.)
    virtual int GetSubBlocks() override { return 3; }

    /// Get the offset of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockOffset(int nblock) override { return m_nodes[nblock]->NodeGetOffset_w(); }

    /// Get the size of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockSize(int nblock) override { return 9; }

    //What is this used for?  What is the definition?
    //void EvaluateSectionVelNorm(double U, ChVector<>& Result);

    /// Get the pointers to the contained ChVariables, appending to the mvars vector.
    virtual void LoadableGetVariables(std::vector<ChVariables*>& mvars) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at U,V coordinates of the surface, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    virtual void ComputeNF(const double U,              ///< parametric coordinate in surface
                           ChVectorDynamic<>& Qi,       ///< Return result of Q = N'*F  here
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is =n. field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate
                           ) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at U,V,W coordinates of the volume, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    virtual void ComputeNF(const double U,              ///< parametric coordinate in volume
                           const double V,              ///< parametric coordinate in volume
                           const double W,              ///< parametric coordinate in volume
                           ChVectorDynamic<>& Qi,       ///< Return result of N'*F  here, maybe with offset block_offset
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is = n.field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate Q
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate Q
                           ) override;

    /// This is needed so that it can be accessed by ChLoaderVolumeGravity.
    /// Density is mass per unit surface.
    virtual double GetDensity() override;

    /// Gets the tangent to the centerline at the parametric coordinate U.
    /// Each coordinate ranging in -1..+1.
    ChVector<> ComputeTangent(const double U);


    /// Compute the mass matrix & generalized gravity force of the element.
    /// Note: in this 'basic' implementation, a constant rectangular cross-section and
    /// a constant density material are assumed
    void ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc);

    /// Compute Jacobians of the internal forces.
    /// This function calculates a linear combination of the stiffness (K) and damping (R) matrices,
    ///     J = Kfactor * K + Rfactor * R
    /// for given coefficients Kfactor and Rfactor.
    /// This Jacobian will be further combined with the global mass matrix M and included in the global
    /// stiffness matrix H in the function ComputeKRMmatricesGlobal().
    void ComputeInternalJacobians(ChMatrixNM<double, 27, 27>& JacobianMatrix, double Kfactor, double Rfactor);
    void ComputeInternalJacobiansAnalytic(ChMatrixNM<double, 27, 27>& JacobianMatrix, double Kfactor, double Rfactor);
    void PrecomputeInternalForceMatricesWeights();
  private:
    /// Initial setup. This is used to precompute matrices that do not change during the simulation, such as the local
    /// stiffness of each element (if any), the mass, etc.
    virtual void SetupInitial(ChSystem* system) override;

    //void PrecomputeInternalForceMatricesWeights();

    // Internal computations
    // ---------------------

    //Calculate the calculate the Jacobian of the internal force integrand at a single point (called for Gauss-Quadrature integration) - D0 (No Poisson Terms)
    void ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, unsigned int index, ChVectorN<double, 6>& D0, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    //Calculate the calculate the Jacobian of the internal force integrand at a single point (called for Gauss-Quadrature integration) - Dv (Poisson Terms)
    void ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, unsigned int index, ChMatrix33<double>& Dv, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    //Calculate the calculate separately both the dense and compact portions of the Jacobian of the internal force integrand at a single point (called for Gauss-Quadrature integration) - General
    void ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, double GQ_weight, double xi, double eta, double zeta, ChMatrixNM<double, 6, 6>& D, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    //Calculate the calculate the generalized internal force integrand at a single point (called for Gauss-Quadrature integration) - D0 (No Poisson Terms)
    void ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, unsigned int index, ChVectorN<double, 6>& D0, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    //Calculate the calculate the generalized internal force integrand at a single point (called for Gauss-Quadrature integration) - Dv (Poisson Terms)
    void ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, unsigned int index, ChMatrix33<double>& Dv, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    //Calculate the calculate the generalized internal force integrand at a single point (called for Gauss-Quadrature integration) - General
    void ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, double xi, double eta, double zeta, ChMatrixNM<double, 6, 6>& D, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot);

    // Calculate the current 9x3 matrix of nodal coordinates.
    void CalcCoordMatrix(ChMatrixNMc<double, 9, 3>& e);

    // Calculate the current 27x1 vector of nodal coordinates.
    void CalcCoordVector(ChVectorN<double, 27>& e);

    // Calculate the current 27x1 vector of nodal coordinate time derivatives.
    void CalcCoordDerivMatrix(ChMatrixNMc<double, 9, 3>& edot);

    // Calculate the current 27x1 vector of nodal coordinate time derivatives.
    void CalcCoordDerivVector(ChVectorN<double, 27>& edot);

    //Calculate the 3x27 Sparse & Repetitive Normalized Shape Function Matrix 
    void Calc_Sxi(ChMatrixNM<double, 3, 27>& Sxi, double xi, double eta, double zeta);

    //Calculate the 9x1 Compact Vector of the Normalized Shape Functions 
    void Calc_Sxi_compact(ChVectorN<double, 9>& Sxi_compact, double xi, double eta, double zeta);

    //Calculate the 27x3 Compact Shape Function Derivative Matrix
    void Calc_Sxi_D(ChMatrixNMc<double, 9, 3>& Sxi_D, double xi, double eta, double zeta);

    //Calculate the element Jacobian of the reference configuration with respect to the normalized configuration
    void Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta);

    //Calculate the determinate of the element Jacobian of the reference configuration with respect to the normalized configuration
    double Calc_det_J_0xi(double xi, double eta, double zeta);

    /// Access a statically-allocated set of tables, from 0 to a 10th order,
    /// with precomputed tables.
    static ChQuadratureTables* GetStaticGQTables();

    std::vector<std::shared_ptr<ChNodeFEAxyzDD> > m_nodes;  ///< element nodes
    double m_lenX;                                          ///< total element length
    double m_thicknessY;                                    ///< total element thickness along Y
    double m_thicknessZ;                                    ///< total element thickness along Z
    double m_Alpha;                                         ///< structural damping
    bool m_damping_enabled;                                 ///< Flag to run internal force damping calculations
    bool m_gravity_on;                                      ///< enable/disable gravity calculation
    ChVectorN<double, 27> m_GravForce;                      ///< Gravity Force
    ChMatrixNM<double, 9, 9> m_MassMatrix;                  ///< mass matrix - in compact form for reduced memory and reduced KRM matrix computations
    std::shared_ptr<ChMaterialBeamANCF_MT30> m_material;    ///< beam material
    StrainFormulation m_strain_form;                        ///< Strain formulation
    ChMatrixNMc<double, 9, 3> m_e0_bar;                      ///< Element Position Coordinate Matrix for the Reference Configuration
    ChMatrixNMc<double, 144, 3> m_SD_precompute_D0;          ///< Precomputed corrected normalized shape function derivative matrices for no Poisson Effect
    ChVectorN<double, 16> m_GQWeight_det_J_0xi_D0;          ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors for no Poisson Effect
    ChMatrixNMc<double, 36, 3> m_SD_precompute_Dv;           ///< Precomputed corrected normalized shape function derivative matrices for Poisson Effect on the beam axis only
    ChVectorN<double, 4> m_GQWeight_det_J_0xi_Dv;           ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factor for Poisson Effect on the beam axis only

    //For Liu Method
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 81, 81> m_O1;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 81, 81> m_O2;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> m_K3Compact;
    //Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 9, 9> m_K13Compact;
    ChMatrixNMc<double, 81, 81> m_O1;
    ChMatrixNMc<double, 81, 81> m_O2;
    ChMatrixNMc<double, 9, 9> m_K3Compact;
    ChMatrixNMc<double, 9, 9> m_K13Compact;


  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

/// @} fea_elements

}  // end of namespace fea
}  // end of namespace chrono

#endif
