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
// Fully Parameterized ANCF beam element with 2 nodes (24DOF). A Description of this element can be found in: Johannes
// Gerstmayr and Ahmed A Shabana. Efficient integration of the elastic forces and thin three - dimensional beam elements
// in the absolute nodal coordinate formulation. In Proceeding of Multibody Dynamics ECCOMAS Thematic Conference,
// Madrid, Spain, 2005.
//
// A description of the Enhanced Continuum Mechanics based method can be found in: K. Nachbagauer, P. Gruber, and J.
// Gerstmayr. Structural and Continuum Mechanics Approaches for a 3D Shear Deformable ANCF Beam Finite Element :
// Application to Static and Linearized Dynamic Examples.J.Comput.Nonlinear Dynam, 8 (2) : 021004, 2012.
// =============================================================================
// The "Continuous Integration" style calculation for the generalized internal force is based on modifications to
// (including a new analytical Jacobian):  Gerstmayr, J., Shabana, A.A.: Efficient integration of the elastic forces and
// thin three-dimensional beam elements in the absolute nodal coordinate formulation.In: Proceedings of the Multibody
// Dynamics Eccomas thematic Conference, Madrid(2005).
//
// The "Pre-Integration" style calculation is based on modifications
// to Liu, Cheng, Qiang Tian, and Haiyan Hu. "Dynamics of a large scale rigid�flexible multibody system composed of
// composite laminated plates." Multibody System Dynamics 26, no. 3 (2011): 283-305.
//
// A report covering the detailed mathematics and implementation both of these generalized internal force calculations
// and their Jacobians can be found in: Taylor, M.: Technical Report TR-2020-09 Efficient CPU Based Calculations of the
// Generalized Internal Forces and Jacobian of the Generalized Internal Forces for ANCF Continuum Mechanics Elements
// with Linear Viscoelastic Materials, Simulation Based Engineering Lab, University of Wisconsin-Madison; 2021.
// =============================================================================
// This element class has been templatized by the number of Gauss quadrature points to use for the generalized internal
// force calculations and its Jacobian with the recommended values as the default.  Using fewer than 2 Gauss quadrature
// points along the beam axis (NP) or through each cross section direction (NT) will likely result in numerical issues
// with the element.
// =============================================================================

#ifndef CHELEMENTBEAMANCF3243_H
#define CHELEMENTBEAMANCF3243_H

#include <vector>

#include "chrono/core/ChQuadrature.h"
#include "chrono/fea/ChElementBeam.h"
#include "chrono/fea/ChNodeFEAxyzDDD.h"

namespace chrono {
namespace fea {

/// @addtogroup fea_elements
/// @{

/// ANCF beam element with two nodes.  The coordinates at each node are the position vector
/// and 3 position vector gradients.
///
/// The node numbering, as follows:
/// <pre>
///               v
///               ^
///               |
/// A o-----+-----+-----+-----o B -> u
///              /
///             w
/// </pre>

/// Material definition.
/// This class implements material properties for an ANCF Brick.
class ChMaterialBeamANCF_3243 {
  public:
    /// Construct an isotropic material.
    ChMaterialBeamANCF_3243(double rho,        ///< material density
                            double E,          ///< Young's modulus
                            double nu,         ///< Poisson ratio
                            const double& k1,  ///< Shear correction factor along beam local y axis
                            const double& k2   ///< Shear correction factor along beam local z axis
    );

    /// Construct a (possibly) orthotropic material.
    ChMaterialBeamANCF_3243(double rho,            ///< material density
                            const ChVector<>& E,   ///< elasticity moduli (E_x, E_y, E_z)
                            const ChVector<>& nu,  ///< Poisson ratios (nu_xy, nu_xz, nu_yz)
                            const ChVector<>& G,   ///< shear moduli (G_xy, G_xz, G_yz)
                            const double& k1,      ///< Shear correction factor along beam local y axis
                            const double& k2       ///< Shear correction factor along beam local z axis
    );

    /// Return the material density.
    double Get_rho() const { return m_rho; }

    /// Complete Elasticity Tensor in 6x6 matrix form
    const ChMatrixNM<double, 6, 6>& Get_D() const { return m_D; }

    /// Diagonal components of the 6x6 elasticity matrix form without the terms contributing to the Poisson effect
    const ChVectorN<double, 6>& Get_D0() const { return m_D0; }

    /// Upper 3x3 block of the elasticity matrix with the terms contributing to the Poisson effect
    const ChMatrixNM<double, 3, 3>& Get_Dv() const { return m_Dv; }

  private:
    /// Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of
    /// the Poisson effect k1 and k2 are Timoshenko shear correction factors.
    void Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G, double k1, double k2);

    double m_rho;                  ///< density
    ChMatrixNM<double, 6, 6> m_D;  ///< full 6x6 matrix of elastic coefficients
    ChVectorN<double, 6> m_D0;     ///< Diagonal components of the 6x6 elasticity matrix form without the terms
                                   ///< contributing to the Poisson effect
    ChMatrixNM<double, 3, 3>
        m_Dv;  ///< Upper 3x3 block of the elasticity matrix with the terms contributing to the Poisson effect

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <int NP = 3, int NT = 2>
class ChElementBeamANCF_3243 : public ChElementBeam, public ChLoadableU, public ChLoadableUVW {
  public:
    static const int NIP_D0 = NP * NT * NT;  ///< number of Gauss quadrature points excluding the Poisson effect for the
                                             ///< Enhanced Continuum Mechanics method
    static const int NIP_Dv = NP;  ///< number of Gauss quadrature points including the Poisson effect only along the
                                   ///< beam axis for the Enhanced Continuum Mechanics method
    static const int NSF = 8;      ///< number of shape functions

    // Short-cut for defining a column-major Eigen matrix instead of the typically used row-major format
    template <typename T, int M, int N>
    using ChMatrixNMc = Eigen::Matrix<T, M, N, Eigen::ColMajor>;

    using VectorN = ChVectorN<double, NSF>;
    using Vector3N = ChVectorN<double, 3 * NSF>;
    using VectorNIP_D0 = ChVectorN<double, NIP_D0>;
    using VectorNIP_Dv = ChVectorN<double, NIP_Dv>;
    using Matrix3xN = ChMatrixNM<double, 3, NSF>;
    using MatrixNx3 = ChMatrixNM<double, NSF, 3>;
    using MatrixNx3c = ChMatrixNMc<double, NSF, 3>;
    using MatrixNx6 = ChMatrixNM<double, NSF, 6>;
    using MatrixNxN = ChMatrixNM<double, NSF, NSF>;

    /// Internal force calculation method
    enum class IntFrcMethod {
        ContInt,  ///< "Continuous Integration" style method - Fastest for a small number of layers
        PreInt    ///< "Pre-Integration" style method - Fastest for a large number of layers
    };

    ChElementBeamANCF_3243();
    ~ChElementBeamANCF_3243() {}

    /// Get the number of nodes used by this element.
    virtual int GetNnodes() override { return 2; }

    /// Get the number of coordinates in the field used by the referenced nodes.
    virtual int GetNdofs() override { return 2 * 12; }

    /// Get the number of coordinates from the n-th node used by this element.
    virtual int GetNodeNdofs(int n) override { return 12; }

    /// Specify the nodes of this element.
    void SetNodes(std::shared_ptr<ChNodeFEAxyzDDD> nodeA, std::shared_ptr<ChNodeFEAxyzDDD> nodeB);

    /// Specify the element dimensions.
    void SetDimensions(double lenX, double thicknessY, double thicknessZ);

    /// Specify the element material.
    void SetMaterial(std::shared_ptr<ChMaterialBeamANCF_3243> beam_mat);

    /// Return the material.
    std::shared_ptr<ChMaterialBeamANCF_3243> GetMaterial() const { return m_material; }

    /// Access the n-th node of this element.
    virtual std::shared_ptr<ChNodeFEAbase> GetNodeN(int n) override { return m_nodes[n]; }

    /// Get a handle to the first node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeA() const { return m_nodes[0]; }

    /// Get a handle to the second node of this element.
    std::shared_ptr<ChNodeFEAxyzDDD> GetNodeB() const { return m_nodes[1]; }

    /// Turn gravity on/off.
    void SetGravityOn(bool val) { m_gravity_on = val; }

    /// Set the structural damping.
    void SetAlphaDamp(double a);

    /// Get the element length in the xi direction.
    double GetLengthX() const { return m_lenX; }

    /// Get the total thickness of the beam element in the eta direction
    double GetThicknessY() { return m_thicknessY; }

    /// Get the total thickness of the beam element in the zeta direction
    double GetThicknessZ() { return m_thicknessZ; }

    /// Set the calculation method to use for the generalized internal force and its Jacobian calculations.  This should
    /// be set prior to the start of the simulation since can be a significant amount of pre-calculation overhead
    /// required.
    void SetIntFrcCalcMethod(IntFrcMethod method);

    /// Return the type of calculation method currently set for the generalized internal force and its Jacobian
    /// calculations.
    IntFrcMethod GetIntFrcCalcMethod() { return m_method; }

    /// Get the Green-Lagrange strain tensor at the normalized element coordinates (xi, eta, zeta) at the current state
    /// of the element.  Normalized element coordinates span from -1 to 1.
    void GetGreenLagrangeStrain(const double xi, const double eta, const double zeta, ChMatrix33<>& E);

    /// Get the 2nd Piola-Kirchoff stress tensor at the normalized element coordinates (xi, eta, zeta) at the current
    /// state of the element.  Normalized element coordinates span from -1 to 1.
    void GetPK2Stress(const double xi, const double eta, const double zeta, ChMatrix33<>& SPK2);

    /// Get the von Mises stress value at the normalized element coordinates (xi, eta, zeta) at the current state
    /// of the element.  Normalized element coordinates span from -1 to 1.
    double GetVonMissesStress(const double xi, const double eta, const double zeta);

  public:
    // Interface to ChElementBase base class
    // -------------------------------------

    /// Fill the D vector (column matrix) with the current field values at the nodes of the element, with proper
    /// ordering. If the D vector has not the size of this->GetNdofs(), it will be resized.
    ///  {Pos_a Du_a Dv_a Dw_a  Pos_b Du_b Dv_b Dw_b  Pos_c Du_c Dv_c Dw_c  Pos_d Du_d Dv_d Dw_d}
    virtual void GetStateBlock(ChVectorDynamic<>& mD) override;

    /// Update the state of this element.
    virtual void Update() override;

    /// Set M equal to the global mass matrix.
    virtual void ComputeMmatrixGlobal(ChMatrixRef M) override;

    /// Add contribution of element inertia to total nodal masses
    virtual void ComputeNodalMass() override;

    /// Compute the generalized internal force vector for the current nodal coordinates as set the value in the Fi
    /// vector.
    virtual void ComputeInternalForces(ChVectorDynamic<>& Fi) override;

    /// Set H as a linear combination of M, K, and R.
    ///   H = Mfactor * [M] + Kfactor * [K] + Rfactor * [R],
    /// where [M] is the mass matrix, [K] is the stiffness matrix, and [R] is the damping matrix.
    virtual void ComputeKRMmatricesGlobal(ChMatrixRef H,
                                          double Kfactor,
                                          double Rfactor = 0,
                                          double Mfactor = 0) override;

    // Interface to ChElementBeam base class (and similar methods)
    // --------------------------------------

    // Dummy method definition - Does not translate to an ANCF continuum mechanics based beam element
    virtual void EvaluateSectionStrain(const double, chrono::ChVector<double>&) override {}

    // Dummy method definition - Does not translate to an ANCF continuum mechanics based beam element
    virtual void EvaluateSectionForceTorque(const double,
                                            chrono::ChVector<double>&,
                                            chrono::ChVector<double>&) override {}

    /// Gets the xyz displacement of a point on the beam line,
    /// and the rotation RxRyRz of section plane, at abscissa '(xi,0,0)'.
    /// xi = -1 at node A and xi = 1 at node B
    virtual void EvaluateSectionDisplacement(const double xi, ChVector<>& u_displ, ChVector<>& u_rotaz) override {}

    /// Gets the absolute xyz position of a point on the beam line,
    /// and the absolute rotation of section plane, at abscissa '(xi,0,0)'.
    /// xi = -1 at node A and xi = 1 at node B
    virtual void EvaluateSectionFrame(const double xi, ChVector<>& point, ChQuaternion<>& rot) override;

    /// Gets the absolute xyz position of a point on the beam line specified in normalized coordinates
    /// xi = -1 at node A and xi = 1 at node B
    void EvaluateSectionPoint(const double xi, ChVector<>& point);

    /// Gets the absolute xyz velocity of a point on the beam line specified in normalized coordinates
    /// xi = -1 at node A and xi = 1 at node B
    void EvaluateSectionVel(const double xi, ChVector<>& Result);

    // Functions for ChLoadable interface
    // ----------------------------------

    /// Gets the number of DOFs affected by this element (position part).
    virtual int LoadableGet_ndof_x() override { return 2 * 12; }

    /// Gets the number of DOFs affected by this element (velocity part).
    virtual int LoadableGet_ndof_w() override { return 2 * 12; }

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
    virtual int Get_field_ncoords() override { return 12; }

    /// Tell the number of DOFs blocks (ex. =1 for a body, =4 for a tetrahedron, etc.)
    virtual int GetSubBlocks() override { return 2; }

    /// Get the offset of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockOffset(int nblock) override { return m_nodes[nblock]->NodeGetOffset_w(); }

    /// Get the size of the i-th sub-block of DOFs in global vector.
    virtual unsigned int GetSubBlockSize(int nblock) override { return 12; }

    /// Get the pointers to the contained ChVariables, appending to the mvars vector.
    virtual void LoadableGetVariables(std::vector<ChVariables*>& mvars) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at xi coordinate of the beam line, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    virtual void ComputeNF(const double xi,             ///< parametric coordinate along the beam axis
                           ChVectorDynamic<>& Qi,       ///< Return result of Q = N'*F  here
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is =n. field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate
                           ) override;

    /// Evaluate N'*F , where N is some type of shape function
    /// evaluated at xi,eta,zeta coordinates of the volume, each ranging in -1..+1
    /// F is a load, N'*F is the resulting generalized load
    /// Returns also det[J] with J=[dx/du,..], that might be useful in gauss quadrature.
    virtual void ComputeNF(const double xi,             ///< parametric coordinate in volume
                           const double eta,            ///< parametric coordinate in volume
                           const double zeta,           ///< parametric coordinate in volume
                           ChVectorDynamic<>& Qi,       ///< Return result of N'*F  here, maybe with offset block_offset
                           double& detJ,                ///< Return det[J] here
                           const ChVectorDynamic<>& F,  ///< Input F vector, size is = n.field coords.
                           ChVectorDynamic<>* state_x,  ///< if != 0, update state (pos. part) to this, then evaluate Q
                           ChVectorDynamic<>* state_w   ///< if != 0, update state (speed part) to this, then evaluate Q
                           ) override;

    /// This is needed so that it can be accessed by ChLoaderVolumeGravity.
    /// Density is mass per unit surface.
    virtual double GetDensity() override;

    /// Gets the tangent to the beam axis at the parametric coordinate xi.
    /// xi = -1 at node A and xi = 1 at node B
    ChVector<> ComputeTangent(const double xi);

  private:
    /// Initial setup. This is used to precompute matrices that do not change during the simulation, such as the local
    /// stiffness of each element (if any), the mass, etc.
    virtual void SetupInitial(ChSystem* system) override;

    // Internal computations
    // ---------------------

    /// Compute the mass matrix & generalized gravity force of the element.
    /// Note: in this implementation, a constant density material is assumed
    void ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc);

    /// Precalculate constant matrices and scalars for the internal force calculations.  This selects and calls the
    /// method for the style of internal force calculations that is currently selected.
    void PrecomputeInternalForceMatricesWeights();

    /// Precalculate constant matrices and scalars for the "Continuous Integration" style method
    void PrecomputeInternalForceMatricesWeightsContInt();

    /// Precalculate constant matrices and scalars for the "Pre-Integration" style method
    void PrecomputeInternalForceMatricesWeightsPreInt();

    /// Calculate the generalized internal force for the element at the current nodal coordinates and time derivatives
    /// of the nodal coordinates using the "Continuous Integration" style method assuming damping is included
    void ComputeInternalForcesContIntDamping(ChVectorDynamic<>& Fi);

    /// Calculate the generalized internal force for the element at the current nodal coordinates and time derivatives
    /// of the nodal coordinates using the "Continuous Integration" style method assuming no damping
    void ComputeInternalForcesContIntNoDamping(ChVectorDynamic<>& Fi);

    /// Calculate the generalized internal force for the element at the current nodal coordinates and time derivatives
    /// of the nodal coordinates using the "Pre-Integration" style method assuming damping (works well for the case of
    /// no damping as well)
    void ComputeInternalForcesContIntPreInt(ChVectorDynamic<>& Fi);

    /// Calculate the calculate the Jacobian of the internal force integrand using the "Continuous Integration" style
    /// method assuming damping is included This function calculates a linear combination of the stiffness (K) and
    /// damping (R) matrices,
    ///     J = Kfactor * K + Rfactor * R + Mfactor * M
    /// for given coefficients Kfactor, Rfactor, and Mfactor.
    /// This Jacobian includes the global mass matrix M with the global stiffness and damping matrix in H.
    void ComputeInternalJacobianContIntDamping(ChMatrixRef& H, double Kfactor, double Rfactor, double Mfactor);

    /// Calculate the calculate the Jacobian of the internal force integrand using the "Continuous Integration" style
    /// method assuming damping is not included This function calculates just the stiffness (K) matrix,
    ///     J = Kfactor * K + Mfactor * M
    /// for given coefficients Kfactor and Mfactor.
    /// This Jacobian includes the global mass matrix M with the global stiffness in H.
    void ComputeInternalJacobianContIntNoDamping(ChMatrixRef& H, double Kfactor, double Mfactor);

    /// Calculate the calculate the Jacobian of the internal force integrand using the "Pre-Integration" style method
    /// assuming damping is included This function calculates a linear combination of the stiffness (K) and damping (R)
    /// matrices,
    ///     J = Kfactor * K + Rfactor * R + Mfactor * M
    /// for given coefficients Kfactor, Rfactor, and Mfactor.
    /// This Jacobian includes the global mass matrix M with the global stiffness and damping matrix in H.
    void ComputeInternalJacobianPreInt(ChMatrixRef& H, double Kfactor, double Rfactor, double Mfactor);

    /// Calculate the current 3Nx1 vector of nodal coordinates.
    void CalcCoordVector(Vector3N& e);

    /// Calculate the current 3xN matrix of nodal coordinates.
    void CalcCoordMatrix(Matrix3xN& ebar);

    /// Calculate the current 3Nx1 vector of nodal coordinate time derivatives.
    void CalcCoordDerivVector(Vector3N& edot);

    /// Calculate the current 3xN matrix of nodal coordinate time derivatives.
    void CalcCoordDerivMatrix(Matrix3xN& ebardot);

    /// Calculate the current Nx6 matrix of the transpose of the nodal coordinates and nodal coordinate time
    /// derivatives.
    void CalcCombinedCoordMatrix(MatrixNx6& ebar_ebardot);

    /// Calculate the Nx1 Compact Vector of the Normalized Shape Functions (just the unique values)
    void Calc_Sxi_compact(VectorN& Sxi_compact, double xi, double eta, double zeta);

    /// Calculate the Nx1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to xi
    void Calc_Sxi_xi_compact(VectorN& Sxi_xi_compact, double xi, double eta, double zeta);

    /// Calculate the Nx1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to eta
    void Calc_Sxi_eta_compact(VectorN& Sxi_eta_compact, double xi, double eta, double zeta);

    /// Calculate the Nx1 Compact Vector of the Derivative of the Normalized Shape Functions with respect to zeta
    void Calc_Sxi_zeta_compact(VectorN& Sxi_zeta_compact, double xi, double eta, double zeta);

    /// Calculate the Nx3 Compact Matrix of the Derivatives of the Normalized Shape Functions with respect to xi, eta,
    /// and then zeta
    void Calc_Sxi_D(MatrixNx3c& Sxi_D, double xi, double eta, double zeta);

    /// Calculate the element Jacobian of the reference configuration with respect to the normalized configuration
    void Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta);

    /// Calculate the determinate of the element Jacobian of the reference configuration with respect to the normalized
    /// configuration
    double Calc_det_J_0xi(double xi, double eta, double zeta);

    /// Calculate the rotated 6x6 stiffness matrix and reorder it to match the Voigt notation order used with this
    /// element
    void RotateReorderStiffnessMatrix(ChMatrixNM<double, 6, 6>& D, double theta);

    /// Access a statically-allocated set of tables, from 0 to a 10th order, with precomputed tables.
    static ChQuadratureTables* GetStaticGQTables();

    IntFrcMethod m_method;  ///< Generalized internal force and Jacobian calculation method
    std::shared_ptr<ChMaterialBeamANCF_3243> m_material;    ///< material model
    std::vector<std::shared_ptr<ChNodeFEAxyzDDD>> m_nodes;  ///< element nodes
    double m_lenX;                                          ///< total element length along X
    double m_thicknessY;                                    ///< total element length along Y
    double m_thicknessZ;                                    ///< total element length along Z
    double m_Alpha;                                         ///< structural damping
    bool m_damping_enabled;                                 ///< Flag to run internal force damping calculations
    bool m_gravity_on;                                      ///< enable/disable gravity calculation
    Vector3N m_GravForce;                                   ///< Gravity Force
    Matrix3xN m_ebar0;  ///< Element Position Coordinate Vector for the Reference Configuration
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        m_SD_D0;  ///< Precomputed corrected normalized shape function derivative matrices ordered by columns instead of
                  ///< by Gauss quadrature points used for the "Continuous Integration" style method for excluding the
                  ///< Poisson effect in the Enhanced Continuum Mechanics method
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        m_kGQ_D0;  ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors used for the "Continuous
                   ///< Integration" style method for excluding the Poisson effect in the Enhanced Continuum Mechanics
                   ///< method
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        m_SD_Dv;  ///< Precomputed corrected normalized shape function derivative matrices ordered by columns instead of
                  ///< by Gauss quadrature points used for the "Continuous Integration" style method for including the
                  ///< Poisson effect only along the beam axis in the Enhanced Continuum Mechanics method
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        m_kGQ_Dv;  ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors used for the "Continuous
                   ///< Integration" style method for including the Poisson effect only along the beam axis in the
                   ///< Enhanced Continuum Mechanics method
    ChVectorN<double, (NSF * (NSF + 1)) / 2>
        m_MassMatrix;  /// Mass Matrix in extra compact form (Upper Triangular Part only)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        m_O1;  ///< Precomputed Matrix combined with the nodal coordinates used for the "Pre-Integration" style method
               ///< internal force calculation
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        m_O2;  ///< Precomputed Matrix combined with the nodal coordinates used for the "Pre-Integration" style method
               ///< Jacobian calculation
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        m_K3Compact;  ///< Precomputed Matrix combined with the nodal coordinates used for the "Pre-Integration" style
                      ///< method internal force calculation
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        m_K13Compact;  ///< Saved results from the generalized internal force calculation that are reused for the
                       ///< Jacobian calculations for the "Pre-Integration" style method

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @} fea_elements

#include "ChElementBeamANCF_3243_impl.h"

}  // end of namespace fea
}  // end of namespace chrono

#endif
