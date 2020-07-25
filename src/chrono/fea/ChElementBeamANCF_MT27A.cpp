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
// ANCF beam element with 3 nodes.
// =============================================================================
//
// MT27A = MT27 with internal force timing tests (No F calc)
//  Mass Matrix = Constant, pre-calculated 27x27 matrix
//  Generalized Force due to gravity = Constant 27x1 Vector 
//     (assumption that gravity is constant too)
//  Generalized Internal Force Vector = Calculated in the typical paper way: 
//     e = 27x1 and S = 3x27
//     Inverse of the Element Jacobian (J_0xi) is generated from e0 every time
//     Math direct translation from papers
//     "Full Integration"-1 Number of GQ Integration Points (4x2x2)
//  Jacobian of the Generalized Internal Force Vector = Calculated by numeric 
//     differentiation
//
// =============================================================================

//// MIKE
//// Add function to manually set the initial configuration of the element (What update functions should this call?  Right now, if I change the material, I don't think the mass matrix gets updated
//// Should the Mass Matrix be stored as a sparse matrix? If so, make this change in a later revision
//// Check with Radu on function calls (e.g. calculated det_J_0xi).  If this can be improved, make this change in a later revision
//// Ask Radu about Heap vs. Stack calls with Matrices.
////
//// What should be done with ChVector<> ChElementBeamANCF_MT27A::EvaluateBeamSectionStrains()... It doesn't seems to make sense for this element since there are no input arguments
////   In reality, there should be a function that returns the entire stress and strain tensors (which version(s)) is more of a question.
//// There is another block of base class functions that are commented out in the current Chrono implementation.  What needs to be done with these?
////
//// When is the right time to add new content, like the ability to apply a torque to an element (or different materials)?  Currently only applied forces appear to be supported
////
//// For ChElementBeamANCF_MT27A::ComputeNF, Need to figure out the state update stuff and exactly what Jacobian is being asked for
////   Can the 1D versions just call the 3D version to eliminate code duplication?
////
//// What is "EvaluateSectionVelNorm(double U, ChVector<>& Result)"?  Is it even needed?
////
//// Figure out what is missing to enable irrlicht animations (missing "beam section" functions...How does this affect the entire interface for the element?)


//// RADU
//// A lot more to do here...
//// - more use of Eigen expressions
//// - remove unecessary initializations to zero

#include "chrono/core/ChQuadrature.h"
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChElementBeamANCF_MT27A.h"
#include <cmath>
#include <Eigen/Dense>

namespace chrono {
namespace fea {


// ------------------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------------------

ChElementBeamANCF_MT27A::ChElementBeamANCF_MT27A() : m_gravity_on(false), m_thicknessY(0), m_thicknessZ(0), m_lenX(0), m_Alpha(0), m_damping_enabled(false){
    m_nodes.resize(3);
}

// ------------------------------------------------------------------------------
// Set element nodes
// ------------------------------------------------------------------------------

void ChElementBeamANCF_MT27A::SetNodes(std::shared_ptr<ChNodeFEAxyzDD> nodeA,
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
void ChElementBeamANCF_MT27A::SetupInitial(ChSystem* system) {
    // Compute mass matrix and gravitational forces and store them since they are constants
    ComputeMassMatrixAndGravityForce(system->Get_G_acc());
    PrecomputeInternalForceMatricesWeights();
}

// State update.
void ChElementBeamANCF_MT27A::Update() {
    ChElementGeneric::Update();
}

// Fill the D vector with the current field values at the element nodes.
void ChElementBeamANCF_MT27A::GetStateBlock(ChVectorDynamic<>& mD) {
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
void ChElementBeamANCF_MT27A::ComputeKRMmatricesGlobal(ChMatrixRef H, double Kfactor, double Rfactor, double Mfactor) {
    assert((H.rows() == 27) && (H.cols() == 27));

#if true
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi;
    ChMatrixNM<double, 9, 9> Jacobian_CompactPart = Mfactor * m_MassMatrix;
    ChMatrixNM<double, 3, 3> S;
    for (unsigned int index = 0; index < 16; index++) {
        
        S(0, 0) = m_SPK2(6 * index);
        S(1, 1) = m_SPK2((6 * index) + 1);
        S(2, 2) = m_SPK2((6 * index) + 2);
        S(2, 1) = m_SPK2((6 * index) + 3);
        S(1, 2) = m_SPK2((6 * index) + 3);
        S(0, 2) = m_SPK2((6 * index) + 4);
        S(2, 0) = m_SPK2((6 * index) + 4);
        S(0, 1) = m_SPK2((6 * index) + 5);
        S(1, 0) = m_SPK2((6 * index) + 5);

        Sxi_D_0xi = m_SD_precompute.block<9,3>(0, 3 * index);

        Jacobian_CompactPart -= Kfactor*Sxi_D_0xi*S*Sxi_D_0xi.transpose();
    }

    ChVectorN<double, 3> Ediag;
    ChVectorN<double, 3> Sdiag;
    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();
    for (unsigned int index = 16; index < 20; index++) {

        Sxi_D_0xi = m_SD_precompute.block<9,3>(0, 3 * index);

        Ediag(0) = m_Ediag((3 * (index - 16)));
        Ediag(1) = m_Ediag((3 * (index - 16)) + 1);
        Ediag(2) = m_Ediag((3 * (index - 16)) + 2);

        Sdiag = Dv*Ediag;

        Jacobian_CompactPart -= Kfactor*Sxi_D_0xi*Sdiag.asDiagonal()*Sxi_D_0xi.transpose();
    }

    H.setZero();
    //Inflate the Mass Matrix since it is stored in compact form.
    //In MATLAB notation:	 
    //H(1:3:end,1:3:end) = H(1:3:end,1:3:end) + Jacobian_CompactPart;
    //H(2:3:end,2:3:end) = H(2:3:end,2:3:end) + Jacobian_CompactPart;
    //H(3:3:end,3:3:end) = H(3:3:end,3:3:end) + Jacobian_CompactPart;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            H(3 * i, 3 * j) += Jacobian_CompactPart(i, j);
            H(3 * i + 1, 3 * j + 1) += Jacobian_CompactPart(i, j);
            H(3 * i + 2, 3 * j + 2) += Jacobian_CompactPart(i, j);
        }
    }

    Eigen::Map<ChVectorN<double, 27>> Sxi_D_0xiReshaped(Sxi_D_0xi.data(), Sxi_D_0xi.size());

    ChVectorN<double, 9> SDCol0;
    ChVectorN<double, 9> SDCol1;
    ChVectorN<double, 9> SDCol2;

    ChRowVectorN<double, 3> Fcol0transpose;
    ChRowVectorN<double, 3> Fcol1transpose;
    ChRowVectorN<double, 3> Fcol2transpose;

    ChMatrixNM<double, 6, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    ChMatrixNM<double, 27, 3> partial_e_tempA;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedA(partial_e_tempA.data(), partial_e_tempA.size());
    ChMatrixNM<double, 27, 3> partial_e_tempB;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedB(partial_e_tempB.data(), partial_e_tempB.size());
    ChMatrixNM<double, 27, 3> partial_e_tempC;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedC(partial_e_tempC.data(), partial_e_tempC.size());

    ChMatrixNM<double, 3, 27> partial_epsilon_partial_e_Dv;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    ChMatrixNM<double, 9, 3> partial_e_temp0;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped0(partial_e_temp0.data(), partial_e_temp0.size());
    ChMatrixNM<double, 9, 3> partial_e_temp1;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped1(partial_e_temp1.data(), partial_e_temp1.size());
    ChMatrixNM<double, 9, 3> partial_e_temp2;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped2(partial_e_temp2.data(), partial_e_temp2.size());

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();

    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrixNM<double, 6, 27> Scaled_Combined_partial_epsilon_partial_e;
        ChRowVectorN<double, 3> Fdotcol0transpose;
        ChRowVectorN<double, 3> Fdotcol1transpose;
        ChRowVectorN<double, 3> Fdotcol2transpose;
        ChRowVectorN<double, 3> Scaled_Combined_Fcol0transpose;
        ChRowVectorN<double, 3> Scaled_Combined_Fcol1transpose;
        ChRowVectorN<double, 3> Scaled_Combined_Fcol2transpose;

        for (unsigned int index = 0; index < 16; index++) {
            Sxi_D_0xi = m_SD_precompute.block<9,3>(0, 3 * index);

            Fcol0transpose = m_F_Block.block<3,1>(0, 3 * index).transpose();
            Fcol1transpose = m_F_Block.block<3,1>(0, (3 * index) + 1).transpose();
            Fcol2transpose = m_F_Block.block<3,1>(0, (3 * index) + 2).transpose();

            partial_e_tempA = Sxi_D_0xiReshaped*Fcol0transpose;
            partial_e_tempB = Sxi_D_0xiReshaped*Fcol1transpose;
            partial_e_tempC = Sxi_D_0xiReshaped*Fcol2transpose;

            partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block<1,27>(0, 0);
            partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block<1, 27>(0, 27);
            partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block<1, 27>(0, 54);
            partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block<1, 27>(0, 54) + partial_e_tempReshapedC.block<1, 27>(0, 27);
            partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block<1, 27>(0, 54) + partial_e_tempReshapedC.block<1, 27>(0, 0);
            partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block<1, 27>(0, 27) + partial_e_tempReshapedB.block<1, 27>(0, 0);

            Fdotcol0transpose = m_Fdot_Block.block<3, 1>(0, 3 * index).transpose();
            Fdotcol1transpose = m_Fdot_Block.block<3, 1>(0, (3 * index) + 1).transpose();
            Fdotcol2transpose = m_Fdot_Block.block<3, 1>(0, (3 * index) + 2).transpose();

            Scaled_Combined_Fcol0transpose = m_GQWeight_det_J_0xi(index)*((Kfactor + m_Alpha*Rfactor)*Fcol0transpose + (m_Alpha*Kfactor)*Fdotcol0transpose);
            Scaled_Combined_Fcol1transpose = m_GQWeight_det_J_0xi(index)*((Kfactor + m_Alpha*Rfactor)*Fcol1transpose + (m_Alpha*Kfactor)*Fdotcol1transpose);
            Scaled_Combined_Fcol2transpose = m_GQWeight_det_J_0xi(index)*((Kfactor + m_Alpha*Rfactor)*Fcol2transpose + (m_Alpha*Kfactor)*Fdotcol2transpose);

            partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_Fcol0transpose;
            partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_Fcol1transpose;
            partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_Fcol2transpose;

            Scaled_Combined_partial_epsilon_partial_e.row(0) = D0(0)*partial_e_tempReshapedA.block<1, 27>(0, 0);
            Scaled_Combined_partial_epsilon_partial_e.row(1) = D0(1)*partial_e_tempReshapedB.block<1, 27>(0, 27);
            Scaled_Combined_partial_epsilon_partial_e.row(2) = D0(2)*partial_e_tempReshapedC.block<1, 27>(0, 54);
            Scaled_Combined_partial_epsilon_partial_e.row(3).noalias() = D0(3)*(partial_e_tempReshapedB.block<1, 27>(0, 54) + partial_e_tempReshapedC.block<1, 27>(0, 27));
            Scaled_Combined_partial_epsilon_partial_e.row(4).noalias() = D0(4)*(partial_e_tempReshapedA.block<1, 27>(0, 54) + partial_e_tempReshapedC.block<1, 27>(0, 0));
            Scaled_Combined_partial_epsilon_partial_e.row(5).noalias() = D0(5)*(partial_e_tempReshapedA.block<1, 27>(0, 27) + partial_e_tempReshapedB.block<1, 27>(0, 0));

            //Calculate and accumulate the dense component of the Jacobian.
            H -= partial_epsilon_partial_e.transpose()*Scaled_Combined_partial_epsilon_partial_e;
        }

        ChMatrixNM<double, 3, 27> Scaled_Combined_partial_epsilon_partial_e_Dv;
        for (unsigned int index = 16; index < 20; index++) {
            SDCol0 = m_SD_precompute.block<9,1>(0, 3 * index);
            SDCol1 = m_SD_precompute.block<9, 1>(0, (3 * index) + 1);
            SDCol2 = m_SD_precompute.block<9, 1>(0, (3 * index) + 2);

            Fcol0transpose = m_F_Block.block<3, 1>(0, 3 * index).transpose();
            Fcol1transpose = m_F_Block.block<3, 1>(0, (3 * index) + 1).transpose();
            Fcol2transpose = m_F_Block.block<3, 1>(0, (3 * index) + 2).transpose();

            partial_e_temp0 = SDCol0*Fcol0transpose;
            partial_epsilon_partial_e_Dv.row(0) = partial_e_tempReshaped0;
            partial_e_temp1 = SDCol1*Fcol1transpose;
            partial_epsilon_partial_e_Dv.row(1) = partial_e_tempReshaped1;
            partial_e_temp2 = SDCol2*Fcol2transpose;
            partial_epsilon_partial_e_Dv.row(2) = partial_e_tempReshaped2;

            Fdotcol0transpose = m_Fdot_Block.block<3, 1>(0, 3 * index).transpose();
            Fdotcol1transpose = m_Fdot_Block.block<3, 1>(0, (3 * index) + 1).transpose();
            Fdotcol2transpose = m_Fdot_Block.block<3, 1>(0, (3 * index) + 2).transpose();

            Scaled_Combined_Fcol0transpose = m_GQWeight_det_J_0xi(index)*((Kfactor + m_Alpha*Rfactor)*Fcol0transpose + (m_Alpha*Kfactor)*Fdotcol0transpose);
            Scaled_Combined_Fcol1transpose = m_GQWeight_det_J_0xi(index)*((Kfactor + m_Alpha*Rfactor)*Fcol1transpose + (m_Alpha*Kfactor)*Fdotcol1transpose);
            Scaled_Combined_Fcol2transpose = m_GQWeight_det_J_0xi(index)*((Kfactor + m_Alpha*Rfactor)*Fcol2transpose + (m_Alpha*Kfactor)*Fdotcol2transpose);

            //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
            //combined with epsilon dot with the correct Jacobian weighting factors 
            //with respect to the nodal coordinates entry by entry in the vector epsilon
            partial_e_temp0 = SDCol0*Scaled_Combined_Fcol0transpose;
            Scaled_Combined_partial_epsilon_partial_e_Dv.row(0) = partial_e_tempReshaped0;
            partial_e_temp1 = SDCol1*Scaled_Combined_Fcol1transpose;
            Scaled_Combined_partial_epsilon_partial_e_Dv.row(1) = partial_e_tempReshaped1;
            partial_e_temp2 = SDCol2*Scaled_Combined_Fcol2transpose;
            Scaled_Combined_partial_epsilon_partial_e_Dv.row(2) = partial_e_tempReshaped2;

            //Calculate and accumulate the dense component of the Jacobian.
            H -= partial_epsilon_partial_e_Dv.transpose()*Dv*Scaled_Combined_partial_epsilon_partial_e_Dv;
        }
    }
    else {
        for (unsigned int index = 0; index < 16; index++) {
            Sxi_D_0xi = m_SD_precompute.block(0, 3 * index, 9, 3);

            Fcol0transpose = m_F_Block.block<3, 1>(0, 3 * index).transpose();
            Fcol1transpose = m_F_Block.block<3, 1>(0, (3 * index) + 1).transpose();
            Fcol2transpose = m_F_Block.block<3, 1>(0, (3 * index) + 2).transpose();

            partial_e_tempA = Sxi_D_0xiReshaped*Fcol0transpose;
            partial_e_tempB = Sxi_D_0xiReshaped*Fcol1transpose;
            partial_e_tempC = Sxi_D_0xiReshaped*Fcol2transpose;

            partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block<1, 27>(0, 0);
            partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block<1, 27>(0, 27);
            partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block<1, 27>(0, 54);
            partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block<1, 27>(0, 54) + partial_e_tempReshapedC.block<1, 27>(0, 27);
            partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block<1, 27>(0, 54) + partial_e_tempReshapedC.block<1, 27>(0, 0);
            partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block<1, 27>(0, 27) + partial_e_tempReshapedB.block<1, 27>(0, 0);

            H -= m_GQWeight_det_J_0xi(index)*Kfactor*partial_epsilon_partial_e.transpose()*D0.asDiagonal()*partial_epsilon_partial_e;
        }

        for (unsigned int index = 16; index < 20; index++) {
            SDCol0 = m_SD_precompute.block<9,1>(0, 3 * index);
            SDCol1 = m_SD_precompute.block<9, 1>(0, (3 * index) + 1);
            SDCol2 = m_SD_precompute.block<9, 1>(0, (3 * index) + 2);

            Fcol0transpose = m_F_Block.block<3, 1>(0, 3 * index).transpose();
            Fcol1transpose = m_F_Block.block<3, 1>(0, (3 * index) + 1).transpose();
            Fcol2transpose = m_F_Block.block<3, 1>(0, (3 * index) + 2).transpose();

            partial_e_temp0 = SDCol0*Fcol0transpose;
            partial_epsilon_partial_e_Dv.row(0) = partial_e_tempReshaped0;
            partial_e_temp1 = SDCol1*Fcol1transpose;
            partial_epsilon_partial_e_Dv.row(1) = partial_e_tempReshaped1;
            partial_e_temp2 = SDCol2*Fcol2transpose;
            partial_epsilon_partial_e_Dv.row(2) = partial_e_tempReshaped2;

            H -= partial_epsilon_partial_e_Dv.transpose()*(m_GQWeight_det_J_0xi(index)*Kfactor*Dv)*partial_epsilon_partial_e_Dv;
        }
    }


    ////ChMatrixNM<double, 27, 27> JacobianMatrix_Analytic = H;
    //////ComputeInternalJacobiansAnalytic(JacobianMatrix_Analytic, Kfactor, Rfactor);
    //ChMatrixNM<double, 27, 27> Delta = H - JacobianMatrix;
    //Delta = Delta.cwiseAbs();

    //std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    //std::cout << "Numeric Jacobian = " << std::endl;
    //std::cout << JacobianMatrix << std::endl;

    //std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    //std::cout << "Analytic Jacobian = " << std::endl;
    //std::cout << H << std::endl;

    //std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    //std::cout << "Analytic Jacobian - Numeric Jacobian = " << std::endl;
    //std::cout << H - JacobianMatrix << std::endl;

    //std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    //std::cout << "Max Delta = " << Delta.colwise().maxCoeff() << std::endl;

    ////ComputeInternalJacobiansAnalytic(JacobianMatrix, Kfactor, Rfactor);

    //// Load Jac + Mfactor*[M] into H
    //H = JacobianMatrix;
    //for (unsigned int i = 0; i < 9; i++) {
    //    for (unsigned int j = 0; j < 9; j++) {
    //        H(3 * i, 3 * j) += Mfactor * m_MassMatrix(i, j);
    //        H(3 * i + 1, 3 * j + 1) += Mfactor * m_MassMatrix(i, j);
    //        H(3 * i + 2, 3 * j + 2) += Mfactor * m_MassMatrix(i, j);
    //    }
    //}

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
#endif // true

}

// Return the mass matrix.
void ChElementBeamANCF_MT27A::ComputeMmatrixGlobal(ChMatrixRef M) {
    M.setZero();

    //Inflate the Mass Matrix since it is stored in compact form.
    //In MATLAB notation:	 
    //M(1:3:end,1:3:end) = m_MassMatrix;
    //M(2:3:end,2:3:end) = m_MassMatrix;
    //M(3:3:end,3:3:end) = m_MassMatrix;
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
void ChElementBeamANCF_MT27A::ComputeMassMatrixAndGravityForce(const ChVector<>& g_acc) {
    // For this element, 5 GQ Points are needed in the xi direction
    //  and 2 GQ Points are needed in the eta & zeta directions
    //  for exact integration of the element's mass matrix, even if 
    //  the reference configuration is not straight
    // Since the major pieces of the generalized force due to gravity
    //  can also be used to calculate the mass matrix, these calculations
    //  are performed at the same time.

    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 4;       // 5 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1; // 2 Point Gauss-Quadrature;

    double rho = GetMaterial()->Get_rho(); //Density of the material for the element

    //Set these to zeros since they will be incremented as the vector/matrix is calculated
    m_MassMatrix.setZero();
    m_GravForce.setZero();

    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] * GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                double det_J_0xi = Calc_det_J_0xi(xi, eta, zeta); // determinate of the element Jacobian (volume ratio)
                ChMatrixNM<double, 3, 27> Sxi; //3x27 Normalized Shape Function Matrix
                Calc_Sxi(Sxi, xi, eta, zeta);
                ChVectorN<double, 9> Sxi_compact; //9x1 Vector of the Unique Normalized Shape Functions
                Calc_Sxi_compact(Sxi_compact, xi, eta, zeta);

                m_GravForce += (GQ_weight*rho*det_J_0xi)*Sxi.transpose()*g_acc.eigen();
                m_MassMatrix += (GQ_weight*rho*det_J_0xi)*Sxi_compact*Sxi_compact.transpose();
            }
        }
    }

}

//Precalculate constant matrices and scalars for the internal force calculations
void ChElementBeamANCF_MT27A::PrecomputeInternalForceMatricesWeights() {
    ChQuadratureTables* GQTable = GetStaticGQTables();
    unsigned int GQ_idx_xi = 3;       // 4 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1; // 2 Point Gauss-Quadrature;

    //Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight reference configuration & GQ Weights times the determiate of the element Jacobian for later
    //Calculating the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] * GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta*GQTable->Lroots[GQ_idx_eta_zeta].size() + it_xi*GQTable->Lroots[GQ_idx_eta_zeta].size()*GQTable->Lroots[GQ_idx_eta_zeta].size();
                ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
                ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta);
                J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
   
                m_SD_precompute_D0.block(0, 3*index, 9, 3) = Sxi_D*J_0xi.inverse();
                m_GQWeight_det_J_0xi_D0(index) = -J_0xi.determinant()*GQ_weight;
            }
        }
    }

    //Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight reference configuration & GQ Weights times the determiate of the element Jacobian for later
    //Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * 2 * 2;
        double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
        ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
        ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives

        Calc_Sxi_D(Sxi_D, xi, 0, 0);
        J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;

        m_SD_precompute_Dv.block(0, 3*it_xi, 9, 3) = Sxi_D*J_0xi.inverse();
        m_GQWeight_det_J_0xi_Dv(it_xi) = -J_0xi.determinant()*GQ_weight;
    }

    m_SD_precompute.block(0, 0, 9, 48) = m_SD_precompute_D0;
    m_SD_precompute.block(0, 48, 9, 12) = m_SD_precompute_Dv;
    m_GQWeight_det_J_0xi.block(0, 0, 16, 1) = m_GQWeight_det_J_0xi_D0;
    m_GQWeight_det_J_0xi.block(16, 0, 4, 1) = m_GQWeight_det_J_0xi_Dv;

}

/// This class computes and adds corresponding masses to ElementGeneric member m_TotalMass
void ChElementBeamANCF_MT27A::ComputeNodalMass() {
    m_nodes[0]->m_TotalMass += m_MassMatrix(0, 0) + m_MassMatrix(0, 3) + m_MassMatrix(0, 6);
    m_nodes[1]->m_TotalMass += m_MassMatrix(3, 3) + m_MassMatrix(3, 0) + m_MassMatrix(3, 6);
    m_nodes[2]->m_TotalMass += m_MassMatrix(6, 6) + m_MassMatrix(6, 0) + m_MassMatrix(6, 3);
}

// -----------------------------------------------------------------------------
// Elastic force calculation
// -----------------------------------------------------------------------------

// Set structural damping.
void ChElementBeamANCF_MT27A::SetAlphaDamp(double a) {
    m_Alpha = a;
    if (std::abs(m_Alpha) > 1e-10)
        m_damping_enabled = true;
    else
        m_damping_enabled = false;
}

void ChElementBeamANCF_MT27A::ComputeInternalForces(ChVectorDynamic<>& Fi) {
    ChMatrixNMc<double, 9, 3> e_bar;
    CalcCoordMatrix(e_bar);
    
    //Runs faster if the internal force with or without damping calculations are not combined into the same function using the common calculations with an if statement for the damping in the middle to calculate the different P_transpose_scaled_Block components
    if (m_damping_enabled) {
        ChMatrixNMc<double, 9, 3> e_bar_dot;
        CalcCoordDerivMatrix(e_bar_dot);

        ComputeInternalForcesAtState(Fi, e_bar, e_bar_dot);
    }
    else {
        ComputeInternalForcesAtStateNoDamping(Fi, e_bar);
    }

    if (m_gravity_on) {
        Fi += m_GravForce;
    }
}

void ChElementBeamANCF_MT27A::ComputeInternalForcesAtState(ChVectorDynamic<>& Fi, const ChMatrixNMc<double, 9, 3>& e_bar, const ChMatrixNMc<double, 9, 3>& e_bar_dot) {

    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full Integration"
    // However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in roughly 1/3 of the calculations

    m_F_Block.setOnes();
    m_Fdot_Block.setOnes();
    //m_F_Block.noalias() = e_bar.transpose()*m_SD_precompute; //Deformation Gradient tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in a big matrix
    //m_Fdot_Block.noalias() = e_bar_dot.transpose()*m_SD_precompute; //Time derivative of the Deformation Gradient tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in a big matrix
    ChMatrixNM<double, 60, 3> P_transpose_scaled_Block; //1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in a big matrix
    ChVectorN<double, 3> FCol0;             //1st Column of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FCol1;             //2nd Column of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FCol2;             //3rd Column of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FdotCol0;          //1st Column of the Time derivative of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FdotCol1;          //2nd Column of the Time derivative of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FdotCol2;          //3rd Column of the Time derivative of the Deformation Gradient tensor for the current GQ point
    //ChVectorN<double, 6> SPK2;              //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Voigt Notation
    ChVectorN<double, 3> Sdiag;             //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Diagonal in this special case due to the structure of the 6x6 stiffness matrix for this part of the selective reduced Gauss-Quadrature
    ChVectorN<double, 3> Ediag;             //Diagonal entries of the Green-Lagrange strain tensor (symmetric tensor)

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    I_3x3.setIdentity();

    for (unsigned int index = 0; index < 16; index++) {

        FCol0 = m_F_Block.block(0, 3 * index, 3, 1);
        FCol1 = m_F_Block.block(0, (3 * index) + 1, 3, 1);
        FCol2 = m_F_Block.block(0, (3 * index) + 2, 3, 1);
        FdotCol0 = m_Fdot_Block.block(0, 3 * index, 3, 1);
        FdotCol1 = m_Fdot_Block.block(0, (3 * index) + 1, 3, 1);
        FdotCol2 = m_Fdot_Block.block(0, (3 * index) + 2, 3, 1);

        m_SPK2(6 * index) = m_GQWeight_det_J_0xi(index)*D0(0)*(0.5*FCol0.dot(FCol0) - 0.5 + m_Alpha*FCol0.dot(FdotCol0)); //Smat(0, 0)
        m_SPK2((6 * index) + 1) = m_GQWeight_det_J_0xi(index)*D0(1)*(0.5*FCol1.dot(FCol1) - 0.5 + m_Alpha*FCol1.dot(FdotCol1)); //Smat(1, 1)
        m_SPK2((6 * index) + 2) = m_GQWeight_det_J_0xi(index)*D0(2)*(0.5*FCol2.dot(FCol2) - 0.5 + m_Alpha*FCol2.dot(FdotCol2)); //Smat(2, 2)
        m_SPK2((6 * index) + 3) = m_GQWeight_det_J_0xi(index)*D0(3)*(FCol1.dot(FCol2) + m_Alpha*(FCol1.dot(FdotCol2)+ FCol2.dot(FdotCol1))); //Smat(1, 2) = Smat(2, 1)
        m_SPK2((6 * index) + 4) = m_GQWeight_det_J_0xi(index)*D0(4)*(FCol0.dot(FCol2) + m_Alpha*(FCol0.dot(FdotCol2) + FCol2.dot(FdotCol0))); //Smat(0, 2) = Smat(2, 0)
        m_SPK2((6 * index) + 5) = m_GQWeight_det_J_0xi(index)*D0(5)*(FCol0.dot(FCol1) + m_Alpha*(FCol0.dot(FdotCol1) + FCol1.dot(FdotCol0))); //Smat(0, 1) = Smat(1, 0)

        P_transpose_scaled_Block.block(3 * index, 0, 1, 3).noalias() = m_SPK2(6 * index)*FCol0.transpose() + m_SPK2((6 * index) + 5)*FCol1.transpose() + m_SPK2((6 * index) + 4)*FCol2.transpose();
        P_transpose_scaled_Block.block(3 * index + 1, 0, 1, 3).noalias() = m_SPK2((6 * index) + 5)*FCol0.transpose() + m_SPK2((6 * index) + 1)*FCol1.transpose() + m_SPK2((6 * index) + 3)*FCol2.transpose();
        P_transpose_scaled_Block.block(3 * index + 2, 0, 1, 3).noalias() = m_SPK2((6 * index) + 4)*FCol0.transpose() + m_SPK2((6 * index) + 3)*FCol1.transpose() + m_SPK2((6 * index) + 2)*FCol2.transpose();

    }

    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();
    ChRowVectorN<double, 3> DvRow0 = Dv.row(0);
    ChRowVectorN<double, 3> DvRow1 = Dv.row(1);
    ChRowVectorN<double, 3> DvRow2 = Dv.row(2);

    for (unsigned int index = 16; index < 20; index++) {

        FCol0 = m_F_Block.block(0, (3 * index), 3, 1);
        FCol1 = m_F_Block.block(0, (3 * index) + 1, 3, 1);
        FCol2 = m_F_Block.block(0, (3 * index) + 2, 3, 1);
        FdotCol0 = m_Fdot_Block.block(0, (3 * index), 3, 1);
        FdotCol1 = m_Fdot_Block.block(0, (3 * index) + 1, 3, 1);
        FdotCol2 = m_Fdot_Block.block(0, (3 * index) + 2, 3, 1);

        Ediag(0) = m_GQWeight_det_J_0xi(index)*(0.5*FCol0.dot(FCol0) - 0.5 + m_Alpha*FCol0.dot(FdotCol0));
        Ediag(1) = m_GQWeight_det_J_0xi(index)*(0.5*FCol1.dot(FCol1) - 0.5 + m_Alpha*FCol1.dot(FdotCol1));
        Ediag(2) = m_GQWeight_det_J_0xi(index)*(0.5*FCol2.dot(FCol2) - 0.5 + m_Alpha*FCol2.dot(FdotCol2));

        //m_Ediag.block((3 * (index-16)), 0, 3, 1).noalias() = Ediag;
        //Faster to cache these values one by one rather than using the block command
        m_Ediag((3 * (index - 16))) = Ediag(0);
        m_Ediag((3 * (index - 16)) + 1) = Ediag(1);
        m_Ediag((3 * (index - 16)) + 2) = Ediag(2);

        P_transpose_scaled_Block.block((3 * index), 0, 1, 3).noalias() = Ediag.dot(DvRow0)*FCol0.transpose();
        P_transpose_scaled_Block.block((3 * index) + 1, 0, 1, 3).noalias() = Ediag.dot(DvRow1)*FCol1.transpose();
        P_transpose_scaled_Block.block((3 * index) + 2, 0, 1, 3).noalias() = Ediag.dot(DvRow2)*FCol2.transpose();
    }

    ChMatrixNM<double, 9, 3> QiCompact = m_SD_precompute*P_transpose_scaled_Block;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementBeamANCF_MT27A::ComputeInternalForcesAtStateNoDamping(ChVectorDynamic<>& Fi, const ChMatrixNMc<double, 9, 3>& e_bar) {

    // Straight & Normalized Internal Force Integrand is of order : 8 in xi, order : 4 in eta, and order : 4 in zeta.
    // This requires GQ 5 points along the xi direction and 3 points along the eta and zeta directions for "Full Integration"
    // However, very similar results can be obtained with 1 fewer GQ point in  each direction, resulting in roughly 1/3 of the calculations

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    I_3x3.setIdentity();

    m_F_Block.setOnes();
    //m_F_Block.noalias() = e_bar.transpose()*m_SD_precompute; //Deformation Gradient tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in a big matrix
    ChMatrixNM<double, 60, 3> P_transpose_scaled_Block; //1st tensor Piola-Kirchoff stress tensor (non-symmetric tensor) - Tiled across all Gauss-Quadrature points in a big matrix
    ChVectorN<double, 3> FCol0;             //1st Column of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FCol1;             //2nd Column of the Deformation Gradient tensor for the current GQ point
    ChVectorN<double, 3> FCol2;             //3rd Column of the Deformation Gradient tensor for the current GQ point
    //ChVectorN<double, 6> SPK2;              //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Voigt Notation
    //ChVectorN<double, 3> Sdiag;             //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Diagonal in this special case due to the structure of the 6x6 stiffness matrix for this part of the selective reduced Gauss-Quadrature
    ChVectorN<double, 3> Ediag;             //Diagonal entries of the Green-Lagrange strain tensor (symmetric tensor)

    for (unsigned int index = 0; index < 16; index++) {

        FCol0 = m_F_Block.block<3,1>(0, 3 * index);
        FCol1 = m_F_Block.block<3, 1>(0, (3 * index) + 1);
        FCol2 = m_F_Block.block<3, 1>(0, (3 * index) + 2);

        m_SPK2(6 * index) = m_GQWeight_det_J_0xi(index)*D0(0)*(0.5*FCol0.dot(FCol0) - 0.5); //Smat(0, 0)
        m_SPK2((6 * index) + 1) = m_GQWeight_det_J_0xi(index)*D0(1)*(0.5*FCol1.dot(FCol1) - 0.5); //Smat(1, 1)
        m_SPK2((6 * index) + 2) = m_GQWeight_det_J_0xi(index)*D0(2)*(0.5*FCol2.dot(FCol2) - 0.5); //Smat(2, 2)
        m_SPK2((6 * index) + 3) = m_GQWeight_det_J_0xi(index)*D0(3)*FCol1.dot(FCol2); //Smat(1, 2) = Smat(2, 1)
        m_SPK2((6 * index) + 4) = m_GQWeight_det_J_0xi(index)*D0(4)*FCol0.dot(FCol2); //Smat(0, 2) = Smat(2, 0)
        m_SPK2((6 * index) + 5) = m_GQWeight_det_J_0xi(index)*D0(5)*FCol0.dot(FCol1); //Smat(0, 1) = Smat(1, 0)

        P_transpose_scaled_Block.block<1, 3>(3 * index, 0).noalias() = m_SPK2(6 * index)*FCol0.transpose() + m_SPK2((6 * index) + 5)*FCol1.transpose() + m_SPK2((6 * index) + 4)*FCol2.transpose();
        P_transpose_scaled_Block.block<1, 3>(3 * index + 1, 0).noalias() = m_SPK2((6 * index) + 5)*FCol0.transpose() + m_SPK2((6 * index) + 1)*FCol1.transpose() + m_SPK2((6 * index) + 3)*FCol2.transpose();
        P_transpose_scaled_Block.block<1, 3>(3 * index + 2, 0).noalias() = m_SPK2((6 * index) + 4)*FCol0.transpose() + m_SPK2((6 * index) + 3)*FCol1.transpose() + m_SPK2((6 * index) + 2)*FCol2.transpose();
    }

    ChRowVectorN<double, 3> DvRow0 = Dv.row(0);
    ChRowVectorN<double, 3> DvRow1 = Dv.row(1);
    ChRowVectorN<double, 3> DvRow2 = Dv.row(2);

    for (unsigned int index = 16; index < 20; index++) {

        FCol0 = m_F_Block.block<3, 1>(0, (3 * index));
        FCol1 = m_F_Block.block<3, 1>(0, (3 * index) + 1);
        FCol2 = m_F_Block.block<3, 1>(0, (3 * index) + 2);

        Ediag(0) = m_GQWeight_det_J_0xi(index)*(0.5*FCol0.dot(FCol0) - 0.5);
        Ediag(1) = m_GQWeight_det_J_0xi(index)*(0.5*FCol1.dot(FCol1) - 0.5);
        Ediag(2) = m_GQWeight_det_J_0xi(index)*(0.5*FCol2.dot(FCol2) - 0.5);

        //m_Ediag.block((3 * (index-16)), 0, 3, 1).noalias() = Ediag;
        //Faster to cache these values one by one rather than using the block command
        m_Ediag((3 * (index - 16))) = Ediag(0);
        m_Ediag((3 * (index - 16)) + 1) = Ediag(1);
        m_Ediag((3 * (index - 16)) + 2) = Ediag(2);

        P_transpose_scaled_Block.block<1, 3>((3 * index), 0).noalias() = Ediag.dot(DvRow0)*FCol0.transpose();
        P_transpose_scaled_Block.block<1, 3>((3 * index) + 1, 0).noalias() = Ediag.dot(DvRow1)*FCol1.transpose();
        P_transpose_scaled_Block.block<1, 3>((3 * index) + 2, 0).noalias() = Ediag.dot(DvRow2)*FCol2.transpose();

        //m_Sdiag((3 * (index - 16))) = Ediag.dot(DvRow0);
        //P_transpose_scaled_Block.block((3 * index), 0, 1, 3).noalias() = m_Sdiag((3 * (index - 16)))*FCol0.transpose();
        //m_Ediag((3 * (index - 16)) + 1) = Ediag.dot(DvRow1);
        //P_transpose_scaled_Block.block((3 * index) + 1, 0, 1, 3).noalias() = m_Ediag((3 * (index - 16)) + 1)*FCol1.transpose();
        //m_Ediag((3 * (index - 16)) + 2) = Ediag.dot(DvRow2);
        //P_transpose_scaled_Block.block((3 * index) + 2, 0, 1, 3).noalias() = m_Ediag((3 * (index - 16)) + 2)*FCol2.transpose();
    }

    ChMatrixNM<double, 9, 3> QiCompact = m_SD_precompute*P_transpose_scaled_Block;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    Fi = QiReshaped;
}

void ChElementBeamANCF_MT27A::ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, unsigned int index, ChVectorN<double, 6>& D0, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {
    //ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    //ChMatrixNM<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_D0.block(0, 3*index, 9, 3);     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrixNM<double,3,3> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrix33<double> P_transpose_scaled;  //Transpose of the 1st Piola-Kirchoff stress tensor (non-symmetric tensor) Scaled by the determinate of the element Jacobian
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    //ChMatrixNM<double, 9, 3> QiCompact;     //The compact result of the final generalized internal force computation

    ////Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    //// matrix for potentially non-straight reference configurations
    //Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    //J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    //Sxi_D_0xi.noalias() = m_SD_precompute_D0.block(9 * index, 0, 9, 3);  //Faster to use this than to reference m_SD_precompute_D0 directly in the calculations 

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    //F.noalias() = e_bar.transpose()*m_SD_precompute_D0.block(0, 3 * index, 9, 3);
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);  //Faster than subtracting off identity separately term by term

                                                  //If linear Kelvin-Voigt viscoelastic material model is enabled (otherwise its pure linear elastic)
                                                  //Combine the viscous damping contribution to the Strain tensor with E (so E is no longer really "E")
    if (m_damping_enabled) {
        ChMatrix33<double> F_dot;
        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        //F_dot.noalias() = e_bar_dot.transpose()*m_SD_precompute_D0.block(0, 3 * index, 9, 3);

        //Combine alpha*E_dot with E (i.e E_combined = E + Alpha*E_dot)
        E += m_Alpha* 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
    }

    //2nd PK2 Stress tensor is symmetric just like the Green-Lagrange Strain tensor
    //Carry out the multiplication with the specific provided diagonal 6x6 stiffness tensor
    // (without explicitly forming the matrix multiplication in Voigt notation)
    S(0, 0) = D0(0)*E(0, 0);
    S(0, 1) = 2.0*D0(5)*E(0, 1);
    S(0, 2) = 2.0*D0(4)*E(0, 2);
    S(1, 0) = S(0, 1);
    S(1, 1) = D0(1)*E(1, 1);
    S(1, 2) = 2.0*D0(3)*E(1, 2);
    S(2, 0) = S(0, 2);
    S(2, 1) = S(1, 2);
    S(2, 2) = D0(2)*E(2, 2);

    //Calculate the transpose of the 1st Piola Kirchoff Stress Tensor
    //Scale the PK1 Stress Tensor by J_0xi.determinant() so that this multiplication is already included when calculating Qi
    // See the following reference for more details on why the PK1 Stress Tensor is
    // used for more efficient calculation of the generalized internal forces.
    //      J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
    //      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
    //      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.
    //P_transpose_scaled.noalias() = m_GQWeight_det_J_0xi_D0(index)*S*F.transpose(); //S is symmetric, so S' = S
    //QiCompact+= Sxi_D_0xi*P_transpose_scaled;

    //P_transpose_scaled.noalias() = m_GQWeight_det_J_0xi_D0(index)*S.selfadjointView<Eigen::Upper>()*F.transpose(); //S is symmetric, so S' = S
    //QiCompact += Sxi_D_0xi*P_transpose_scaled;

    //QiCompact += m_GQWeight_det_J_0xi_D0(index)*Sxi_D_0xi*S.selfadjointView<Eigen::Upper>()*F.transpose();
    QiCompact += m_GQWeight_det_J_0xi_D0(index)*Sxi_D_0xi*S*F.transpose();
    //QiCompact += m_GQWeight_det_J_0xi_D0(index)*m_SD_precompute_D0.block(0, 3 * index, 9, 3)*S*F.transpose();
}

void ChElementBeamANCF_MT27A::ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, unsigned int index, ChMatrix33<double>& Dv, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {
    //ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    //ChMatrixNM<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_Dv.block(0, 3*index, 9, 3);     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChVectorN<double, 3> S;                 //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Diagonal in this special case due to the structure of the 6x6 stiffness matrix
    ChMatrix33<double> P_transpose_scaled;  //Transpose of the 1st Piola-Kirchoff stress tensor (non-symmetric tensor) Scaled by the determinate of the element Jacobian
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    //ChMatrixNM<double, 9, 3> QiCompact;     //The compact result of the final generalized internal force computation

    ////Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    //// matrix for potentially non-straight reference configurations
    //Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    //J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    //Sxi_D_0xi.noalias() = m_SD_precompute_Dv.block(9 * index, 0, 9, 3);

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    //F.noalias() = e_bar.transpose()*m_SD_precompute_Dv.block(0, 3 * index, 9, 3);
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);

    //If linear Kelvin-Voigt viscoelastic material model is enabled (otherwise its pure linear elastic)
    //Combine the viscous damping contribution to the Strain tensor with E (so E is no longer really "E")
    if (m_damping_enabled) {
        ChMatrix33<double> F_dot;
        //ChMatrix33<double> alpha_E_dot;
        //ChVectorN<double, 6> epsilon_dot;
        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        //F_dot.noalias() = e_bar_dot.transpose()*m_SD_precompute_Dv.block(0, 3 * index, 9, 3);

        //Combine alpha E_dot with E (E_combined = E + Alpha*E)
        E += m_Alpha* 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
    }

    //2nd PK2 Stress tensor is symmetric just like the Green-Lagrange Strain tensor
    //In this case its a diagonal matrix, so just compute the diagonal entries
    S.noalias() = Dv*E.diagonal();

    //Calculate the transpose of the 1st Piola Kirchoff Stress Tensor
    //Scale the PK1 Stress Tensor by J_0xi.determinant() so that this multiplication is already included when calculating Qi
    // See the following reference for more details on why the PK1 Stress Tensor is
    // used for more efficient calculation of the generalized internal forces.
    //      J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
    //      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
    //      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.
    //P_transpose_scaled.row(0).noalias() = m_GQWeight_det_J_0xi_Dv(index)*F.col(0)*S(0);
    //P_transpose_scaled.row(1).noalias() = m_GQWeight_det_J_0xi_Dv(index)*F.col(1)*S(1);
    //P_transpose_scaled.row(2).noalias() = m_GQWeight_det_J_0xi_Dv(index)*F.col(2)*S(2);

    //P_transpose_scaled.noalias() = m_GQWeight_det_J_0xi_D0(index)*S.asDiagonal()*F.transpose(); //S is symmetric, so S' = S
    //QiCompact+= Sxi_D_0xi*P_transpose_scaled;

    QiCompact += m_GQWeight_det_J_0xi_D0(index)*Sxi_D_0xi*S.asDiagonal()*F.transpose();
    //QiCompact += m_GQWeight_det_J_0xi_D0(index)*m_SD_precompute_Dv.block(0, 3 * index, 9, 3)*S.asDiagonal()*F.transpose();

}

void ChElementBeamANCF_MT27A::ComputeInternalForcesSingleGQPnt(ChMatrixNM<double, 9, 3>& QiCompact, double xi, double eta, double zeta, ChMatrixNM<double, 6, 6>& D, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {
    ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi;     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChVectorN<double, 6> epsilon;           //Green-Lagrange strain tensor in Voigt notation
    ChVectorN<double, 6> sigma_PK2;         //2nd Piola-Kirchoff stress tensor in Voigt notation
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrix33<double> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrix33<double> P_transpose_scaled;  //Transpose of the 1st Piola-Kirchoff stress tensor (non-symmetric tensor) Scaled by the determinate of the element Jacobian
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    //ChMatrixNM<double, 9, 3> QiCompact;     //The compact result of the final generalized internal force computation

    //Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    // matrix for potentially non-straight reference configurations
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    Sxi_D_0xi.noalias() = Sxi_D*J_0xi.inverse();

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);
    epsilon(0) = E(0, 0);
    epsilon(1) = E(1, 1);
    epsilon(2) = E(2, 2);
    epsilon(3) = 2.0*E(1, 2);
    epsilon(4) = 2.0*E(0, 2);
    epsilon(5) = 2.0*E(0, 1);

    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrix33<double> F_dot;
        ChMatrix33<double> E_dot;
        ChVectorN<double, 6> epsilon_dot;
        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        E_dot.noalias() = 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
        epsilon_dot(0) = E_dot(0, 0);
        epsilon_dot(1) = E_dot(1, 1);
        epsilon_dot(2) = E_dot(2, 2);
        epsilon_dot(3) = 2.0*E_dot(1, 2);
        epsilon_dot(4) = 2.0*E_dot(0, 2);
        epsilon_dot(5) = 2.0*E_dot(0, 1);

        sigma_PK2.noalias() = D*(epsilon + m_Alpha*epsilon_dot);
    }
    else {                          //No damping, so just a linear elastic material law
        sigma_PK2.noalias() = D*epsilon;
    }

    //Now convert from Voigt notation back into matrix form
    S(0, 0) = sigma_PK2(0);
    S(1, 0) = sigma_PK2(5);
    S(2, 0) = sigma_PK2(4);
    S(0, 1) = sigma_PK2(5);
    S(1, 1) = sigma_PK2(1);
    S(2, 1) = sigma_PK2(3);
    S(0, 2) = sigma_PK2(4);
    S(1, 2) = sigma_PK2(3);
    S(2, 2) = sigma_PK2(2);

    //Calculate the transpose of the 1st Piola Kirchoff Stress Tensor
    //Scale the PK1 Stress Tensor by J_0xi.determinant() so that this multiplication is already included when calculating Qi
    // See the following reference for more details on why the PK1 Stress Tensor is
    // used for more efficient calculation of the generalized internal forces.
    //      J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
    //      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
    //      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.
    P_transpose_scaled.noalias() = -J_0xi.determinant()*S*F.transpose(); //S is symmetric, so S' = S
    QiCompact+= Sxi_D_0xi*P_transpose_scaled;
}

// -----------------------------------------------------------------------------
// Jacobians of internal forces
// -----------------------------------------------------------------------------

void ChElementBeamANCF_MT27A::ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, unsigned int index, ChVectorN<double, 6>& D0, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {

    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_D0.block(0, 3*index, 9, 3);  //Faster to use this than to reference m_SD_precompute_D0 directly in the calculations 
    ChMatrixNM<double, 3, 3> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrixNM<double, 3, 3> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrixNM<double, 3, 3> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrixNM<double, 3, 3> I_3x3;               //3x3 identity matrix
    ChMatrixNM<double, 6, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    ChMatrixNM<double, 6, 27> Scaled_Combined_partial_epsilon_partial_e;
    Eigen::Map<ChVectorN<double, 27>> Sxi_D_0xiReshaped(Sxi_D_0xi.data(), Sxi_D_0xi.size());
    ChMatrixNM<double, 27, 3> partial_e_tempA;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedA(partial_e_tempA.data(), partial_e_tempA.size());
    ChMatrixNM<double, 27, 3> partial_e_tempB;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedB(partial_e_tempB.data(), partial_e_tempB.size());
    ChMatrixNM<double, 27, 3> partial_e_tempC;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedC(partial_e_tempC.data(), partial_e_tempC.size());

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    //partial_e_tempA = Sxi_D_0xiReshaped*F.col(0).transpose();
    //partial_e_tempB = Sxi_D_0xiReshaped*F.col(1).transpose();
    //partial_e_tempC = Sxi_D_0xiReshaped*F.col(2).transpose();

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    ChRowVectorN<double, 3> Fcol0transpose = F.col(0).transpose();
    ChRowVectorN<double, 3> Fcol1transpose = F.col(1).transpose();
    ChRowVectorN<double, 3> Fcol2transpose = F.col(2).transpose();
    partial_e_tempA = Sxi_D_0xiReshaped*Fcol0transpose;
    partial_e_tempB = Sxi_D_0xiReshaped*Fcol1transpose;
    partial_e_tempC = Sxi_D_0xiReshaped*Fcol2transpose;

    partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
    partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrixNM<double, 3, 3> F_dot;
        ChMatrixNM<double, 3, 3> Scaled_Combined_F;
        Eigen::Map<ChRowVectorN<double, 9>> Scaled_Combined_FReshaped(F.data(), F.size());

        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;

        //Combine alpha*E_dot with E (i.e E_combined = E + Alpha*E_dot)
        E += m_Alpha* 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);

        Scaled_Combined_F = m_GQWeight_det_J_0xi_D0(index)*((Kfactor + m_Alpha*Rfactor)*F + (m_Alpha*Kfactor)*F_dot);

        ChRowVectorN<double, 3> Scaled_Combined_Fcol0transpose = Scaled_Combined_F.col(0).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol1transpose = Scaled_Combined_F.col(1).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol2transpose = Scaled_Combined_F.col(2).transpose();

        //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
        //combined with epsilon dot with the correct Jacobian weighting factors 
        //with respect to the nodal coordinates entry by entry in the vector epsilon
        //partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_F.col(0).transpose();
        //partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_F.col(1).transpose();
        //partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_F.col(2).transpose();
        partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_Fcol0transpose;
        partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_Fcol1transpose;
        partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_Fcol2transpose;

        Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*Scaled_Combined_partial_epsilon_partial_e;

    }
    else {                          //No damping, so just a linear elastic material law
        //Scaled_Combined_partial_epsilon_partial_e.row(0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(0))*partial_epsilon_partial_e.row(0);
        //Scaled_Combined_partial_epsilon_partial_e.row(1) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(1))*partial_epsilon_partial_e.row(1);
        //Scaled_Combined_partial_epsilon_partial_e.row(2) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(2))*partial_epsilon_partial_e.row(2);
        //Scaled_Combined_partial_epsilon_partial_e.row(3) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(3))*partial_epsilon_partial_e.row(3);
        //Scaled_Combined_partial_epsilon_partial_e.row(4) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(4))*partial_epsilon_partial_e.row(4);
        //Scaled_Combined_partial_epsilon_partial_e.row(5) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor*D0(5))*partial_epsilon_partial_e.row(5);

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += m_GQWeight_det_J_0xi_D0(index)*Kfactor*partial_epsilon_partial_e.transpose()*D0.asDiagonal()*partial_epsilon_partial_e;

        //No timing difference between the above methods 5/3/2020
    }



    //2nd PK2 Stress tensor is symmetric just like the Green-Lagrange Strain tensor
    //Carry out the multiplication with the specific provided diagonal 6x6 stiffness tensor
    // (without explicitly forming the matrix multiplication in Voigt notation)
    //Also include the weighting factors needed for the Jacobian calculation in the next step
    S(0, 0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*D0(0)*E(0, 0);
    S(1, 0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*2.0*D0(5)*E(0, 1);
    S(2, 0) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*2.0*D0(4)*E(0, 2);
    S(0, 1) = S(1, 0);
    S(1, 1) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*D0(1)*E(1, 1);
    S(2, 1) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*2.0*D0(3)*E(1, 2);
    S(0, 2) = S(2, 0);
    S(1, 2) = S(2, 1);
    S(2, 2) = (m_GQWeight_det_J_0xi_D0(index)*Kfactor)*D0(2)*E(2, 2);

    //Calculate and accumulate the compact component of the Jacobian.  This will need to 
    //be expanded to full size once all of the GQ points have been accounted for.  
    Jac_Compact += Sxi_D_0xi*S*Sxi_D_0xi.transpose();
}

void ChElementBeamANCF_MT27A::ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, unsigned int index, ChMatrix33<double>& Dv, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot) {

    ChMatrixNMc<double, 9, 3> Sxi_D_0xi = m_SD_precompute_Dv.block(0, 3*index, 9, 3);     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
                                            //Faster to use this than to reference m_SD_precompute_D0 directly in the calculations 
    ChMatrixNM<double, 3, 3> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChVectorN<double, 3> S;                 //2nd Piola-Kirchoff stress tensor (symmetric tensor) - Diagonal in this special case due to the structure of the 6x6 stiffness matrix
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    ChMatrixNM<double, 3, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    ChMatrixNM<double, 9, 3> partial_e_temp0;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped0(partial_e_temp0.data(), partial_e_temp0.size());
    ChMatrixNM<double, 9, 3> partial_e_temp1;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped1(partial_e_temp1.data(), partial_e_temp1.size());
    ChMatrixNM<double, 9, 3> partial_e_temp2;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 27>> partial_e_tempReshaped2(partial_e_temp2.data(), partial_e_temp2.size());


    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    //Only need the first 3 terms in this special case due to the structure of the 6x6 stiffness matrix
    //partial_e_temp0 = Sxi_D_0xi.col(0)*F.col(0).transpose();
    //partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
    //partial_e_temp1 = Sxi_D_0xi.col(1)*F.col(1).transpose();
    //partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
    //partial_e_temp2 = Sxi_D_0xi.col(2)*F.col(2).transpose();
    //partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    //Only need the first 3 terms in this special case due to the structure of the 6x6 stiffness matrix
    ChRowVectorN<double, 3> Fcol0transpose = F.col(0).transpose();
    ChRowVectorN<double, 3> Fcol1transpose = F.col(1).transpose();
    ChRowVectorN<double, 3> Fcol2transpose = F.col(2).transpose();
    partial_e_temp0 = Sxi_D_0xi.col(0)*Fcol0transpose;
    partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
    partial_e_temp1 = Sxi_D_0xi.col(1)*Fcol1transpose;
    partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
    partial_e_temp2 = Sxi_D_0xi.col(2)*Fcol2transpose;
    partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;


    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrix33<double> F_dot;
        ChMatrix33<double> E_dot;
        ChMatrix33<double> Scaled_Combined_F;
        ChMatrixNM<double, 3, 27> Scaled_Combined_partial_epsilon_partial_e;

        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        E_dot.noalias() = 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);

        S.noalias() = Dv*(E.diagonal() + m_Alpha*E_dot.diagonal());

        Scaled_Combined_F = m_GQWeight_det_J_0xi_Dv(index)*((Kfactor + m_Alpha*Rfactor)*F + (m_Alpha*Kfactor)*F_dot);

        ChRowVectorN<double, 3> Scaled_Combined_Fcol0transpose = Scaled_Combined_F.col(0).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol1transpose = Scaled_Combined_F.col(1).transpose();
        ChRowVectorN<double, 3> Scaled_Combined_Fcol2transpose = Scaled_Combined_F.col(2).transpose();

        //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
        //combined with epsilon dot with the correct Jacobian weighting factors 
        //with respect to the nodal coordinates entry by entry in the vector epsilon
        //partial_e_temp0 = Sxi_D_0xi.col(0)*Scaled_Combined_F.col(0).transpose();
        //Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
        //partial_e_temp1 = Sxi_D_0xi.col(1)*Scaled_Combined_F.col(1).transpose();
        //Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
        //partial_e_temp2 = Sxi_D_0xi.col(2)*Scaled_Combined_F.col(2).transpose();
        //Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;
        partial_e_temp0 = Sxi_D_0xi.col(0)*Scaled_Combined_Fcol0transpose;
        Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshaped0;
        partial_e_temp1 = Sxi_D_0xi.col(1)*Scaled_Combined_Fcol1transpose;
        Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshaped1;
        partial_e_temp2 = Sxi_D_0xi.col(2)*Scaled_Combined_Fcol2transpose;
        Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshaped2;

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*Dv*Scaled_Combined_partial_epsilon_partial_e;

    }
    else {                          //No damping, so just a linear elastic material law
        S.noalias() = Dv*E.diagonal();

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*(m_GQWeight_det_J_0xi_Dv(index)*Kfactor*Dv)*partial_epsilon_partial_e;
    }

    //Calculate and accumulate the compact component of the Jacobian.  This will need to 
    //be expanded to full size once all of the GQ points have been accounted for. 
    //Jac_Compact += Sxi_D_0xi*(m_GQWeight_det_J_0xi_Dv(index)*Kfactor*S.asDiagonal())*Sxi_D_0xi.transpose();
    //Jac_Compact += (m_GQWeight_det_J_0xi_Dv(index)*Kfactor)*(S(0)*Sxi_D_0xi.col(0)*Sxi_D_0xi.col(0).transpose() + 
    //    S(1)*Sxi_D_0xi.col(1)*Sxi_D_0xi.col(1).transpose() + S(2)*Sxi_D_0xi.col(2)*Sxi_D_0xi.col(2).transpose());
    Jac_Compact += m_GQWeight_det_J_0xi_Dv(index)*Kfactor*Sxi_D_0xi*S.asDiagonal()*Sxi_D_0xi.transpose();
}

void ChElementBeamANCF_MT27A::ComputeInternalJacobianSingleGQPnt(ChMatrixNM<double, 27, 27>& Jac_Dense, ChMatrixNM<double, 9, 9>& Jac_Compact, double Kfactor, double Rfactor, double GQ_weight, double xi, double eta, double zeta, ChMatrixNM<double, 6, 6>& D, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& e_bar_dot){
    ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
    ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives
    ChMatrixNMc<double, 9, 3> Sxi_D_0xi;     //Matrix of normalized shape function derivatives corrected for a potentially non-straight reference configuration
    ChVectorN<double, 6> epsilon;           //Green-Lagrange strain tensor in Voigt notation
    ChVectorN<double, 6> sigma_PK2;         //2nd Piola-Kirchoff stress tensor in Voigt notation
    ChMatrix33<double> F;                   //Deformation Gradient tensor (non-symmetric tensor)
    ChMatrix33<double> E;                   //Green-Lagrange strain tensor (symmetric tensor) - potentially combined with some damping contributions for speed
    ChMatrix33<double> S;                   //2nd Piola-Kirchoff stress tensor (symmetric tensor)
    ChMatrix33<double> I_3x3;               //3x3 identity matrix
    ChMatrixNM<double, 6, 27> partial_epsilon_partial_e;//partial derivate of the Green-Lagrange strain tensor in Voigt notation with respect to the nodal coordinates
    Eigen::Map<ChVectorN<double, 27>> Sxi_D_0xiReshaped(Sxi_D_0xi.data(), Sxi_D_0xi.size());
    ChMatrixNM<double, 27, 3> partial_e_tempA;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedA(partial_e_tempA.data(), partial_e_tempA.size());
    ChMatrixNM<double, 27, 3> partial_e_tempB;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedB(partial_e_tempB.data(), partial_e_tempB.size());
    ChMatrixNM<double, 27, 3> partial_e_tempC;//temporary matrix for calculating the partial derivatives of the strains with respect to the nodal coordinates
    Eigen::Map<ChRowVectorN<double, 81>> partial_e_tempReshapedC(partial_e_tempC.data(), partial_e_tempC.size());


    //Calculate the normalized shape function derivatives in matrix form and the corrected version of the 
    // matrix for potentially non-straight reference configurations
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi.noalias() = m_e0_bar.transpose()*Sxi_D;
    Sxi_D_0xi.noalias() = Sxi_D*J_0xi.inverse();
    double GQWeight_det_J_0xi = GQ_weight*J_0xi.determinant();

    //Calculate the deformation gradient tensor and the Green-Lagrange strain tensor
    I_3x3.setIdentity();
    F.noalias() = e_bar.transpose()*Sxi_D_0xi;
    E.noalias() = 0.5*(F.transpose()*F - I_3x3);
    epsilon(0) = E(0, 0);
    epsilon(1) = E(1, 1);
    epsilon(2) = E(2, 2);
    epsilon(3) = 2.0*E(1, 2);
    epsilon(4) = 2.0*E(0, 2);
    epsilon(5) = 2.0*E(0, 1);

    //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
    //with respect to the nodal coordinates entry by entry in the vector epsilon
    partial_e_tempA = Sxi_D_0xiReshaped*F.col(0).transpose();
    partial_e_tempB = Sxi_D_0xiReshaped*F.col(1).transpose();
    partial_e_tempC = Sxi_D_0xiReshaped*F.col(2).transpose();

    partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
    partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
    partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
    partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

    //Calculate the 2nd Piola-Kirchoff Stress tensor in Voigt notation
    if (m_damping_enabled) {        //If linear Kelvin-Voigt viscoelastic material model is enabled 
        ChMatrix33<double> F_dot;
        ChMatrix33<double> E_dot;
        ChVectorN<double, 6> epsilon_dot;
        ChMatrix33<double> Scaled_Combined_F;
        ChMatrixNM<double, 6, 27> Scaled_Combined_partial_epsilon_partial_e;

        F_dot.noalias() = e_bar_dot.transpose()*Sxi_D_0xi;
        E_dot.noalias() = 0.5* (F_dot.transpose()*F + F.transpose()*F_dot);
        epsilon_dot(0) = E_dot(0, 0);
        epsilon_dot(1) = E_dot(1, 1);
        epsilon_dot(2) = E_dot(2, 2);
        epsilon_dot(3) = 2.0*E_dot(1, 2);
        epsilon_dot(4) = 2.0*E_dot(0, 2);
        epsilon_dot(5) = 2.0*E_dot(0, 1);

        sigma_PK2.noalias() = D*(epsilon + m_Alpha*epsilon_dot);

        Scaled_Combined_F = GQWeight_det_J_0xi*((Kfactor + m_Alpha*Rfactor)*F + (m_Alpha*Kfactor)*F_dot);

        //Calculate the partial derivative of epsilon (Green-Lagrange strain tensor in Voigt notation) 
        //combined with epsilon dot with the correct Jacobian weighting factors 
        //with respect to the nodal coordinates entry by entry in the vector epsilon
        partial_e_tempA = Sxi_D_0xiReshaped*Scaled_Combined_F.col(0).transpose();
        partial_e_tempB = Sxi_D_0xiReshaped*Scaled_Combined_F.col(1).transpose();
        partial_e_tempC = Sxi_D_0xiReshaped*Scaled_Combined_F.col(2).transpose();

        Scaled_Combined_partial_epsilon_partial_e.row(0) = partial_e_tempReshapedA.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(1) = partial_e_tempReshapedB.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(2) = partial_e_tempReshapedC.block(0, 54, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(3).noalias() = partial_e_tempReshapedB.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 27, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(4).noalias() = partial_e_tempReshapedA.block(0, 54, 1, 27) + partial_e_tempReshapedC.block(0, 0, 1, 27);
        Scaled_Combined_partial_epsilon_partial_e.row(5).noalias() = partial_e_tempReshapedA.block(0, 27, 1, 27) + partial_e_tempReshapedB.block(0, 0, 1, 27);

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*D*Scaled_Combined_partial_epsilon_partial_e;

    }
    else {                          //No damping, so just a linear elastic material law
        sigma_PK2.noalias() = D*epsilon;

        //Calculate and accumulate the dense component of the Jacobian.
        Jac_Dense += partial_epsilon_partial_e.transpose()*(GQWeight_det_J_0xi*Kfactor*D)*partial_epsilon_partial_e;
    }

    //Now convert from Voigt notation back into matrix form
    S(0, 0) = sigma_PK2(0);
    S(1, 0) = sigma_PK2(5);
    S(2, 0) = sigma_PK2(4);
    S(0, 1) = sigma_PK2(5);
    S(1, 1) = sigma_PK2(1);
    S(2, 1) = sigma_PK2(3);
    S(0, 2) = sigma_PK2(4);
    S(1, 2) = sigma_PK2(3);
    S(2, 2) = sigma_PK2(2);

    //Calculate and accumulate the compact component of the Jacobian.  This will need to 
    //be expanded to full size once all of the GQ points have been accounted for.  
    Jac_Compact += (GQWeight_det_J_0xi*Kfactor)*Sxi_D_0xi*S*Sxi_D_0xi.transpose();

}

void ChElementBeamANCF_MT27A::ComputeInternalJacobians(ChMatrixNM<double, 27, 27>& JacobianMatrix, double Kfactor, double Rfactor) {
    // The integrated quantity represents the 27x27 Jacobian
    //      Kfactor * [K] + Rfactor * [R]


    ChVectorDynamic<double> FiOrignal(27);
    ChVectorDynamic<double> FiDelta(27);
    ChMatrixNMc<double, 9, 3> e_bar;
    //ChMatrixNMc<double, 9, 3> e_bar_dot;

    CalcCoordMatrix(e_bar);
    //CalcCoordDerivMatrix(e_bar_dot);

    double delta = 1e-6;

    //Compute the Jacobian via numerical differentiation of the generalized internal force vector
    //Since the generalized force vector due to gravity is a constant, it doesn't affect this 
    //Jacobian calculation
    //Runs faster if the internal force with or without damping calculations are not combined into the same function using the common calculations with an if statement for the damping in the middle to calculate the different P_transpose_scaled_Block components
    if (m_damping_enabled) {
        ChMatrixNMc<double, 9, 3> e_bar_dot;
        CalcCoordDerivMatrix(e_bar_dot);

        ComputeInternalForcesAtState(FiOrignal, e_bar, e_bar_dot);
        for (unsigned int i = 0; i < 27; i++) {
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) + delta;
            ComputeInternalForcesAtState(FiDelta, e_bar, e_bar_dot);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) - delta;

            e_bar_dot(i / 3, i % 3) = e_bar_dot(i / 3, i % 3) + delta;
            ComputeInternalForcesAtState(FiDelta, e_bar, e_bar_dot);
            JacobianMatrix.col(i) += -Rfactor / delta * (FiDelta - FiOrignal);
            e_bar_dot(i / 3, i % 3) = e_bar_dot(i / 3, i % 3) - delta;
        }
    }
    else {
        ComputeInternalForcesAtStateNoDamping(FiOrignal, e_bar);
        for (unsigned int i = 0; i < 27; i++) {
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) + delta;
            ComputeInternalForcesAtStateNoDamping(FiDelta, e_bar);
            JacobianMatrix.col(i) = -Kfactor / delta * (FiDelta - FiOrignal);
            e_bar(i / 3, i % 3) = e_bar(i / 3, i % 3) - delta;
        }
    }
}

void ChElementBeamANCF_MT27A::ComputeInternalJacobiansAnalytic(ChMatrixNM<double, 27, 27>& JacobianMatrix, double Kfactor, double Rfactor) {
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> e_bar_dot;

    CalcCoordMatrix(e_bar);
    CalcCoordDerivMatrix(e_bar_dot);

    //Set JacobianMatrix to zero since the results from each GQ point will be added to this matrix
    JacobianMatrix.setZero();
    ChMatrixNM<double, 9, 9> JacobianMatrixCompact;
    JacobianMatrixCompact.setZero();

    ChVectorN<double, 6> D0 = GetMaterial()->Get_D0();
    ChMatrix33<double> Dv = GetMaterial()->Get_Dv();

    //Calculate the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int index = 0; index < m_GQWeight_det_J_0xi_D0.size(); index++) {
        ComputeInternalJacobianSingleGQPnt(JacobianMatrix, JacobianMatrixCompact, -Kfactor, -Rfactor, index, D0, e_bar, e_bar_dot);
    }

    //Calculate the portion of the Selective Reduced Integration that account for the Poisson effect, but only on the beam axis
    if (GetStrainFormulation() == ChElementBeamANCF_MT27A::StrainFormulation::CMPoisson) {
        for (unsigned int index = 0; index < m_GQWeight_det_J_0xi_Dv.size(); index++) {
            ComputeInternalJacobianSingleGQPnt(JacobianMatrix, JacobianMatrixCompact, -Kfactor, -Rfactor, index, Dv, e_bar, e_bar_dot);
        }
    }

    //Inflate the Mass Matrix since it is stored in compact form.
    //In MATLAB notation:	 
    //JacobianMatrix(1:3:end,1:3:end) = JacobianMatrix(1:3:end,1:3:end) + JacobianMatrixCompact;
    //JacobianMatrix(2:3:end,2:3:end) = JacobianMatrix(2:3:end,2:3:end) + JacobianMatrixCompact;
    //JacobianMatrix(3:3:end,3:3:end) = JacobianMatrix(3:3:end,3:3:end) + JacobianMatrixCompact;
    for (unsigned int i = 0; i < 9; i++) {
        for (unsigned int j = 0; j < 9; j++) {
            JacobianMatrix(3 * i, 3 * j) += JacobianMatrixCompact(i, j);
            JacobianMatrix(3 * i + 1, 3 * j + 1) += JacobianMatrixCompact(i, j);
            JacobianMatrix(3 * i + 2, 3 * j + 2) += JacobianMatrixCompact(i, j);
        }
    }



}

// -----------------------------------------------------------------------------
// Shape functions
// -----------------------------------------------------------------------------

// 3x27 Sparse Form of the Normalized Shape Functions
// [s1*I_3x3, s2*I_3x3, s3*I_3x3, ...]
void ChElementBeamANCF_MT27A::Calc_Sxi(ChMatrixNM<double, 3, 27>& Sxi, double xi, double eta, double zeta) {
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
void ChElementBeamANCF_MT27A::Calc_Sxi_compact(ChVectorN<double, 9>& Sxi_compact, double xi, double eta, double zeta) {
    Sxi_compact(0) = 0.5*(xi*xi - xi);
    Sxi_compact(1) = 0.25*m_thicknessY*eta*(xi*xi - xi);
    Sxi_compact(2) = 0.25*m_thicknessZ*zeta*(xi*xi - xi);
    Sxi_compact(3) = 0.5*(xi*xi + xi);
    Sxi_compact(4) = 0.25*m_thicknessY*eta*(xi*xi + xi);
    Sxi_compact(5) = 0.25*m_thicknessZ*zeta*(xi*xi + xi);
    Sxi_compact(6) = 1.0 - xi*xi;
    Sxi_compact(7) = 0.5*m_thicknessY*eta*(1.0 - xi*xi);
    Sxi_compact(8) = 0.5*m_thicknessZ*zeta*(1.0 - xi*xi);
}

//Calculate the 27x3 Compact Shape Function Derivative Matrix modified by the inverse of the element Jacobian
//            [partial(s_1)/(partial xi)      partial(s_2)/(partial xi)     ...]T
//Sxi_D_0xi = [partial(s_1)/(partial eta)     partial(s_2)/(partial eta)    ...]  J_0xi^(-1)
//            [partial(s_1)/(partial zeta)    partial(s_2)/(partial zeta)   ...]
// See: J.Gerstmayr,A.A.Shabana,Efficient integration of the elastic forces and 
//      thin three-dimensional beam elements in the absolute nodal coordinate formulation, 
//      Proceedings of Multibody Dynamics 2005 ECCOMAS Thematic Conference, Madrid, Spain, 2005.

void ChElementBeamANCF_MT27A::Calc_Sxi_D(ChMatrixNMc<double, 9, 3>& Sxi_D, double xi, double eta, double zeta) {
    Sxi_D(0, 0) = xi - 0.5;
    Sxi_D(1, 0) = 0.25*m_thicknessY*eta*(2.0*xi - 1.0);
    Sxi_D(2, 0) = 0.25*m_thicknessZ*zeta*(2.0*xi - 1.0);
    Sxi_D(3, 0) = xi + 0.5;
    Sxi_D(4, 0) = 0.25*m_thicknessY*eta*(2.0*xi + 1.0);
    Sxi_D(5, 0) = 0.25*m_thicknessZ*zeta*(2.0*xi + 1.0);
    Sxi_D(6, 0) = -2.0*xi;
    Sxi_D(7, 0) = -m_thicknessY*eta*xi;
    Sxi_D(8, 0) = -m_thicknessZ*zeta*xi;

    Sxi_D(0, 1) = 0.0;
    Sxi_D(1, 1) = 0.25*m_thicknessY*(xi*xi - xi);
    Sxi_D(2, 1) = 0.0;
    Sxi_D(3, 1) = 0.0;
    Sxi_D(4, 1) = 0.25*m_thicknessY*(xi*xi + xi);
    Sxi_D(5, 1) = 0.0;
    Sxi_D(6, 1) = 0.0;
    Sxi_D(7, 1) = 0.5*m_thicknessY*(1 - xi*xi);
    Sxi_D(8, 1) = 0.0;

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.25*m_thicknessZ*(xi*xi - xi);
    Sxi_D(3, 2) = 0.0;
    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.25*m_thicknessZ*(xi*xi + xi);
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = 0.0;
    Sxi_D(8, 2) = 0.5*m_thicknessZ*(1 - xi*xi);
}


// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

void ChElementBeamANCF_MT27A::CalcCoordMatrix(ChMatrixNMc<double, 9, 3>& e) {
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

void ChElementBeamANCF_MT27A::CalcCoordVector(ChVectorN<double, 27>& e) {
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

void ChElementBeamANCF_MT27A::CalcCoordDerivMatrix(ChMatrixNMc<double, 9, 3>& edot) {
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

void ChElementBeamANCF_MT27A::CalcCoordDerivVector(ChVectorN<double, 27>& edot) {
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


//Calculate the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
void ChElementBeamANCF_MT27A::Calc_J_0xi(ChMatrix33<double>& J_0xi, double xi, double eta, double zeta) {
    ChMatrixNMc<double, 9, 3> Sxi_D;
    Calc_Sxi_D(Sxi_D, xi, eta, zeta);

    J_0xi = m_e0_bar.transpose()*Sxi_D;
}

//Calculate the determinate of the 3x3 Element Jacobian at the given point (xi,eta,zeta) in the element
double ChElementBeamANCF_MT27A::Calc_det_J_0xi(double xi, double eta, double zeta) {
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChMatrix33<double> J_0xi;

    Calc_Sxi_D(Sxi_D, xi, eta, zeta);
    J_0xi = m_e0_bar.transpose()*Sxi_D;
    return(J_0xi.determinant());
}

// -----------------------------------------------------------------------------
// Interface to ChElementShell base class
// -----------------------------------------------------------------------------
//ChVector<> ChElementBeamANCF_MT27A::EvaluateBeamSectionStrains() {
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
// void ChElementBeamANCF_MT27A::EvaluateSectionDisplacement(const double u,
//                                                    const double v,
//                                                    ChVector<>& u_displ,
//                                                    ChVector<>& u_rotaz) {
//    // this is not a corotational element, so just do:
//    EvaluateSectionPoint(u, v, displ, u_displ);
//    u_rotaz = VNULL;  // no angles.. this is ANCF (or maybe return here the slope derivatives?)
//}

void ChElementBeamANCF_MT27A::EvaluateSectionFrame(const double eta, ChVector<>& point, ChQuaternion<>& rot) {

    ChMatrixNMc<double, 9, 3> e_bar;
    ChVectorN<double, 9> Sxi_compact; 
    ChMatrixNMc<double, 9, 3> Sxi_D;

    CalcCoordMatrix(e_bar);
    Calc_Sxi_compact(Sxi_compact, eta, 0, 0);
    Calc_Sxi_D(Sxi_D, eta, 0, 0);

    // r = Se
    point = e_bar.transpose()*Sxi_compact;

    //Since ANCF does not use rotations, calculate an approximate
    //rotation based off the position vector gradients
    ChVector<double> BeamAxisTangent = e_bar.transpose()*Sxi_D.col(0);
    ChVector<double> CrossSectionY = e_bar.transpose()*Sxi_D.col(1);

    //Since the position vector gradients are not in general orthogonal,
    // set the Dx direction tangent to the beam axis and  
    // compute the Dy and Dz directions by using a
    // Gram-Schmidt orthonormalization, guided by the cross section Y direction
    ChMatrix33<> msect;
    msect.Set_A_Xdir(BeamAxisTangent, CrossSectionY);

    rot = msect.Get_A_quaternion();
}

// void ChElementBeamANCF_MT27A::EvaluateSectionPoint(const double u,
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
void ChElementBeamANCF_MT27A::LoadableGetStateBlock_x(int block_offset, ChState& mD) {
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
void ChElementBeamANCF_MT27A::LoadableGetStateBlock_w(int block_offset, ChStateDelta& mD) {
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
void ChElementBeamANCF_MT27A::LoadableStateIncrement(const unsigned int off_x,
                                               ChState& x_new,
                                               const ChState& x,
                                               const unsigned int off_v,
                                               const ChStateDelta& Dv) {
    m_nodes[0]->NodeIntStateIncrement(off_x, x_new, x, off_v, Dv);
    m_nodes[1]->NodeIntStateIncrement(off_x + 9, x_new, x, off_v + 9, Dv);
    m_nodes[2]->NodeIntStateIncrement(off_x + 18, x_new, x, off_v + 18, Dv);
}

//void ChElementBeamANCF_MT27A::EvaluateSectionVelNorm(double U, ChVector<>& Result) {
//    ShapeVector N;
//    ShapeFunctions(N, U, 0, 0);
//    for (unsigned int ii = 0; ii < 3; ii++) {
//        Result += N(ii * 3) * m_nodes[ii]->GetPos_dt();
//        Result += N(ii * 3 + 1) * m_nodes[ii]->GetPos_dt();
//    }
//}

// Get the pointers to the contained ChVariables, appending to the mvars vector.
void ChElementBeamANCF_MT27A::LoadableGetVariables(std::vector<ChVariables*>& mvars) {
    for (int i = 0; i < m_nodes.size(); ++i) {
        mvars.push_back(&m_nodes[i]->Variables());
        mvars.push_back(&m_nodes[i]->Variables_D());
        mvars.push_back(&m_nodes[i]->Variables_DD());
    }
}

// Evaluate N'*F , where N is the shape function evaluated at (U) coordinates of the centerline.
void ChElementBeamANCF_MT27A::ComputeNF(
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
void ChElementBeamANCF_MT27A::ComputeNF(
    const double U,              // parametric coordinate in volume
    const double V,              // parametric coordinate in volume
    const double W,              // parametric coordinate in volume
    ChVectorDynamic<>& Qi,       // Return result of N'*F  here, maybe with offset block_offset
    double& detJ,                // Return det[J] here
    const ChVectorDynamic<>& F,  // Input F vector, size is = n.field coords.
    ChVectorDynamic<>* state_x,  // if != 0, update state (pos. part) to this, then evaluate Q
    ChVectorDynamic<>* state_w   // if != 0, update state (speed part) to this, then evaluate Q
) {
    //Compute the generalized force vector for the applied force
    ChMatrixNM<double, 3, 27> Sxi;
    Calc_Sxi(Sxi, U, 0, 0);
    Qi = Sxi.transpose()*F.segment(0, 3);

    //Compute the generalized force vector for the applied moment
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChMatrixNM<double, 3, 9> Sxi_D_transpose;
    ChMatrix33<double> J_Cxi;
    ChMatrix33<double> J_Cxi_Inv;
    ChVectorN<double, 9> G_A;
    ChVectorN<double, 9> G_B;
    ChVectorN<double, 9> G_C;
    ChVectorN<double, 3> M_scaled = 0.5*F.segment(3, 3);

    CalcCoordMatrix(e_bar);
    Calc_Sxi_D(Sxi_D, U, V, W);

    J_Cxi.noalias() = e_bar.transpose()*Sxi_D;
    J_Cxi_Inv = J_Cxi.inverse();

    //Compute the unique pieces that make up the moment projection matrix "G"
    //See: Antonio M Recuero, Javier F Aceituno, Jose L Escalona, and Ahmed A Shabana. 
    //A nonlinear approach for modeling rail flexibility using the absolute nodal coordinate
    //formulation. Nonlinear Dynamics, 83(1-2):463-481, 2016.
    Sxi_D_transpose = Sxi_D.transpose();
    G_A = Sxi_D_transpose.row(0)*J_Cxi_Inv(0, 0) + Sxi_D_transpose.row(1)*J_Cxi_Inv(1, 0) + Sxi_D_transpose.row(2)*J_Cxi_Inv(2, 0);
    G_B = Sxi_D_transpose.row(0)*J_Cxi_Inv(0, 1) + Sxi_D_transpose.row(1)*J_Cxi_Inv(1, 1) + Sxi_D_transpose.row(2)*J_Cxi_Inv(2, 1);
    G_C = Sxi_D_transpose.row(0)*J_Cxi_Inv(0, 2) + Sxi_D_transpose.row(1)*J_Cxi_Inv(1, 2) + Sxi_D_transpose.row(2)*J_Cxi_Inv(2, 2);

    //Compute G'M without actually forming the complete matrix "G" (since it has a sparsity pattern to it)
    //// MIKE Clean-up when slicing becomes available in Eigen 3.4
    for (unsigned int i = 0; i < 9; i++) {
        Qi(3 * i) += M_scaled(1) * G_C(i) - M_scaled(2) * G_B(i);
        Qi((3 * i) + 1) += M_scaled(2) * G_A(i) - M_scaled(0) * G_C(i);
        Qi((3 * i) + 2) += M_scaled(0) * G_B(i) - M_scaled(1) * G_A(i);
    }

    //Compute the element Jacobian between the current configuration and the normalized configuration
    //This is different than the element Jacobian between the reference configuration and the normalized
    //  configuration used in the internal force calculations
    detJ = J_Cxi.determinant();
}

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// Calculate average element density (needed for ChLoaderVolumeGravity).
double ChElementBeamANCF_MT27A::GetDensity() {
    return GetMaterial()->Get_rho();
}

// Calculate tangent to the centerline at (U) coordinates.
ChVector<> ChElementBeamANCF_MT27A::ComputeTangent(const double U) {
    ChMatrixNMc<double, 9, 3> e_bar;
    ChMatrixNMc<double, 9, 3> Sxi_D;
    ChVector<> r_xi;

    CalcCoordMatrix(e_bar);
    Calc_Sxi_D(Sxi_D, U, 0, 0);
    r_xi = e_bar.transpose()*Sxi_D.col(1);

    return r_xi.GetNormalized();
}

////////////////////////////////////////////////////////////////

//#ifndef CH_QUADRATURE_STATIC_TABLES
#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_tables_MT27A(1, CH_QUADRATURE_STATIC_TABLES);
//#endif // !CH_QUADRATURE_STATIC_TABLES

ChQuadratureTables* ChElementBeamANCF_MT27A::GetStaticGQTables() {
    return &static_tables_MT27A;
}

////////////////////////////////////////////////////////////////

// ============================================================================
// Implementation of ChMaterialBeamANCF_MT27A methods
// ============================================================================

// Construct an isotropic material.
ChMaterialBeamANCF_MT27A::ChMaterialBeamANCF_MT27A(double rho,        // material density
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
ChMaterialBeamANCF_MT27A::ChMaterialBeamANCF_MT27A(double rho,            // material density
                                       const ChVector<>& E,   // elasticity moduli (E_x, E_y, E_z)
                                       const ChVector<>& nu,  // Poisson ratios (nu_xy, nu_xz, nu_yz)
                                       const ChVector<>& G,   // shear moduli (G_xy, G_xz, G_yz)
                                       const double& k1,      // Shear correction factor along beam local y axis
                                       const double& k2       // Shear correction factor along beam local z axis
                                       )
    : m_rho(rho) {
    Calc_D0_Dv(E, nu, G, k1, k2);
}

//Calculate the matrix form of two stiffness tensors used by the ANCF beam for selective reduced integration of the Poisson effect
void ChMaterialBeamANCF_MT27A::Calc_D0_Dv(const ChVector<>& E, const ChVector<>& nu, const ChVector<>& G, double k1, double k2) {

    //orthotropic material ref: http://homes.civil.aau.dk/lda/Continuum/material.pdf
    //except position of the shear terms is different to match the original ANCF reference paper

    double nu_12 = nu.x();
    double nu_13 = nu.y();
    double nu_23 = nu.z();
    double nu_21 = nu_12 * E.y() / E.x();
    double nu_31 = nu_13 * E.z() / E.x();
    double nu_32 = nu_23 * E.z() / E.y();
    double k = 1.0 - nu_23*nu_32 - nu_12*nu_21 - nu_13*nu_31 - nu_12*nu_23*nu_31 - nu_21*nu_32*nu_13;

    //Component of Stiffness Tensor that does not contain the Poisson Effect
    m_D0(0) = E.x();
    m_D0(1) = E.y();
    m_D0(2) = E.z();
    m_D0(3) = G.z();
    m_D0(4) = G.y()*k1;
    m_D0(5) = G.x()*k2;

    //Remaining components of the Stiffness Tensor that contain the Poisson Effect 
    m_Dv(0, 0) = E.x()*(1 - nu_23*nu_32) / k - m_D0(0);
    m_Dv(1, 0) = E.y()*(nu_13*nu_32 + nu_12) / k;
    m_Dv(2, 0) = E.z()*(nu_12*nu_23 + nu_13) / k;

    m_Dv(0, 1) = E.x()*(nu_23*nu_31 + nu_21) / k;
    m_Dv(1, 1) = E.y()*(1 - nu_13*nu_31) / k - m_D0(1);
    m_Dv(2, 1) = E.z()*(nu_13*nu_21 + nu_23) / k;

    m_Dv(0, 2) = E.x()*(nu_21*nu_32 + nu_31) / k;
    m_Dv(1, 2) = E.y()*(nu_12*nu_31 + nu_32) / k;
    m_Dv(2, 2) = E.z()*(1 - nu_12*nu_21) / k - m_D0(2);
}

}  // end of namespace fea
}  // end of namespace chrono
