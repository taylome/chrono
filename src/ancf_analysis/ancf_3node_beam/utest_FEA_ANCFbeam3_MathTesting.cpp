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
// Authors: Mike Taylor and Radu Serban
// =============================================================================
//
// Small Displacement, Small Deformation, Linear Isotropic Benchmark test for
// ANCF beam elements - Square cantilevered beam with a time-dependent tip load
//
// García-Vallejo, D., Mayo, J., Escalona, J. L., & Dominguez, J. (2004).
// Efficient evaluation of the elastic forces and the Jacobian in the absolute
// nodal coordinate formulation. Nonlinear Dynamics, 35(4), 313-329.
//
// =============================================================================

//#include "mkl.h"

#include <string>
//#include "unsupported/Eigen/FFT"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono_postprocess/ChGnuPlot.h"
#include "chrono_thirdparty/filesystem/path.h"

#include "chrono/fea/ChElementBeamANCF.h"
#include "chrono/fea/ChElementBeamANCF_MT01.h"
#include "chrono/fea/ChElementBeamANCF_MT02.h"
#include "chrono/fea/ChElementBeamANCF_MT03.h"
#include "chrono/fea/ChElementBeamANCF_MT04.h"
#include "chrono/fea/ChElementBeamANCF_MT05.h"
#include "chrono/fea/ChElementBeamANCF_MT06.h"
#include "chrono/fea/ChElementBeamANCF_MT07.h"
#include "chrono/fea/ChElementBeamANCF_MT08.h"
#include "chrono/fea/ChElementBeamANCF_MT09.h"
#include "chrono/fea/ChElementBeamANCF_MT10.h"
#include "chrono/fea/ChElementBeamANCF_MT11.h"
#include "chrono/fea/ChElementBeamANCF_MT12.h"
#include "chrono/fea/ChElementBeamANCF_MT13.h"
#include "chrono/fea/ChElementBeamANCF_MT14.h"
#include "chrono/fea/ChElementBeamANCF_MT15.h"
#include "chrono/fea/ChElementBeamANCF_MT16.h"
#include "chrono/fea/ChElementBeamANCF_MT17.h"
#include "chrono/fea/ChElementBeamANCF_MT18.h"
#include "chrono/fea/ChElementBeamANCF_MT19.h"
#include "chrono/fea/ChElementBeamANCF_MT20.h"
#include "chrono/fea/ChElementBeamANCF_MT21.h"
#include "chrono/fea/ChElementBeamANCF_MT22.h"
#include "chrono/fea/ChElementBeamANCF_MT23.h"
#include "chrono/fea/ChElementBeamANCF_MT24.h"
#include "chrono/fea/ChElementBeamANCF_MT25.h"
#include "chrono/fea/ChElementBeamANCF_MT26.h"
#include "chrono/fea/ChElementBeamANCF_MT27.h"
#include "chrono/fea/ChElementBeamANCF_MT28.h"
#include "chrono/fea/ChElementBeamANCF_MT30.h"
#include "chrono/fea/ChElementBeamANCF_MT31.h"
#include "chrono/fea/ChElementBeamANCF_MT32.h"

#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"
#include "chrono/physics/ChLoadContainer.h"
#include "chrono/fea/ChLoadsBeam.h"

#ifdef CHRONO_IRRLICHT
#include "chrono_irrlicht/ChIrrApp.h"
#endif

#ifdef CHRONO_MKL
#include "chrono_mkl/ChSolverMKL.h"
#endif

#ifdef CHRONO_MUMPS
#include "chrono_mumps/ChSolverMumps.h"
#endif

using namespace chrono;
using namespace chrono::fea;
using namespace postprocess;

// =============================================================================

/// Dense matrix with *fixed size* (known at compile time).
/// A ChMatrixNM is templated by the type of its coefficients and by the matrix dimensions (number of rows and columns).
template <typename T, int M, int N>
using ChMatrixNMc = Eigen::Matrix<T, M, N, Eigen::ColMajor>;

#define CH_QUADRATURE_STATIC_TABLES 10
ChQuadratureTables static_GQ_tables(1, CH_QUADRATURE_STATIC_TABLES);

// =============================================================================

void Calc_Sxi_D(ChMatrixNMc<double, 9, 3>& Sxi_D, double xi, double eta, double zeta, double thicknessY, double thicknessZ) {
    Sxi_D(0, 0) = xi - 0.5;
    Sxi_D(1, 0) = 0.25*thicknessY*eta*(2.0*xi - 1.0);
    Sxi_D(2, 0) = 0.25*thicknessZ*zeta*(2.0*xi - 1.0);
    Sxi_D(3, 0) = xi + 0.5;
    Sxi_D(4, 0) = 0.25*thicknessY*eta*(2.0*xi + 1.0);
    Sxi_D(5, 0) = 0.25*thicknessZ*zeta*(2.0*xi + 1.0);
    Sxi_D(6, 0) = -2.0*xi;
    Sxi_D(7, 0) = -thicknessY * eta*xi;
    Sxi_D(8, 0) = -thicknessZ * zeta*xi;

    Sxi_D(0, 1) = 0.0;
    Sxi_D(1, 1) = 0.25*thicknessY*(xi*xi - xi);
    Sxi_D(2, 1) = 0.0;
    Sxi_D(3, 1) = 0.0;
    Sxi_D(4, 1) = 0.25*thicknessY*(xi*xi + xi);
    Sxi_D(5, 1) = 0.0;
    Sxi_D(6, 1) = 0.0;
    Sxi_D(7, 1) = 0.5*thicknessY*(1 - xi * xi);
    Sxi_D(8, 1) = 0.0;

    Sxi_D(0, 2) = 0.0;
    Sxi_D(1, 2) = 0.0;
    Sxi_D(2, 2) = 0.25*thicknessZ*(xi*xi - xi);
    Sxi_D(3, 2) = 0.0;
    Sxi_D(4, 2) = 0.0;
    Sxi_D(5, 2) = 0.25*thicknessZ*(xi*xi + xi);
    Sxi_D(6, 2) = 0.0;
    Sxi_D(7, 2) = 0.0;
    Sxi_D(8, 2) = 0.5*thicknessZ*(1 - xi * xi);
}

//Precalculate constant matrices and scalars for the internal force calculations
void PrecomputeInternalForceMatricesWeights(ChMatrixNMc<double, 9, 3>& e0_bar,
                                            double thicknessY,
                                            double thicknessZ,
                                            ChVectorN<double, 16>& GQWeight_det_J_0xi_D0,
                                            ChMatrixNMc<double, 9, 16>& SD_precompute_D0_col0_block,
                                            ChMatrixNMc<double, 9, 16>& SD_precompute_D0_col1_block,
                                            ChMatrixNMc<double, 9, 16>& SD_precompute_D0_col2_block,
                                            ChVectorN<double, 4>& GQWeight_det_J_0xi_Dv,
                                            ChMatrixNMc<double, 9, 4>& SD_precompute_Dv_col0_block,
                                            ChMatrixNMc<double, 9, 4>& SD_precompute_Dv_col1_block,
                                            ChMatrixNMc<double, 9, 4>& SD_precompute_Dv_col2_block) {
    ChQuadratureTables* GQTable = &static_GQ_tables;
    unsigned int GQ_idx_xi = 3;       // 4 Point Gauss-Quadrature;
    unsigned int GQ_idx_eta_zeta = 1; // 2 Point Gauss-Quadrature;

    ChMatrixNMc<double, 9, 60> SD_precompute;             ///< Precomputed corrected normalized shape function derivative matrices for no Poisson Effect followed by Poisson Effect on the beam axis only
    ChVectorN<double, 20> GQWeight_det_J_0xi;             ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors for no Poisson Effect followed by Poisson Effect on the beam axis only

    ChMatrixNMc<double, 9, 48> SD_precompute_D0;          ///< Precomputed corrected normalized shape function derivative matrices for no Poisson Effect
    //ChVectorN<double, 16> GQWeight_det_J_0xi_D0;          ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factors for no Poisson Effect
    ChMatrixNMc<double, 9, 12> SD_precompute_Dv;           ///< Precomputed corrected normalized shape function derivative matrices for Poisson Effect on the beam axis only
    //ChVectorN<double, 4> GQWeight_det_J_0xi_Dv;           ///< Precomputed Gauss-Quadrature Weight & Element Jacobian scale factor for Poisson Effect on the beam axis only

    //ChMatrixNMc<double, 9, 16> SD_precompute_D0_col0_block;
    //ChMatrixNMc<double, 9, 16> SD_precompute_D0_col1_block;
    //ChMatrixNMc<double, 9, 16> SD_precompute_D0_col2_block;

    //Precalculate the matrices of normalized shape function derivatives corrected for a potentially non-straight reference configuration & GQ Weights times the determiate of the element Jacobian for later
    //Calculating the portion of the Selective Reduced Integration that does account for the Poisson effect
    for (unsigned int it_xi = 0; it_xi < GQTable->Lroots[GQ_idx_xi].size(); it_xi++) {
        for (unsigned int it_eta = 0; it_eta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_eta++) {
            for (unsigned int it_zeta = 0; it_zeta < GQTable->Lroots[GQ_idx_eta_zeta].size(); it_zeta++) {
                double GQ_weight = GQTable->Weight[GQ_idx_xi][it_xi] * GQTable->Weight[GQ_idx_eta_zeta][it_eta] * GQTable->Weight[GQ_idx_eta_zeta][it_zeta];
                double xi = GQTable->Lroots[GQ_idx_xi][it_xi];
                double eta = GQTable->Lroots[GQ_idx_eta_zeta][it_eta];
                double zeta = GQTable->Lroots[GQ_idx_eta_zeta][it_zeta];
                auto index = it_zeta + it_eta * GQTable->Lroots[GQ_idx_eta_zeta].size() + it_xi * GQTable->Lroots[GQ_idx_eta_zeta].size()*GQTable->Lroots[GQ_idx_eta_zeta].size();
                ChMatrix33<double> J_0xi;               //Element Jacobian between the reference configuration and normalized configuration
                ChMatrixNMc<double, 9, 3> Sxi_D;         //Matrix of normalized shape function derivatives

                Calc_Sxi_D(Sxi_D, xi, eta, zeta, thicknessY, thicknessZ);
                J_0xi.noalias() = e0_bar.transpose()*Sxi_D;

                ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
                SD_precompute_D0.block(0, 3 * index, 9, 3) = SD;

                SD_precompute_D0_col0_block.block(0, index, 9, 1) = SD.col(0);
                SD_precompute_D0_col1_block.block(0, index, 9, 1) = SD.col(1);
                SD_precompute_D0_col2_block.block(0, index, 9, 1) = SD.col(2);

                GQWeight_det_J_0xi_D0(index) = -J_0xi.determinant()*GQ_weight;
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

        Calc_Sxi_D(Sxi_D, xi, 0, 0, thicknessY, thicknessZ);
        J_0xi.noalias() = e0_bar.transpose()*Sxi_D;

        ChMatrixNMc<double, 9, 3> SD = Sxi_D * J_0xi.inverse();
        SD_precompute_Dv.block(0, 3 * it_xi, 9, 3) = SD;

        SD_precompute_Dv_col0_block.block(0, it_xi, 9, 1) = SD.col(0);
        SD_precompute_Dv_col1_block.block(0, it_xi, 9, 1) = SD.col(1);
        SD_precompute_Dv_col2_block.block(0, it_xi, 9, 1) = SD.col(2);

        GQWeight_det_J_0xi_Dv(it_xi) = -J_0xi.determinant()*GQ_weight;
    }

    SD_precompute.block(0, 0, 9, 48) = SD_precompute_D0;
    SD_precompute.block(0, 48, 9, 12) = SD_precompute_Dv;
    GQWeight_det_J_0xi.block(0, 0, 16, 1) = GQWeight_det_J_0xi_D0;
    GQWeight_det_J_0xi.block(16, 0, 4, 1) = GQWeight_det_J_0xi_Dv;
}

void Check_No_Damping_Internal_Force(ChMatrixNMc<double, 9, 3>& e0_bar, ChMatrixNMc<double, 9, 3>& e_bar) {
    double L = 1;   //m
    double W = 0.1; //m
    double H = 0.1; //m
    double E = 210e9; //Pa
    double nu = 0.3;
    double k2 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross - section
    double k3 = k2;
    double G = E / (2 * (1 + nu));
    ChVectorN<double, 6> D0;
    D0 << E, E, E, G, G*k2, G*k3;

    ChMatrixNM<double, 3, 3> Dv;
    Dv << 2 * nu, 1, 1,
        1, 2 * nu, 1,
        1, 1, 2 * nu;
    Dv *= E * nu / ((1 + nu)*(1 - 2 * nu));

    //ChMatrixNMc<double, 9, 3> e0_bar;
    //e0_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1;

    //ChMatrixNMc<double, 9, 3> e_bar;
    //e_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0.001,
    //    0, 1, 0,
    //    0, 0, 1;

    ChVectorN<double, 16> GQWeight_det_J_0xi_D0;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col0_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col1_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col2_block;
    ChVectorN<double, 4> GQWeight_det_J_0xi_Dv;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col0_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col1_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col2_block;


    PrecomputeInternalForceMatricesWeights(e0_bar, W, H, GQWeight_det_J_0xi_D0,
        SD_precompute_D0_col0_block, SD_precompute_D0_col1_block, SD_precompute_D0_col2_block,
        GQWeight_det_J_0xi_Dv, SD_precompute_Dv_col0_block, SD_precompute_Dv_col1_block, SD_precompute_Dv_col2_block);

    // =============================================================================

    ChMatrixNMc<double, 16, 3> Fcol0_Transpose_D0_Block = SD_precompute_D0_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol1_Transpose_D0_Block = SD_precompute_D0_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol2_Transpose_D0_Block = SD_precompute_D0_col2_block.transpose()*e_bar;

    ChVectorN<double, 16> SPK2_0_D0_Block = 0.5*D0(0)*GQWeight_det_J_0xi_D0.array()*(Fcol0_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 16> SPK2_1_D0_Block = 0.5*D0(1)*GQWeight_det_J_0xi_D0.array()*(Fcol1_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 16> SPK2_2_D0_Block = 0.5*D0(2)*GQWeight_det_J_0xi_D0.array()*(Fcol2_Transpose_D0_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 16> SPK2_3_D0_Block = D0(3)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol1_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum());
    ChVectorN<double, 16> SPK2_4_D0_Block = D0(4)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum());
    ChVectorN<double, 16> SPK2_5_D0_Block = D0(5)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol1_Transpose_D0_Block).rowwise().sum());

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_0_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_1_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_2_D0_Block.array();

    // =============================================================================

    ChMatrixNMc<double, 4, 3> Fcol0_Transpose_Dv_Block = SD_precompute_Dv_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol1_Transpose_Dv_Block = SD_precompute_Dv_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol2_Transpose_Dv_Block = SD_precompute_Dv_col2_block.transpose()*e_bar;

    ChVectorN<double, 4> Ediag_0_Dv_Block = 0.5 * GQWeight_det_J_0xi_Dv.array() * (Fcol0_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 4> Ediag_1_Dv_Block = 0.5 * GQWeight_det_J_0xi_Dv.array() * (Fcol1_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);
    ChVectorN<double, 4> Ediag_2_Dv_Block = 0.5 * GQWeight_det_J_0xi_Dv.array() * (Fcol2_Transpose_Dv_Block.rowwise().squaredNorm().array() - 1);

    ChVectorN<double, 4> Sdiag_0_Dv_Block = Dv(0, 0)*Ediag_0_Dv_Block + Dv(1, 0)*Ediag_1_Dv_Block + Dv(2, 0)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block = Dv(0, 1)*Ediag_0_Dv_Block + Dv(1, 1)*Ediag_1_Dv_Block + Dv(2, 1)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block = Dv(0, 2)*Ediag_0_Dv_Block + Dv(1, 2)*Ediag_1_Dv_Block + Dv(2, 2)*Ediag_2_Dv_Block;

    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block =
        Fcol0_Transpose_Dv_Block.array().colwise()*Sdiag_0_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block =
        Fcol1_Transpose_Dv_Block.array().colwise()*Sdiag_1_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block =
        Fcol2_Transpose_Dv_Block.array().colwise()*Sdiag_2_Dv_Block.array();

    // =============================================================================

    ChMatrixNM<double, 9, 3> QiCompact =
        SD_precompute_D0_col0_block * Ptransposed_scaled_row0_D0_Block +
        SD_precompute_D0_col1_block * Ptransposed_scaled_row1_D0_Block +
        SD_precompute_D0_col2_block * Ptransposed_scaled_row2_D0_Block +
        SD_precompute_Dv_col0_block * Ptransposed_scaled_row0_Dv_Block +
        SD_precompute_Dv_col1_block * Ptransposed_scaled_row1_Dv_Block +
        SD_precompute_Dv_col2_block * Ptransposed_scaled_row2_Dv_Block;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    ChVectorN<double, 27> Fi = QiReshaped;

    std::cout << Fi << std::endl;

}


void Check_Damping_Internal_Force(ChMatrixNMc<double, 9, 3>& e0_bar, ChMatrixNMc<double, 9, 3>& e_bar, ChMatrixNMc<double, 9, 3>& edot_bar) {
    double L = 1;   //m
    double W = 0.1; //m
    double H = 0.1; //m
    double E = 210e9; //Pa
    double nu = 0.3;
    double k2 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross - section
    double k3 = k2;
    double G = E / (2 * (1 + nu));

    double alpha = 0.01;

    ChVectorN<double, 6> D0;
    D0 << E, E, E, G, G*k2, G*k3;

    ChMatrixNM<double, 3, 3> Dv;
    Dv << 2 * nu, 1, 1,
        1, 2 * nu, 1,
        1, 1, 2 * nu;
    Dv *= E * nu / ((1 + nu)*(1 - 2 * nu));

    //ChMatrixNMc<double, 9, 3> e0_bar;
    //e0_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1;

    //ChMatrixNMc<double, 9, 3> e_bar;
    //e_bar << 0, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    1, 0, 0,
    //    0, 1, 0,
    //    0, 0, 1,
    //    0.5, 0, 0.001,
    //    0, 1, 0,
    //    0, 0, 1;

    ChVectorN<double, 16> GQWeight_det_J_0xi_D0;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col0_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col1_block;
    ChMatrixNMc<double, 9, 16> SD_precompute_D0_col2_block;
    ChVectorN<double, 4> GQWeight_det_J_0xi_Dv;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col0_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col1_block;
    ChMatrixNMc<double, 9, 4> SD_precompute_Dv_col2_block;


    PrecomputeInternalForceMatricesWeights(e0_bar, W, H, GQWeight_det_J_0xi_D0,
        SD_precompute_D0_col0_block, SD_precompute_D0_col1_block, SD_precompute_D0_col2_block,
        GQWeight_det_J_0xi_Dv, SD_precompute_Dv_col0_block, SD_precompute_Dv_col1_block, SD_precompute_Dv_col2_block);

    // =============================================================================

    ChMatrixNMc<double, 16, 3> Fcol0_Transpose_D0_Block = SD_precompute_D0_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol1_Transpose_D0_Block = SD_precompute_D0_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fcol2_Transpose_D0_Block = SD_precompute_D0_col2_block.transpose()*e_bar;
    ChMatrixNMc<double, 16, 3> Fdotcol0_Transpose_D0_Block = SD_precompute_D0_col0_block.transpose()*edot_bar;
    ChMatrixNMc<double, 16, 3> Fdotcol1_Transpose_D0_Block = SD_precompute_D0_col1_block.transpose()*edot_bar;
    ChMatrixNMc<double, 16, 3> Fdotcol2_Transpose_D0_Block = SD_precompute_D0_col2_block.transpose()*edot_bar;

    ChVectorN<double, 16> SPK2_0_D0_Block = D0(0)*GQWeight_det_J_0xi_D0.array()*(0.5*Fcol0_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum().array());
    ChVectorN<double, 16> SPK2_1_D0_Block = D0(1)*GQWeight_det_J_0xi_D0.array()*(0.5*Fcol1_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum().array());
    ChVectorN<double, 16> SPK2_2_D0_Block = D0(2)*GQWeight_det_J_0xi_D0.array()*(0.5*Fcol2_Transpose_D0_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum().array());
    ChVectorN<double, 16> SPK2_3_D0_Block = D0(3)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol1_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum() + alpha * (Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum() + Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum()));
    ChVectorN<double, 16> SPK2_4_D0_Block = D0(4)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol2_Transpose_D0_Block).rowwise().sum() + alpha * (Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol2_Transpose_D0_Block).rowwise().sum() + Fcol2_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum()));
    ChVectorN<double, 16> SPK2_5_D0_Block = D0(5)*GQWeight_det_J_0xi_D0.cwiseProduct(Fcol0_Transpose_D0_Block.cwiseProduct(Fcol1_Transpose_D0_Block).rowwise().sum() + alpha * (Fcol0_Transpose_D0_Block.cwiseProduct(Fdotcol1_Transpose_D0_Block).rowwise().sum() + Fcol1_Transpose_D0_Block.cwiseProduct(Fdotcol0_Transpose_D0_Block).rowwise().sum()));

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row0_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_0_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row1_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_5_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_1_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array();

    ChMatrixNM<double, 16, 3> Ptransposed_scaled_row2_D0_Block =
        Fcol0_Transpose_D0_Block.array().colwise()*SPK2_4_D0_Block.array() +
        Fcol1_Transpose_D0_Block.array().colwise()*SPK2_3_D0_Block.array() +
        Fcol2_Transpose_D0_Block.array().colwise()*SPK2_2_D0_Block.array();

    // =============================================================================

    ChMatrixNMc<double, 4, 3> Fcol0_Transpose_Dv_Block = SD_precompute_Dv_col0_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol1_Transpose_Dv_Block = SD_precompute_Dv_col1_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fcol2_Transpose_Dv_Block = SD_precompute_Dv_col2_block.transpose()*e_bar;
    ChMatrixNMc<double, 4, 3> Fdotcol0_Transpose_Dv_Block = SD_precompute_Dv_col0_block.transpose()*edot_bar;
    ChMatrixNMc<double, 4, 3> Fdotcol1_Transpose_Dv_Block = SD_precompute_Dv_col1_block.transpose()*edot_bar;
    ChMatrixNMc<double, 4, 3> Fdotcol2_Transpose_Dv_Block = SD_precompute_Dv_col2_block.transpose()*edot_bar;

    ChVectorN<double, 4> Ediag_0_Dv_Block = GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol0_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol0_Transpose_Dv_Block.cwiseProduct(Fdotcol0_Transpose_Dv_Block).rowwise().sum().array());
    ChVectorN<double, 4> Ediag_1_Dv_Block = GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol1_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol1_Transpose_Dv_Block.cwiseProduct(Fdotcol1_Transpose_Dv_Block).rowwise().sum().array());
    ChVectorN<double, 4> Ediag_2_Dv_Block = GQWeight_det_J_0xi_Dv.array() * (0.5*Fcol2_Transpose_Dv_Block.rowwise().squaredNorm().array() - 0.5 + alpha * Fcol2_Transpose_Dv_Block.cwiseProduct(Fdotcol2_Transpose_Dv_Block).rowwise().sum().array());

    ChVectorN<double, 4> Sdiag_0_Dv_Block = Dv(0, 0)*Ediag_0_Dv_Block + Dv(1, 0)*Ediag_1_Dv_Block + Dv(2, 0)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_1_Dv_Block = Dv(0, 1)*Ediag_0_Dv_Block + Dv(1, 1)*Ediag_1_Dv_Block + Dv(2, 1)*Ediag_2_Dv_Block;
    ChVectorN<double, 4> Sdiag_2_Dv_Block = Dv(0, 2)*Ediag_0_Dv_Block + Dv(1, 2)*Ediag_1_Dv_Block + Dv(2, 2)*Ediag_2_Dv_Block;

    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row0_Dv_Block =
        Fcol0_Transpose_Dv_Block.array().colwise()*Sdiag_0_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row1_Dv_Block =
        Fcol1_Transpose_Dv_Block.array().colwise()*Sdiag_1_Dv_Block.array();
    ChMatrixNM<double, 4, 3> Ptransposed_scaled_row2_Dv_Block =
        Fcol2_Transpose_Dv_Block.array().colwise()*Sdiag_2_Dv_Block.array();

    // =============================================================================

    ChMatrixNM<double, 9, 3> QiCompact =
        SD_precompute_D0_col0_block * Ptransposed_scaled_row0_D0_Block +
        SD_precompute_D0_col1_block * Ptransposed_scaled_row1_D0_Block +
        SD_precompute_D0_col2_block * Ptransposed_scaled_row2_D0_Block +
        SD_precompute_Dv_col0_block * Ptransposed_scaled_row0_Dv_Block +
        SD_precompute_Dv_col1_block * Ptransposed_scaled_row1_Dv_Block +
        SD_precompute_Dv_col2_block * Ptransposed_scaled_row2_Dv_Block;
    Eigen::Map<ChVectorN<double, 27>> QiReshaped(QiCompact.data(), QiCompact.size());
    ChVectorN<double, 27> Fi = QiReshaped;

    std::cout << Fi << std::endl;

}

int main(int argc, char* argv[]) {
    
    ChMatrixNMc<double, 9, 3> e0_bar;
    e0_bar << 0, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0.5, 0, 0,
        0, 1, 0,
        0, 0, 1;

    ChMatrixNMc<double, 9, 3> e_bar = e0_bar;
    ChMatrixNMc<double, 9, 3> edot_bar;
    edot_bar.setZero();

    std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
    Check_No_Damping_Internal_Force(e0_bar, e_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - Small Displacement, No Velocity = " << std::endl;
    e_bar(6, 2) = 0.001;
    Check_No_Damping_Internal_Force(e0_bar, e_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - No Displacement, No Velocity (With Damping Code) = " << std::endl;
    e_bar(6, 2) = 0.0;
    Check_Damping_Internal_Force(e0_bar, e_bar, edot_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - Small Displacement, No Velocity (With Damping Code) = " << std::endl;
    e_bar(6, 2) = 0.001;
    Check_Damping_Internal_Force(e0_bar, e_bar, edot_bar);

    std::cout << std::endl << std::endl << "---------------------------" << std::endl << std::endl;
    std::cout << "Generalized Internal Force - No Displacement, Small Velocity (With Damping Code) = " << std::endl;
    e_bar(6, 2) = 0.0;
    edot_bar(6, 2) = 0.001;
    Check_Damping_Internal_Force(e0_bar, e_bar, edot_bar);


    return(0);
}
