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
// Authors: Alessandro Tasora, Radu Serban
// =============================================================================

#include "chrono/physics/ChLinkMate.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMate)

void ChLinkMate::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMate>();

    // serialize parent class
    ChLink::ArchiveOut(archive_out);

    // serialize all member data:
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMate::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMate>();

    // deserialize parent class
    ChLink::ArchiveIn(archive_in);

    // deserialize all member data:
}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateGeneric)

ChLinkMateGeneric::ChLinkMateGeneric(bool mc_x, bool mc_y, bool mc_z, bool mc_rx, bool mc_ry, bool mc_rz)
    : c_x(mc_x), c_y(mc_y), c_z(mc_z), c_rx(mc_rx), c_ry(mc_ry), c_rz(mc_rz) {
    SetupLinkMask();
}

ChLinkMateGeneric::ChLinkMateGeneric(const ChLinkMateGeneric& other) : ChLinkMate(other) {
    c_x = other.c_x;
    c_y = other.c_y;
    c_z = other.c_z;
    c_rx = other.c_rx;
    c_ry = other.c_ry;
    c_rz = other.c_rz;

    SetupLinkMask();
}


void ChLinkMateGeneric::SetConstrainedCoords(bool mc_x, bool mc_y, bool mc_z, bool mc_rx, bool mc_ry, bool mc_rz) {
    c_x = mc_x;
    c_y = mc_y;
    c_z = mc_z;
    c_rx = mc_rx;
    c_ry = mc_ry;
    c_rz = mc_rz;

    SetupLinkMask();
}

void ChLinkMateGeneric::SetupLinkMask() {
    int nc = 0;
    if (c_x)
        nc++;
    if (c_y)
        nc++;
    if (c_z)
        nc++;
    if (c_rx)
        nc++;
    if (c_ry)
        nc++;
    if (c_rz)
        nc++;

    mask.ResetNconstr(nc);

    C.setZero(nc);

    P = ChMatrix33<>(0.5);

    ChangedLinkMask();
}

void ChLinkMateGeneric::ChangedLinkMask() {
    ndoc = mask.GetMaskDoc();
    ndoc_c = mask.GetMaskDoc_c();
    ndoc_d = mask.GetMaskDoc_d();
}

void ChLinkMateGeneric::SetDisabled(bool mdis) {
    ChLinkMate::SetDisabled(mdis);

    if (mask.SetAllDisabled(mdis) > 0)
        ChangedLinkMask();
}

void ChLinkMateGeneric::SetBroken(bool mbro) {
    ChLinkMate::SetBroken(mbro);

    if (mask.SetAllBroken(mbro) > 0)
        ChangedLinkMask();
}

int ChLinkMateGeneric::RestoreRedundant() {
    int mchanges = mask.RestoreRedundant();
    if (mchanges)
        ChangedLinkMask();
    return mchanges;
}

void ChLinkMateGeneric::Update(double mytime, bool update_assets) {
    // Inherit time changes of parent class (ChLink), basically doing nothing :)
    ChLink::Update(mytime, update_assets);

    if (this->Body1 && this->Body2) {
        this->mask.SetTwoBodiesVariables(&Body1->Variables(), &Body2->Variables());

        ChFrame<> F1_W = this->frame1 >> (*this->Body1);
        ChFrame<> F2_W = this->frame2 >> (*this->Body2);
        ChFrame<> F1_wrt_F2;
        F2_W.TransformParentToLocal(F1_W, F1_wrt_F2);
        // Now 'F1_wrt_F2' contains the position/rotation of frame 1 respect to frame 2, in frame 2 coords.

        ChMatrix33<> Jx1 = F2_W.GetRotMat().transpose();
        ChMatrix33<> Jx2 = -F2_W.GetRotMat().transpose();

        ChMatrix33<> Jr1 = -F2_W.GetRotMat().transpose() * Body1->GetRotMat() * ChStarMatrix33<>(frame1.GetPos());
        ChVector3d r12_B2 = Body2->GetRotMat().transpose() * (F1_W.GetPos() - F2_W.GetPos());
        ChMatrix33<> Jr2 = this->frame2.GetRotMat().transpose() * ChStarMatrix33<>(frame2.GetPos() + r12_B2);

        // Premultiply by Jw1 and Jw2 by P = 0.5 * [Fp(q_resid^*)]'.bottomRow(3) to get residual as imaginary part of a
        // quaternion. For small misalignment this effect is almost insignificant because P ~= [I33], but otherwise it
        // is needed (if you want to use the stabilization term - if not, you can live without).
        this->P = 0.5 * (ChMatrix33<>(F1_wrt_F2.GetRot().e0()) + ChStarMatrix33<>(F1_wrt_F2.GetRot().GetVector()));

        ChMatrix33<> Jw1 = this->P.transpose() * F2_W.GetRotMat().transpose() * Body1->GetRotMat();
        ChMatrix33<> Jw2 = -this->P.transpose() * F2_W.GetRotMat().transpose() * Body2->GetRotMat();

        // Another equivalent expression:
        // ChMatrix33<> Jw1 = this->P * F1_W.GetRotMat().transpose() * Body1->GetRotMat();
        // ChMatrix33<> Jw2 = -this->P * F1_W.GetRotMat().transpose() * Body2->GetRotMat();

        // The Jacobian matrix of constraint is:
        // Cq = [ Jx1,  Jr1,  Jx2,  Jr2 ]
        //      [   0,  Jw1,    0,  Jw2 ]

        int nc = 0;

        if (c_x) {
            C(nc) = F1_wrt_F2.GetPos().x();
            mask.Constr_N(nc).Get_Cq_a().segment(0, 3) = Jx1.row(0);
            mask.Constr_N(nc).Get_Cq_a().segment(3, 3) = Jr1.row(0);
            mask.Constr_N(nc).Get_Cq_b().segment(0, 3) = Jx2.row(0);
            mask.Constr_N(nc).Get_Cq_b().segment(3, 3) = Jr2.row(0);
            nc++;
        }
        if (c_y) {
            C(nc) = F1_wrt_F2.GetPos().y();
            mask.Constr_N(nc).Get_Cq_a().segment(0, 3) = Jx1.row(1);
            mask.Constr_N(nc).Get_Cq_a().segment(3, 3) = Jr1.row(1);
            mask.Constr_N(nc).Get_Cq_b().segment(0, 3) = Jx2.row(1);
            mask.Constr_N(nc).Get_Cq_b().segment(3, 3) = Jr2.row(1);
            nc++;
        }
        if (c_z) {
            C(nc) = F1_wrt_F2.GetPos().z();
            mask.Constr_N(nc).Get_Cq_a().segment(0, 3) = Jx1.row(2);
            mask.Constr_N(nc).Get_Cq_a().segment(3, 3) = Jr1.row(2);
            mask.Constr_N(nc).Get_Cq_b().segment(0, 3) = Jx2.row(2);
            mask.Constr_N(nc).Get_Cq_b().segment(3, 3) = Jr2.row(2);
            nc++;
        }
        if (c_rx) {
            C(nc) = F1_wrt_F2.GetRot().e1();
            mask.Constr_N(nc).Get_Cq_a().setZero();
            mask.Constr_N(nc).Get_Cq_b().setZero();
            mask.Constr_N(nc).Get_Cq_a().segment(3, 3) = Jw1.row(0);
            mask.Constr_N(nc).Get_Cq_b().segment(3, 3) = Jw2.row(0);
            nc++;
        }
        if (c_ry) {
            C(nc) = F1_wrt_F2.GetRot().e2();
            mask.Constr_N(nc).Get_Cq_a().setZero();
            mask.Constr_N(nc).Get_Cq_b().setZero();
            mask.Constr_N(nc).Get_Cq_a().segment(3, 3) = Jw1.row(1);
            mask.Constr_N(nc).Get_Cq_b().segment(3, 3) = Jw2.row(1);
            nc++;
        }
        if (c_rz) {
            C(nc) = F1_wrt_F2.GetRot().e3();
            mask.Constr_N(nc).Get_Cq_a().setZero();
            mask.Constr_N(nc).Get_Cq_b().setZero();
            mask.Constr_N(nc).Get_Cq_a().segment(3, 3) = Jw1.row(2);
            mask.Constr_N(nc).Get_Cq_b().segment(3, 3) = Jw2.row(2);
            nc++;
        }
    }
}

void ChLinkMateGeneric::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                   std::shared_ptr<ChBodyFrame> mbody2,
                                   ChFrame<> mabsframe) {
    this->Initialize(mbody1, mbody2, false, mabsframe, mabsframe);
}

void ChLinkMateGeneric::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                   std::shared_ptr<ChBodyFrame> mbody2,
                                   bool pos_are_relative,
                                   ChFrame<> mpos1,
                                   ChFrame<> mpos2) {
    assert(mbody1.get() != mbody2.get());

    this->Body1 = mbody1.get();
    this->Body2 = mbody2.get();
    // this->SetSystem(mbody1->GetSystem());

    this->mask.SetTwoBodiesVariables(&Body1->Variables(), &Body2->Variables());

    if (pos_are_relative) {
        this->frame1 = mpos1;
        this->frame2 = mpos2;
    } else {
        // from abs to body-rel
        static_cast<ChFrame<>*>(this->Body1)->TransformParentToLocal(mpos1, this->frame1);
        static_cast<ChFrame<>*>(this->Body2)->TransformParentToLocal(mpos2, this->frame2);
    }
}

void ChLinkMateGeneric::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                   std::shared_ptr<ChBodyFrame> mbody2,
                                   bool pos_are_relative,
                                   ChVector3d mpt1,
                                   ChVector3d mpt2,
                                   ChVector3d mnorm1,
                                   ChVector3d mnorm2) {
    assert(mbody1.get() != mbody2.get());

    this->Body1 = mbody1.get();
    this->Body2 = mbody2.get();
    // this->SetSystem(mbody1->GetSystem());

    this->mask.SetTwoBodiesVariables(&Body1->Variables(), &Body2->Variables());

    ChVector3d mx, my, mz, mN;
    ChMatrix33<> mrot;

    ChFrame<> mfr1;
    ChFrame<> mfr2;

    if (pos_are_relative) {
        mN = mnorm1;
        mN.DirToDxDyDz(mx, my, mz);
        mrot.SetFromDirectionAxes(mx, my, mz);
        mfr1.SetRot(mrot);
        mfr1.SetPos(mpt1);

        mN = mnorm2;
        mN.DirToDxDyDz(mx, my, mz);
        mrot.SetFromDirectionAxes(mx, my, mz);
        mfr2.SetRot(mrot);
        mfr2.SetPos(mpt2);
    } else {
        ChVector3d temp = VECT_Z;
        // from abs to body-rel
        mN = this->Body1->TransformDirectionParentToLocal(mnorm1);
        mN.DirToDxDyDz(mx, my, mz, temp);
        mrot.SetFromDirectionAxes(mx, my, mz);
        mfr1.SetRot(mrot);
        mfr1.SetPos(this->Body1->TransformPointParentToLocal(mpt1));

        mN = this->Body2->TransformDirectionParentToLocal(mnorm2);
        mN.DirToDxDyDz(mx, my, mz, temp);
        mrot.SetFromDirectionAxes(mx, my, mz);
        mfr2.SetRot(mrot);
        mfr2.SetPos(this->Body2->TransformPointParentToLocal(mpt2));
    }

    this->frame1 = mfr1;
    this->frame2 = mfr2;
}

void ChLinkMateGeneric::SetUseTangentStiffness(bool useKc) {

    if (useKc && this->Kmatr == nullptr) {

        this->Kmatr = chrono_types::make_unique<ChKblockGeneric>(&Body1->Variables(), &Body2->Variables());
        this->Kmatr->Get_K().resize(12, 12);
        this->Kmatr->Get_K().setZero();

    } else if (!useKc && this->Kmatr != nullptr) {

        this->Kmatr.reset();
    }
}

void ChLinkMateGeneric::InjectKRMmatrices(ChSystemDescriptor& descriptor) {
    if (!this->IsActive())
        return;

    if (this->Kmatr)
        descriptor.InsertKblock(Kmatr.get());
}

void ChLinkMateGeneric::KRMmatricesLoad(double Kfactor, double Rfactor, double Mfactor) {
    if (!this->IsActive())
        return;

    if (this->Kmatr) {
        ChMatrix33<> R_B1_W = Body1->GetRotMat();
        ChMatrix33<> R_B2_W = Body2->GetRotMat();
        // ChMatrix33<> R_F1_B1 = frame1.GetRotMat();
        // ChMatrix33<> R_F2_B2 = frame2.GetRotMat();
        ChFrame<> F1_W = this->frame1 >> (*this->Body1);
        ChFrame<> F2_W = this->frame2 >> (*this->Body2);
        ChMatrix33<> R_F1_W = F1_W.GetRotMat();
        ChMatrix33<> R_F2_W = F2_W.GetRotMat();
        ChVector3d r12_B2 = R_B2_W.transpose() * (F1_W.GetPos() - F2_W.GetPos());
        ChFrame<> F1_wrt_F2;
        F2_W.TransformParentToLocal(F1_W, F1_wrt_F2);

        ChVector3d r_F1_B1 = this->frame1.GetPos();
        ChVector3d r_F2_B2 = this->frame2.GetPos();
        ChStarMatrix33<> rtilde_F1_B1(r_F1_B1);
        ChStarMatrix33<> rtilde_F2_B2(r_F2_B2);

        // Precomupte utilities
        ChMatrix33<> R_F2_W_cross_gamma_f = R_F2_W * ChStarMatrix33<>(gamma_f);
        ChMatrix33<> R_W_F2 = R_F2_W.transpose();
        ChVector3d v_F1_F2 = F1_wrt_F2.GetRot().GetVector();
        ChMatrix33<> s_F1_F2_mat(F1_wrt_F2.GetRot().e0());
        ChMatrix33<> R_F2_W_times_G_times_R_W_F1 = R_F2_W * (-0.25 * TensorProduct(gamma_m, v_F1_F2)
                                                   - 0.25 * ChStarMatrix33<>(gamma_m) * (s_F1_F2_mat + ChStarMatrix33<>(v_F1_F2))) * R_F1_W.transpose();
        ChMatrix33<> R_F2_W_cross_gamma_f_times_R_B2_F2 = R_F2_W_cross_gamma_f * R_W_F2 * R_B2_W;

        // Populate Kc
        this->Kmatr->Get_K().block<3, 3>(0, 9) = -R_F2_W_cross_gamma_f * R_W_F2 * R_B2_W;

        this->Kmatr->Get_K().block<3, 3>(3, 3) = rtilde_F1_B1 * R_B1_W.transpose() * R_F2_W_cross_gamma_f * R_W_F2 * R_B1_W + R_B1_W.transpose() * R_F2_W * ChStarMatrix33<>(this->P * gamma_m) * R_W_F2 * R_B1_W
                                                 // stabilization part
                                                 +R_B1_W.transpose() * R_F2_W_times_G_times_R_W_F1 * R_B1_W;

        this->Kmatr->Get_K().block<3, 3>(3, 9) = -rtilde_F1_B1 * R_B1_W.transpose() * R_F2_W_cross_gamma_f * R_W_F2 * R_B2_W - R_B1_W.transpose() * R_F2_W * ChStarMatrix33<>(this->P * gamma_m) * R_W_F2 * R_B2_W 
                                                 // stabilization part
                                                 -R_B1_W.transpose() * R_F2_W_times_G_times_R_W_F1 * R_B2_W;

        this->Kmatr->Get_K().block<3, 3>(6, 9) = R_F2_W_cross_gamma_f * R_W_F2 * R_B2_W;

        this->Kmatr->Get_K().block<3, 3>(9, 0) = R_F2_W_cross_gamma_f_times_R_B2_F2;

        this->Kmatr->Get_K().block<3, 3>(9, 3) = -R_F2_W_cross_gamma_f_times_R_B2_F2 * R_B1_W * rtilde_F1_B1 
                                                 // stabilization part
                                                 -R_B2_W.transpose() * R_F2_W_times_G_times_R_W_F1 * R_B1_W;
        
        this->Kmatr->Get_K().block<3, 3>(9, 6) = -R_F2_W_cross_gamma_f_times_R_B2_F2;

        this->Kmatr->Get_K().block<3, 3>(9, 9) = R_F2_W_cross_gamma_f_times_R_B2_F2 * R_B2_W * ChStarMatrix33<>(r12_B2 + r_F2_B2)
                                                 // stabilization part
                                                 + R_B2_W.transpose() * R_F2_W_times_G_times_R_W_F1 * R_B2_W;

        // The complete tangent stiffness matrix
        this->Kmatr->Get_K() *= Kfactor;
    }
}

//// STATE BOOKKEEPING FUNCTIONS

void ChLinkMateGeneric::IntStateGatherReactions(const unsigned int off_L, ChVectorDynamic<>& L) {
    if (!this->IsActive())
        return;

    int nc = 0;
    if (c_x) {
        if (mask.Constr_N(nc).IsActive())
            L(off_L + nc) = -gamma_f.x();
        nc++;
    }
    if (c_y) {
        if (mask.Constr_N(nc).IsActive())
            L(off_L + nc) = -gamma_f.y();
        nc++;
    }
    if (c_z) {
        if (mask.Constr_N(nc).IsActive())
            L(off_L + nc) = -gamma_f.z();
        nc++;
    }
    if (c_rx) {
        if (mask.Constr_N(nc).IsActive())
            L(off_L + nc) = -gamma_m.x();
        nc++;
    }
    if (c_ry) {
        if (mask.Constr_N(nc).IsActive())
            L(off_L + nc) = -gamma_m.y();
        nc++;
    }
    if (c_rz) {
        if (mask.Constr_N(nc).IsActive())
            L(off_L + nc) = -gamma_m.z();
        nc++;
    }
}

void ChLinkMateGeneric::IntStateScatterReactions(const unsigned int off_L, const ChVectorDynamic<>& L) {
    react_force = VNULL;
    react_torque = VNULL;
    gamma_f = VNULL;
    gamma_m = VNULL;

    if (!this->IsActive())
        return;

    int nc = 0;
    if (c_x) {
        if (mask.Constr_N(nc).IsActive())
            gamma_f.x() = -L(off_L + nc);
        nc++;
    }
    if (c_y) {
        if (mask.Constr_N(nc).IsActive())
            gamma_f.y() = -L(off_L + nc);
        nc++;
    }
    if (c_z) {
        if (mask.Constr_N(nc).IsActive())
            gamma_f.z() = -L(off_L + nc);
        nc++;
    }
    if (c_rx) {
        if (mask.Constr_N(nc).IsActive())
            gamma_m.x() = -L(off_L + nc);
        nc++;
    }
    if (c_ry) {
        if (mask.Constr_N(nc).IsActive())
            gamma_m.y() = -L(off_L + nc);
        nc++;
    }
    if (c_rz) {
        if (mask.Constr_N(nc).IsActive())
            gamma_m.z() = -L(off_L + nc);
        nc++;
    }

    // The transpose of the Jacobian matrix of constraint is:
    // Cq.T = [ Jx1.T;  O      ]
    //        [ Jr1.T;  Jw1.T  ]
    //        [ Jx2.T;  O      ]
    //        [ Jr2.T;  Jw2.T  ]
    // The Lagrange multipliers are:
    // gamma = [ gamma_f ]
    //         [ gamma_m ]
    //
    // The forces and torques acting on Body1 and Body2 are simply calculated as: Cq.T*gamma,
    // where the forces are expressed in the absolute frame,
    // the torques are expressed in the local frame of B1,B2, respectively.
    // This is consistent with the mixed basis of rigid bodies and nodes.
    react_force = gamma_f;
    ChFrame<> F1_W = this->frame1 >> (*this->Body1);
    ChFrame<> F2_W = this->frame2 >> (*this->Body2);
    ChVector3d r12_F2 = F2_W.GetRotMat().transpose() * (F1_W.GetPos() - F2_W.GetPos());
    react_torque = ChStarMatrix33<>(r12_F2) * gamma_f + this->P * gamma_m;
}

void ChLinkMateGeneric::IntLoadResidual_CqL(const unsigned int off_L,
                                            ChVectorDynamic<>& R,
                                            const ChVectorDynamic<>& L,
                                            const double c) {
    int cnt = 0;
    for (int i = 0; i < mask.nconstr; i++) {
        if (mask.Constr_N(i).IsActive()) {
            mask.Constr_N(i).MultiplyTandAdd(R, L(off_L + cnt) * c);
            cnt++;
        }
    }
}

void ChLinkMateGeneric::IntLoadConstraint_C(const unsigned int off_L,
                                            ChVectorDynamic<>& Qc,
                                            const double c,
                                            bool do_clamp,
                                            double recovery_clamp) {
    int cnt = 0;
    for (int i = 0; i < mask.nconstr; i++) {
        if (mask.Constr_N(i).IsActive()) {
            if (do_clamp) {
                if (mask.Constr_N(i).IsUnilateral())
                    Qc(off_L + cnt) += std::max(c * C(cnt), -recovery_clamp);
                else
                    Qc(off_L + cnt) += std::min(std::max(c * C(cnt), -recovery_clamp), recovery_clamp);
            } else
                Qc(off_L + cnt) += c * C(cnt);
            cnt++;
        }
    }
}

void ChLinkMateGeneric::IntLoadConstraint_Ct(const unsigned int off_L, ChVectorDynamic<>& Qc, const double c) {
    // NOT NEEDED BECAUSE NO RHEONOMIC TERM
}

void ChLinkMateGeneric::IntToDescriptor(const unsigned int off_v,
                                        const ChStateDelta& v,
                                        const ChVectorDynamic<>& R,
                                        const unsigned int off_L,
                                        const ChVectorDynamic<>& L,
                                        const ChVectorDynamic<>& Qc) {
    int cnt = 0;
    for (int i = 0; i < mask.nconstr; i++) {
        if (mask.Constr_N(i).IsActive()) {
            mask.Constr_N(i).Set_l_i(L(off_L + cnt));
            mask.Constr_N(i).Set_b_i(Qc(off_L + cnt));
            cnt++;
        }
    }
}

void ChLinkMateGeneric::IntFromDescriptor(const unsigned int off_v,
                                          ChStateDelta& v,
                                          const unsigned int off_L,
                                          ChVectorDynamic<>& L) {
    int cnt = 0;
    for (int i = 0; i < mask.nconstr; i++) {
        if (mask.Constr_N(i).IsActive()) {
            L(off_L + cnt) = mask.Constr_N(i).Get_l_i();
            cnt++;
        }
    }
}

// SOLVER INTERFACES

void ChLinkMateGeneric::InjectConstraints(ChSystemDescriptor& mdescriptor) {
    if (!this->IsActive())
        return;

    for (int i = 0; i < mask.nconstr; i++) {
        if (mask.Constr_N(i).IsActive())
            mdescriptor.InsertConstraint(&mask.Constr_N(i));
    }
}

void ChLinkMateGeneric::ConstraintsBiReset() {
    if (!this->IsActive())
        return;

    for (int i = 0; i < mask.nconstr; i++) {
        mask.Constr_N(i).Set_b_i(0.);
    }
}

void ChLinkMateGeneric::ConstraintsBiLoad_C(double factor, double recovery_clamp, bool do_clamp) {
    if (!this->IsActive())
        return;

    //***TEST***
    /*
        std::cout << "cload: " ;
        if (this->c_x) std::cout << " x";
        if (this->c_y) std::cout << " y";
        if (this->c_z) std::cout << " z";
        if (this->c_rx) std::cout << " Rx";
        if (this->c_ry) std::cout << " Ry";
        if (this->c_rz) std::cout << " Rz";
        std::cout << *this->C << std::endl;
    */
    int cnt = 0;
    for (int i = 0; i < mask.nconstr; i++) {
        if (mask.Constr_N(i).IsActive()) {
            if (do_clamp) {
                if (mask.Constr_N(i).IsUnilateral())
                    mask.Constr_N(i).Set_b_i(mask.Constr_N(i).Get_b_i() +
                                              std::max(factor * C(cnt), -recovery_clamp));
                else
                    mask.Constr_N(i).Set_b_i(mask.Constr_N(i).Get_b_i() +
                                              std::min(std::max(factor * C(cnt), -recovery_clamp), recovery_clamp));
            } else
                mask.Constr_N(i).Set_b_i(mask.Constr_N(i).Get_b_i() + factor * C(cnt));

            cnt++;
        }
    }
}

void ChLinkMateGeneric::ConstraintsBiLoad_Ct(double factor) {
    if (!this->IsActive())
        return;

    // NOT NEEDED BECAUSE NO RHEONOMIC TERM
}

void ChLinkMateGeneric::ConstraintsLoadJacobians() {
    // already loaded when doing Update (which used the matrices of the scalar constraint objects)
}

void ChLinkMateGeneric::ConstraintsFetch_react(double factor) {
    react_force = VNULL;
    react_torque = VNULL;
    gamma_f = VNULL;
    gamma_m = VNULL;

    if (!this->IsActive())
        return;

    int nc = 0;
    if (c_x) {
        if (mask.Constr_N(nc).IsActive())
            gamma_f.x() = -mask.Constr_N(nc).Get_l_i() * factor;
        nc++;
    }
    if (c_y) {
        if (mask.Constr_N(nc).IsActive())
            gamma_f.y() = -mask.Constr_N(nc).Get_l_i() * factor;
        nc++;
    }
    if (c_z) {
        if (mask.Constr_N(nc).IsActive())
            gamma_f.z() = -mask.Constr_N(nc).Get_l_i() * factor;
        nc++;
    }
    if (c_rx) {
        if (mask.Constr_N(nc).IsActive())
            gamma_m.x() = -mask.Constr_N(nc).Get_l_i() * factor;
        nc++;
    }
    if (c_ry) {
        if (mask.Constr_N(nc).IsActive())
            gamma_m.y() = -mask.Constr_N(nc).Get_l_i() * factor;
        nc++;
    }
    if (c_rz) {
        if (mask.Constr_N(nc).IsActive())
            gamma_m.z() = -mask.Constr_N(nc).Get_l_i() * factor;
        nc++;
    }

    react_force = gamma_f;

    ChFrame<> F1_W = this->frame1 >> (*this->Body1);
    ChFrame<> F2_W = this->frame2 >> (*this->Body2);
    ChVector3d r12_F2 = F2_W.GetRotMat().transpose() * (F1_W.GetPos() - F2_W.GetPos());
    react_torque = ChStarMatrix33<>(r12_F2) * gamma_f + this->P * gamma_m;
}

void ChLinkMateGeneric::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMateGeneric>();

    // serialize parent class
    ChLinkMate::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(frame1);
    archive_out << CHNVP(frame2);
    archive_out << CHNVP(c_x);
    archive_out << CHNVP(c_y);
    archive_out << CHNVP(c_z);
    archive_out << CHNVP(c_rx);
    archive_out << CHNVP(c_ry);
    archive_out << CHNVP(c_rz);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMateGeneric::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMateGeneric>();

    // deserialize parent class
    ChLinkMate::ArchiveIn(archive_in);

    // deserialize all member data:
    archive_in >> CHNVP(frame1);
    archive_in >> CHNVP(frame2);

    bool c_x_success = archive_in.in(CHNVP(c_x));
    bool c_y_success = archive_in.in(CHNVP(c_y));
    bool c_z_success = archive_in.in(CHNVP(c_z));
    bool c_rx_success = archive_in.in(CHNVP(c_rx));
    bool c_ry_success = archive_in.in(CHNVP(c_ry));
    bool c_rz_success = archive_in.in(CHNVP(c_rz));
    if (c_x_success && c_y_success && c_z_success && c_rx_success && c_ry_success && c_rz_success)
        this->SetConstrainedCoords(c_x, c_y, c_z, c_rx, c_ry, c_rz);  // takes care of mask

    
}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMatePlane)

ChLinkMatePlane::ChLinkMatePlane(const ChLinkMatePlane& other) : ChLinkMateGeneric(other) {
    flipped = other.flipped;
    separation = other.separation;
}

void ChLinkMatePlane::SetFlipped(bool doflip) {
    if (doflip != this->flipped) {
        // swaps direction of X axis by flipping 180 deg the frame A (slave)

        ChFrame<> frameRotator(VNULL, QuatFromAngleY(CH_C_PI));
        this->frame1.ConcatenatePostTransformation(frameRotator);

        this->flipped = doflip;
    }
}

void ChLinkMatePlane::Initialize(std::shared_ptr<ChBodyFrame> mbody1, 
                                 std::shared_ptr<ChBodyFrame> mbody2,
                                 bool pos_are_relative, 
                                 ChVector3d mpt1,
                                 ChVector3d mpt2,
                                 ChVector3d mnorm1,
                                 ChVector3d mnorm2) {
    // set the two frames so that they have the X axis aligned when the
    // two normals are opposed (default behavior, otherwise is considered 'flipped')

    ChVector3d mnorm1_reversed;
    if (!flipped)
        mnorm1_reversed = -mnorm1;
    else
        mnorm1_reversed = mnorm1;

    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, mnorm1_reversed, mnorm2);
}

/// Override _all_ time, jacobian etc. updating.
void ChLinkMatePlane::Update(double mtime, bool update_assets) {
    // Parent class inherit
    ChLinkMateGeneric::Update(mtime, update_assets);

    // .. then add the effect of imposed distance on C residual vector
    C(0) -= separation;  // for this mate, C = {Cx, Cry, Crz}
}

void ChLinkMatePlane::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMatePlane>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(flipped);
    archive_out << CHNVP(separation);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMatePlane::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMatePlane>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    // deserialize all member data:
    archive_in >> CHNVP(separation);
    archive_in >> CHNVP(flipped);

}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateCoaxial)

ChLinkMateCoaxial::ChLinkMateCoaxial(const ChLinkMateCoaxial& other) : ChLinkMateGeneric(other) {
    flipped = other.flipped;
}

void ChLinkMateCoaxial::SetFlipped(bool doflip) {
    if (doflip != flipped) {
        // swaps direction of X axis by flipping 180 deg the frame A (slave)

        ChFrame<> frameRotator(VNULL, QuatFromAngleY(CH_C_PI));
        this->frame1.ConcatenatePostTransformation(frameRotator);

        flipped = doflip;
    }
}

void ChLinkMateCoaxial::Initialize(std::shared_ptr<ChBodyFrame> mbody1, 
                                   std::shared_ptr<ChBodyFrame> mbody2,
                                   bool pos_are_relative,
                                   ChVector3d mpt1,
                                   ChVector3d mpt2,
                                   ChVector3d mnorm1,
                                   ChVector3d mnorm2) {
    // set the two frames so that they have the X axis aligned when the
    // two normals are opposed (default behavior, otherwise is considered 'flipped')

    ChVector3d mnorm1_reversed;
    if (!flipped)
        mnorm1_reversed = mnorm1;
    else
        mnorm1_reversed = -mnorm1;

    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, mnorm1_reversed, mnorm2);
}

void ChLinkMateCoaxial::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMateCoaxial>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(flipped);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMateCoaxial::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMateCoaxial>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    archive_in >> CHNVP(flipped);
}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateRevolute)

ChLinkMateRevolute::ChLinkMateRevolute(const ChLinkMateRevolute& other) : ChLinkMateGeneric(other) {
    flipped = other.flipped;
}

void ChLinkMateRevolute::SetFlipped(bool doflip) {
    if (doflip != flipped) {
        // swaps direction of Z axis by flipping 180 deg the frame A (slave)

        ChFrame<> frameRotator(VNULL, QuatFromAngleY(CH_C_PI));
        this->frame1.ConcatenatePostTransformation(frameRotator);

        flipped = doflip;
    }
}

void ChLinkMateRevolute::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
    std::shared_ptr<ChBodyFrame> mbody2,
    bool pos_are_relative,
    ChVector3d mpt1,
    ChVector3d mpt2,
    ChVector3d mdir1,
    ChVector3d mdir2) {
    // set the two frames so that they have the Z axis aligned when the
    // two normals are opposed (default behavior, otherwise is considered 'flipped')

    ChVector3d mdir1_reversed;
    if (!flipped)
        mdir1_reversed = mdir1;
    else
        mdir1_reversed = -mdir1;

    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, mdir1_reversed, mdir2);
}

double ChLinkMateRevolute::GetRelativeAngle() {
    ChFrame<> F1_W = frame1 >> *Body1;
    ChFrame<> F2_W = frame2 >> *Body2;
    ChFrame<> F1_F2;
    F2_W.TransformParentToLocal(F1_W, F1_F2);
    double angle = std::remainder(F1_F2.GetRot().GetRotVec().z(), CH_C_2PI); // NB: assumes rotation along z
    return angle;
}

double ChLinkMateRevolute::GetRelativeAngle_dt() {
    ChFrameMoving<> F1_W = ChFrameMoving<>(frame1) >> *Body1;
    ChFrameMoving<> F2_W = ChFrameMoving<>(frame2) >> *Body2;
    ChFrameMoving<> F1_F2;
    F2_W.TransformParentToLocal(F1_W, F1_F2);
    double vel12_W = F1_F2.GetWvel_loc().z(); // NB: assumes rotation along z
    return vel12_W;
}

double ChLinkMateRevolute::GetRelativeAngle_dtdt() {
    ChFrameMoving<> F1_W = ChFrameMoving<>(frame1) >> *Body1;
    ChFrameMoving<> F2_W = ChFrameMoving<>(frame2) >> *Body2;
    ChFrameMoving<> F1_F2;
    F2_W.TransformParentToLocal(F1_W, F1_F2);
    double acc12_W = F1_F2.GetWacc_loc().z(); // NB: assumes rotation along z
    return acc12_W;
}

void ChLinkMateRevolute::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMateRevolute>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(flipped);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMateRevolute::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/archive_in.VersionRead<ChLinkMateRevolute>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    archive_in >> CHNVP(flipped);

}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMatePrismatic)

ChLinkMatePrismatic::ChLinkMatePrismatic(const ChLinkMatePrismatic& other) : ChLinkMateGeneric(other) {
    flipped = other.flipped;
}

void ChLinkMatePrismatic::SetFlipped(bool doflip) {
    if (doflip != flipped) {
        // swaps direction of X axis by flipping 180 deg the frame A (slave)

        ChFrame<> frameRotator(VNULL, QuatFromAngleY(CH_C_PI));
        this->frame1.ConcatenatePostTransformation(frameRotator);

        flipped = doflip;
    }
}

void ChLinkMatePrismatic::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                     std::shared_ptr<ChBodyFrame> mbody2,
                                     bool pos_are_relative,
                                     ChVector3d mpt1,
                                     ChVector3d mpt2,
                                     ChVector3d mdir1,
                                     ChVector3d mdir2) {
    // set the two frames so that they have the X axis aligned when the
    // two normals are opposed (default behavior, otherwise is considered 'flipped')

    ChVector3d mdir1_reversed;
    if (!flipped)
        mdir1_reversed = mdir1;
    else
        mdir1_reversed = -mdir1;

    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, mdir1_reversed, mdir2);
}

double ChLinkMatePrismatic::GetRelativePos() {
    ChFrame<> F1_W = frame1 >> *Body1;
    ChFrame<> F2_W = frame2 >> *Body2;
    ChVector3d pos12_W = F1_W.GetPos() - F2_W.GetPos();
    return pos12_W.x(); // NB: assumes translation along x
}

double ChLinkMatePrismatic::GetRelativePos_dt() {
    ChFrameMoving<> F1_W = ChFrameMoving<>(frame1) >> *Body1;
    ChFrameMoving<> F2_W = ChFrameMoving<>(frame2) >> *Body2;
    ChVector3d vel12_W = F1_W.GetPos_dt() - F2_W.GetPos_dt();
    return vel12_W.x(); // NB: assumes translation along x
}

double ChLinkMatePrismatic::GetRelativePos_dtdt() {
    ChFrameMoving<> F1_W = ChFrameMoving<>(frame1) >> *Body1;
    ChFrameMoving<> F2_W = ChFrameMoving<>(frame2) >> *Body2;
    ChVector3d acc12_W = F1_W.GetPos_dtdt() - F2_W.GetPos_dtdt();
    return acc12_W.x(); // NB: assumes translation along x
}

void ChLinkMatePrismatic::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMatePrismatic>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(flipped);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMatePrismatic::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/archive_in.VersionRead<ChLinkMatePrismatic>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    archive_in >> CHNVP(flipped);
}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateSpherical)

ChLinkMateSpherical::ChLinkMateSpherical(const ChLinkMateSpherical& other) : ChLinkMateGeneric(other) {}

void ChLinkMateSpherical::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                     std::shared_ptr<ChBodyFrame> mbody2,
                                     bool pos_are_relative,
                                     ChVector3d mpt1,
                                     ChVector3d mpt2) {
    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, VECT_X, VECT_X);
}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateXdistance)

ChLinkMateXdistance::ChLinkMateXdistance(const ChLinkMateXdistance& other) : ChLinkMateGeneric(other) {
    distance = other.distance;
}

void ChLinkMateXdistance::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                     std::shared_ptr<ChBodyFrame> mbody2,
                                     bool pos_are_relative,
                                     ChVector3d mpt1,
                                     ChVector3d mpt2,
                                     ChVector3d mdir2) {
    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, mdir2, mdir2);
}

void ChLinkMateXdistance::Update(double mtime, bool update_assets) {
    // Parent class inherit
    ChLinkMateGeneric::Update(mtime, update_assets);

    // .. then add the effect of imposed distance on C residual vector
    C(0) -= distance;  // for this mate, C = {Cx}
}

void ChLinkMateXdistance::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMateXdistance>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(distance);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMateXdistance::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMateXdistance>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    // deserialize all member data:
    archive_in >> CHNVP(distance);

}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateParallel)

ChLinkMateParallel::ChLinkMateParallel(const ChLinkMateParallel& other) : ChLinkMateGeneric(other) {
    flipped = other.flipped;
}

void ChLinkMateParallel::SetFlipped(bool doflip) {
    if (doflip != flipped) {
        // swaps direction of X axis by flipping 180 deg the frame A (slave)

        ChFrame<> frameRotator(VNULL, QuatFromAngleY(CH_C_PI));
        this->frame1.ConcatenatePostTransformation(frameRotator);

        flipped = doflip;
    }
}

void ChLinkMateParallel::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                    std::shared_ptr<ChBodyFrame> mbody2,
                                    bool pos_are_relative,
                                    ChVector3d mpt1,
                                    ChVector3d mpt2,
                                    ChVector3d mnorm1,
                                    ChVector3d mnorm2) {
    // set the two frames so that they have the X axis aligned when the
    // two axes are aligned (default behavior, otherwise is considered 'flipped')

    ChVector3d mnorm1_reversed;
    if (!flipped)
        mnorm1_reversed = mnorm1;
    else
        mnorm1_reversed = -mnorm1;

    ChLinkMateGeneric::Initialize(mbody1, mbody2, pos_are_relative, mpt1, mpt2, mnorm1_reversed, mnorm2);
}

void ChLinkMateParallel::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMateParallel>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(flipped);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMateParallel::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMateParallel>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    // deserialize all member data:
    archive_in >> CHNVP(flipped);
}

// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateOrthogonal)

ChLinkMateOrthogonal::ChLinkMateOrthogonal(const ChLinkMateOrthogonal& other) : ChLinkMateGeneric(other) {
    reldir1 = other.reldir1;
    reldir2 = other.reldir2;
}

void ChLinkMateOrthogonal::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                      std::shared_ptr<ChBodyFrame> mbody2,
                                      bool pos_are_relative,
                                      ChVector3d mpt1,
                                      ChVector3d mpt2,
                                      ChVector3d mnorm1,
                                      ChVector3d mnorm2) {
    // set the two frames so that they have the X axis aligned
    ChVector3d mabsnorm1, mabsnorm2;
    if (pos_are_relative) {
        reldir1 = mnorm1;
        reldir2 = mnorm2;
    } else {
        reldir1 = mbody1->TransformDirectionParentToLocal(mnorm1);
        reldir2 = mbody2->TransformDirectionParentToLocal(mnorm2);
    }

    // do this asap otherwise the following Update() won't work..
    this->Body1 = mbody1.get();
    this->Body2 = mbody2.get();

    // Force the alignment of frames so that the X axis is cross product of two dirs, etc.
    // by calling the custom update function of ChLinkMateOrthogonal.
    this->Update(this->ChTime);

    // Perform initialization (set pointers to variables, etc.)
    ChLinkMateGeneric::Initialize(mbody1, mbody2,
                                  true,  // recycle already-updated frames
                                  this->frame1, this->frame2);
}

/// Override _all_ time, jacobian etc. updating.
void ChLinkMateOrthogonal::Update(double mtime, bool update_assets) {
    // Prepare the alignment of the two frames so that the X axis is orthogonal
    // to the two directions

    ChVector3d mabsD1, mabsD2;

    if (this->Body1 && this->Body2) {
        mabsD1 = this->Body1->TransformDirectionLocalToParent(reldir1);
        mabsD2 = this->Body2->TransformDirectionLocalToParent(reldir2);

        ChVector3d mX = Vcross(mabsD2, mabsD1);
        double xlen = mX.Length();

        // Ops.. parallel directions? -> fallback to singularity handling
        if (fabs(xlen) < 1e-20) {
            ChVector3d ortho_gen;
            if (fabs(mabsD1.z()) < 0.9)
                ortho_gen = VECT_Z;
            if (fabs(mabsD1.y()) < 0.9)
                ortho_gen = VECT_Y;
            if (fabs(mabsD1.x()) < 0.9)
                ortho_gen = VECT_X;
            mX = Vcross(mabsD1, ortho_gen);
            xlen = Vlength(mX);
        }
        mX.Scale(1.0 / xlen);

        ChVector3d mY1, mZ1;
        mZ1 = mabsD1;
        mY1 = Vcross(mZ1, mX);

        ChMatrix33<> mA1;
        mA1.SetFromDirectionAxes(mX, mY1, mZ1);

        ChVector3d mY2, mZ2;
        mY2 = mabsD2;
        mZ2 = Vcross(mX, mY2);

        ChMatrix33<> mA2;
        mA2.SetFromDirectionAxes(mX, mY2, mZ2);

        ChFrame<> absframe1(VNULL, mA1);  // position not needed for orth. constr. computation
        ChFrame<> absframe2(VNULL, mA2);  // position not needed for orth. constr. computation

        // from abs to body-rel
        static_cast<ChFrame<>*>(this->Body1)->TransformParentToLocal(absframe1, this->frame1);
        static_cast<ChFrame<>*>(this->Body2)->TransformParentToLocal(absframe2, this->frame2);
    }

    // Parent class inherit
    ChLinkMateGeneric::Update(mtime, update_assets);
}

void ChLinkMateOrthogonal::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChLinkMateOrthogonal>();

    // serialize parent class
    ChLinkMateGeneric::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(reldir1);
    archive_out << CHNVP(reldir2);
}

/// Method to allow de serialization of transient data from archives.
void ChLinkMateOrthogonal::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/ archive_in.VersionRead<ChLinkMateOrthogonal>();

    // deserialize parent class
    ChLinkMateGeneric::ArchiveIn(archive_in);

    // deserialize all member data:
    archive_in >> CHNVP(reldir1);
    archive_in >> CHNVP(reldir2);
}



// -----------------------------------------------------------------------------

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChLinkMateFix)

ChLinkMateFix::ChLinkMateFix(const ChLinkMateFix& other) : ChLinkMateGeneric(other) {}

void ChLinkMateFix::Initialize(std::shared_ptr<ChBodyFrame> mbody1,
                                     std::shared_ptr<ChBodyFrame> mbody2)
{
    ChLinkMateGeneric::Initialize(
        mbody1, mbody2, 
        false,               // constraint reference frame in abs coords
        ChFrame<>(*mbody1),  // defaults to have constraint reference frame as mbody1 coordsystem
        ChFrame<>(*mbody1)   // defaults to have constraint reference frame as mbody1 coordsystem
        ); 
}




}  // end namespace chrono
