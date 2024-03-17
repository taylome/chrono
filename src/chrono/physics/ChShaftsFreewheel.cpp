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
// Authors: Alessandro Tasora
// =============================================================================

#include "chrono/physics/ChShaft.h"
#include "chrono/physics/ChShaftsFreewheel.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {

// Register into the object factory, to enable run-time dynamic creation and persistence
CH_FACTORY_REGISTER(ChShaftsFreewheel)

ChShaftsFreewheel::ChShaftsFreewheel()
    : step(CH_C_2PI / 24.0),
      torque_react(0),
      jamming_mode(false),
      phase(0),
      violation(0),
      alpha_max(0),
      free_forward(true) {}

ChShaftsFreewheel::ChShaftsFreewheel(const ChShaftsFreewheel& other) : ChShaftsCouple(other), violation(0) {
    step = other.step;
    torque_react = other.torque_react;
    jamming_mode = other.jamming_mode;
    phase = other.phase;
    alpha_max = other.alpha_max;
    free_forward = other.free_forward;
}

bool ChShaftsFreewheel::Initialize(std::shared_ptr<ChShaft> shaft_1, std::shared_ptr<ChShaft> shaft_2) {
    // Parent initialization
    if (!ChShaftsCouple::Initialize(shaft_1, shaft_2))
        return false;

    phase = 0;

    constraint.SetVariables(&shaft_1->Variables(), &shaft_2->Variables());
    constraint.SetBoxedMinMax(0, 1e10);

    SetSystem(shaft1->GetSystem());
    return true;
}

void ChShaftsFreewheel::Update(double mytime, bool update_assets) {
    // Inherit time changes of parent class
    ChShaftsCouple::Update(mytime, update_assets);

    // update class data
    violation = 0;

    double relrot = this->shaft1->GetPos() - this->shaft2->GetPos();
    if (this->free_forward)
        alpha_max = std::max(relrot, alpha_max);
    else
        alpha_max = std::min(relrot, alpha_max);
}

void ChShaftsFreewheel::IntStateGatherReactions(const unsigned int off_L, ChVectorDynamic<>& L) {
    L(off_L) = torque_react;
}

void ChShaftsFreewheel::IntStateScatterReactions(const unsigned int off_L, const ChVectorDynamic<>& L) {
    torque_react = L(off_L);
}

void ChShaftsFreewheel::IntLoadResidual_CqL(const unsigned int off_L,    // offset in L multipliers
                                            ChVectorDynamic<>& R,        // result: the R residual, R += c*Cq'*L
                                            const ChVectorDynamic<>& L,  // the L vector
                                            const double c               // a scaling factor
) {
    constraint.MultiplyTandAdd(R, L(off_L) * c);
}

void ChShaftsFreewheel::IntLoadConstraint_C(const unsigned int off_L,  // offset in Qc residual
                                            ChVectorDynamic<>& Qc,     // result: the Qc residual, Qc += c*C
                                            const double c,            // a scaling factor
                                            bool do_clamp,             // apply clamping to c*C?
                                            double recovery_clamp      // value for min/max clamping of c*C
) {
    double relrot = this->shaft1->GetPos() - this->shaft2->GetPos();

    double res;

    if (this->free_forward)
        res = relrot - (this->phase + this->step * floor((alpha_max - this->phase) / this->step));
    else
        res = -(-relrot - (-this->phase + this->step * floor((-alpha_max + this->phase) / this->step)));

    if (jamming_mode)
        res = 0;

    double cnstr_violation = c * res;

    if (do_clamp) {
        if (this->free_forward)
            cnstr_violation = std::max(cnstr_violation, -recovery_clamp);
        else
            cnstr_violation = std::min(cnstr_violation, recovery_clamp);
    }

    Qc(off_L) += cnstr_violation;
}

void ChShaftsFreewheel::IntToDescriptor(const unsigned int off_v,  // offset in v, R
                                        const ChStateDelta& v,
                                        const ChVectorDynamic<>& R,
                                        const unsigned int off_L,  // offset in L, Qc
                                        const ChVectorDynamic<>& L,
                                        const ChVectorDynamic<>& Qc) {
    constraint.Set_l_i(L(off_L));

    constraint.Set_b_i(Qc(off_L));
}

void ChShaftsFreewheel::IntFromDescriptor(const unsigned int off_v,  // offset in v
                                          ChStateDelta& v,
                                          const unsigned int off_L,  // offset in L
                                          ChVectorDynamic<>& L) {
    L(off_L) = constraint.Get_l_i();
}

void ChShaftsFreewheel::InjectConstraints(ChSystemDescriptor& mdescriptor) {
    // if (!IsActive())
    //	return;

    mdescriptor.InsertConstraint(&constraint);
}

void ChShaftsFreewheel::ConstraintsBiReset() {
    constraint.Set_b_i(0.);
}

void ChShaftsFreewheel::ConstraintsBiLoad_C(double factor, double recovery_clamp, bool do_clamp) {
    // if (!IsActive())
    //	return;

    double res = 0;  // no residual

    constraint.Set_b_i(constraint.Get_b_i() + factor * res);
}

void ChShaftsFreewheel::ConstraintsLoadJacobians() {
    // compute jacobians
    constraint.Get_Cq_a()(0) = 1.0;
    constraint.Get_Cq_b()(0) = -1.0;
}

void ChShaftsFreewheel::ConstraintsFetch_react(double factor) {
    // From constraints to react vector:
    torque_react = constraint.Get_l_i() * factor;
}

void ChShaftsFreewheel::ArchiveOut(ChArchiveOut& archive_out) {
    // version number
    archive_out.VersionWrite<ChShaftsFreewheel>();

    // serialize parent class
    ChShaftsCouple::ArchiveOut(archive_out);

    // serialize all member data:
    archive_out << CHNVP(step);
    archive_out << CHNVP(jamming_mode);
    archive_out << CHNVP(phase);
    archive_out << CHNVP(alpha_max);
    archive_out << CHNVP(free_forward);
}

void ChShaftsFreewheel::ArchiveIn(ChArchiveIn& archive_in) {
    // version number
    /*int version =*/archive_in.VersionRead<ChShaftsFreewheel>();

    // deserialize parent class:
    ChShaftsCouple::ArchiveIn(archive_in);

    // deserialize all member data:
    archive_in >> CHNVP(step);
    archive_in >> CHNVP(jamming_mode);
    archive_in >> CHNVP(phase);
    archive_in >> CHNVP(alpha_max);
    archive_in >> CHNVP(free_forward);
    constraint.SetVariables(&shaft1->Variables(), &shaft2->Variables());
}

}  // end namespace chrono
