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
// Combined Bending Efficiency test for the 3 node ANCF beam element.
//
// The test sweeps the mesh density at a single beam angle and tip load to:
//    Calculate the static displacement of the endpoint and midpoint of the beam
//    via DoStaticNonlinear()
//    - This is used to generate the raw data for a % error vs. # of elements 
//      graph where the error is with respect to a reference nonlinear static 
//      solution using a highly refined mesh in <Commerical FEA Package>
//    Calculate the static displacement of the endpoint and midpoint of the beam
//    through a dynamic test with element damping.
//    - This is used to generate the raw data for a % error vs. # of elements 
//      graph based on the final position of the beam with respect to a reference 
//      nonlinear static solution using a highly refined mesh in <Commerical FEA Package>
//    - This is also used to generate the raw data for a %error vs. wall time
//      graph and a wall time vs. # of elements graph
//    Calculates the dynamic accuracy of the displacement of the midpoint of the
//    beam under a suddenly applied tip load with no applied damping
//    - This is used to generate the raw data for a % error vs. # of elements 
//      graph based on the period of a sine wave fit to the dynamic data for 
//      this simulation vs. that from a reference nonlinear dynamic solution
//      using a highly refined mesh in <Commerical FEA Package>
//    - This is also used to generate the raw data for a %error vs. wall time
//      graph and a wall time vs. # of elements graph
//
// The model parameters are based on the experimental setups in: 
// Dowell, E.H., Traybar, J.J.: An experimental study of the
// nonlinear stiffness of a rotor blade undergoing flap, lag and
// twist deformations.Technical Report 1194, AMS Report,
// January(1975)
// https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770003066.pdf
// Dowell, E.H., Traybar, J.J.: An addendum to AMS report no
// 1194 entitled an experimental study of the nonlinear stiffness
// of a rotor blade undergoing flap, lag, and twist deformations.
// Technical Report 1257, AMS Report, December(1975)
// https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770003067.pdf.
//
// =============================================================================

#include "chrono/ChConfig.h"
#include "chrono/utils/ChBenchmark.h"

#include "chrono/physics/ChSystemSMC.h"

#include "chrono/fea/ChElementBeamANCF.h"
#include "chrono/fea/ChMesh.h"
#include "chrono/fea/ChVisualizationFEAmesh.h"

#include "chrono_thirdparty/filesystem/path.h"

#ifdef CHRONO_IRRLICHT
#include "chrono_irrlicht/ChIrrApp.h"
#endif

#ifdef CHRONO_MUMPS
#include "chrono_mumps/ChSolverMumps.h"
#endif

using namespace chrono;
using namespace chrono::fea;

enum class SolverType {MUMPS, SparseLU};


class TestBeam{
  public:
    TestBeam(int num_elements, double beam_angle_rad, double vert_tip_load_N, bool flip_beam_width_height);
    ~TestBeam() { delete m_system; }

    ChSystem* GetSystem() { return m_system; }
    void ExecuteStep() { m_system->DoStepDynamics(1e-4); }
    void SimulateVis();
    void NonlinearStatics() { m_system->DoStaticNonlinear(50); };
    ChVector<> GetBeamMidPointPos() { return m_nodeMidPoint->GetPos(); }
    ChVector<> GetBeamEndPointPos() { return m_nodeEndPoint->GetPos(); }

  protected:
    ChSystemSMC* m_system;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeMidPoint;
    std::shared_ptr<ChNodeFEAxyzDD> m_nodeEndPoint;
};


TestBeam::TestBeam(int num_elements, double beam_angle_rad, double vert_tip_load_N, bool flip_beam_width_height) {
    m_system = new ChSystemSMC();
    m_system->Set_G_acc(ChVector<>(0, 0, -9.80665));

    // Set solver parameters
    auto solver_type = SolverType::MUMPS;

#ifndef CHRONO_MUMPS
    if (solver_type == SolverType::MUMPS) {
        solver_type = SolverType::SparseLU;
        std::cout << "WARNING! Chrono::MUMPS not enabled. Forcing use of Eigen SparseLU solver" << std::endl;
    }
#endif

    switch (solver_type) {
        case SolverType::MUMPS: {
#ifdef CHRONO_MUMPS
            auto solver = chrono_types::make_shared<ChSolverMumps>();
            solver->UseSparsityPatternLearner(true);
            solver->LockSparsityPattern(true);
            solver->SetVerbose(false);
            m_system->SetSolver(solver);
#endif
        break;
        }
        case SolverType::SparseLU: {
            auto solver = chrono_types::make_shared<ChSolverSparseLU>();
            solver->UseSparsityPatternLearner(true);
            solver->LockSparsityPattern(true);
            solver->SetVerbose(false);
            m_system->SetSolver(solver);
            break;
        }
    }

    // Set up integrator
    m_system->SetTimestepperType(ChTimestepper::Type::HHT);
    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(m_system->GetTimestepper());
    integrator->SetAlpha(-0.2);
    integrator->SetMaxiters(100);
    integrator->SetAbsTolerances(1e-5);
    integrator->SetMode(ChTimestepperHHT::ACCELERATION);
    integrator->SetScaling(true);
    integrator->SetVerbose(true);


    // Mesh properties
    double length = 20*0.0254; //Beam dimension were originally in inches
    //double width = 0.5*0.0254; //Beam dimension were originally in inches
    //double thickness = 0.125*0.0254; //Beam dimension were originally in inches
    double width;
    double thickness;

    if (!flip_beam_width_height) {
        width = 0.5*0.0254; //Beam dimension were originally in inches
        thickness = 0.125*0.0254; //Beam dimension were originally in inches
    }
    else {
        width = 0.125*0.0254; //Beam dimension were originally in inches
        thickness = 0.5*0.0254; //Beam dimension were originally in inches
    }


    //Aluminum 7075-T651
    double rho = 2810; //kg/m^3
    double E = 71.7e9; //Pa
    double nu = 0.33;
    double G = E/(2*(1+nu));
    double k1 = 10 * (1 + nu) / (12 + 11 * nu);  // Timoshenko shear correction coefficient for a rectangular cross-section
    double k2 = k1;                              // Timoshenko shear correction coefficient for a rectangular cross-section

    auto material = chrono_types::make_shared<ChMaterialBeamANCF>(rho, E, nu, k1, k2);


    // Create mesh container
    auto mesh = chrono_types::make_shared<ChMesh>();
    m_system->Add(mesh);

    //Setup visualization - Need to fix this section
    auto vis_surf = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_surf->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_SURFACE);
    vis_surf->SetWireframe(true);
    vis_surf->SetDrawInUndeformedReference(true);
    mesh->AddAsset(vis_surf);

    auto vis_node = chrono_types::make_shared<ChVisualizationFEAmesh>(*mesh);
    vis_node->SetFEMglyphType(ChVisualizationFEAmesh::E_GLYPH_NODE_DOT_POS);
    vis_node->SetFEMdataType(ChVisualizationFEAmesh::E_PLOT_NONE);
    vis_node->SetSymbolsThickness(0.01);
    mesh->AddAsset(vis_node);

    // Populate the mesh container with a the nodes and elements for the meshed beam
    int num_nodes = (2 * num_elements) + 1;
    double dx = length / (num_nodes-1);

    //Rotate the cross section gradients to match the sign convention in the experiment
    ChVector<> dir1(0, cos(-beam_angle_rad), sin(-beam_angle_rad));
    ChVector<> dir2(0, -sin(-beam_angle_rad), cos(-beam_angle_rad));

    auto nodeA = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(0, 0, 0.0), dir1, dir2);
    nodeA->SetFixed(true);
    mesh->AddNode(nodeA);

    for (int i = 1; i <= num_elements; i++) {
        auto nodeB = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx*(2 * i), 0, 0), dir1, dir2);
        auto nodeC = chrono_types::make_shared<ChNodeFEAxyzDD>(ChVector<>(dx*(2 * i - 1), 0, 0), dir1, dir2);
        mesh->AddNode(nodeB);
        mesh->AddNode(nodeC);

        auto element = chrono_types::make_shared<ChElementBeamANCF>();
        element->SetNodes(nodeA, nodeB, nodeC);
        element->SetDimensions(2*dx, thickness, width);
        element->SetMaterial(material);
        element->SetAlphaDamp(0.0);
        element->SetGravityOn(true);  //Enable the efficient ANCF method for calculating the application of gravity to the element
        element->SetStrainFormulation(ChElementBeamANCF::StrainFormulation::CMPoisson);
        //element->SetStrainFormulation(ChElementBeamANCF::StrainFormulation::CMNoPoisson);
        mesh->AddElement(element);

        //Cache the middle node of the beam for reporting displacements
        if ((2 * i - 1) == num_elements) {
            m_nodeMidPoint = nodeC;
        }
        else if ((2 * i) == num_elements) {
            m_nodeMidPoint = nodeB;
        }

        nodeA = nodeB;
    }

    m_nodeEndPoint = nodeA;

    mesh->SetAutomaticGravity(false); //Turn off the default method for applying gravity to the mesh since it is less efficient for ANCF elements

    m_nodeEndPoint->SetForce(ChVector<>(0, 0, -vert_tip_load_N));
}

void TestBeam::SimulateVis() {
#ifdef CHRONO_IRRLICHT
    irrlicht::ChIrrApp application(m_system, L"Princeton Beam Experiment - ANCF 3 Node Beams", irr::core::dimension2d<irr::u32>(800, 600), false, true);
    application.AddTypicalLogo();
    application.AddTypicalSky();
    application.AddTypicalLights();
    application.AddTypicalCamera(irr::core::vector3df(-0.2f, 0.2f, 0.2f), irr::core::vector3df(0, 0, 0));

    application.AssetBindAll();
    application.AssetUpdateAll();

    while (application.GetDevice()->run()) {
        application.BeginScene();
        application.DrawAll();
        irrlicht::ChIrrTools::drawSegment(application.GetVideoDriver(), ChVector<>(0), ChVector<>(1, 0, 0),
                                          irr::video::SColor(255, 255, 0, 0));
        irrlicht::ChIrrTools::drawSegment(application.GetVideoDriver(), ChVector<>(0), ChVector<>(0, 1, 0),
                                          irr::video::SColor(255, 0, 255, 0));
        irrlicht::ChIrrTools::drawSegment(application.GetVideoDriver(), ChVector<>(0), ChVector<>(0, 0, 1),
                                          irr::video::SColor(255, 0, 0, 255));
        ExecuteStep();
        application.EndScene();
    }
#endif
}


// =============================================================================

int main(int argc, char* argv[]) {

    ////Run the Princeton Beam Experiment (More points that the actual experiment):
    //// Loads = 0 to 4lbf in 0.5lbf increments
    //// Beam angles = 0 to 90 deg in 15deg increments

    //int num_elements = 100;
    //bool flip_beam_width_height = false;

    //// Output data
    //const std::string out_dir = GetChronoOutputPath() + "ANCF_PRINCETON_BEAM_STATICS";
    //if (!filesystem::create_directory(filesystem::path(out_dir))) {
    //    std::cout << "Error creating directory " << out_dir << std::endl;
    //    return 1;
    //}

    //std::string filename_X = out_dir + "/princeton_X.csv";
    //chrono::ChStreamOutAsciiFile file_outX(filename_X.c_str());

    //std::string filename_Z = out_dir + "/princeton_Z.csv";
    //chrono::ChStreamOutAsciiFile file_outZ(filename_Z.c_str());

    //std::string filename_W = out_dir + "/princeton_W.csv";
    //chrono::ChStreamOutAsciiFile file_outW(filename_W.c_str());

    //std::string filename_V = out_dir + "/princeton_V.csv";
    //chrono::ChStreamOutAsciiFile file_outV(filename_V.c_str());

    //for (int beam_angle_deg = 0; beam_angle_deg <= 90; beam_angle_deg += 15) {
    //    double beam_angle_rad = beam_angle_deg * CH_C_DEG_TO_RAD;
    //    TestBeam Beam_NonlinearStatics_NoLoad(num_elements, beam_angle_rad, 0, flip_beam_width_height);
    //    Beam_NonlinearStatics_NoLoad.NonlinearStatics();

    //    for (double load_lbf = 0.5; load_lbf <= 4; load_lbf += 0.5) {
    //        double load_N = load_lbf*4.44822;
    //        TestBeam Beam_NonlinearStatics_Loaded(num_elements, beam_angle_rad, load_N, flip_beam_width_height);
    //        Beam_NonlinearStatics_Loaded.NonlinearStatics();

    //        auto DeltaEndPoint_m = (Beam_NonlinearStatics_Loaded.GetBeamEndPointPos() - Beam_NonlinearStatics_NoLoad.GetBeamEndPointPos());

    //        file_outX <<  DeltaEndPoint_m.y() << ", " ;
    //        file_outZ << -DeltaEndPoint_m.z() << ", " ;
    //        file_outW <<  DeltaEndPoint_m.y()*cos(beam_angle_rad) + -DeltaEndPoint_m.z()*sin(beam_angle_rad) << ", " ;
    //        file_outV << -DeltaEndPoint_m.y()*sin(beam_angle_rad) + -DeltaEndPoint_m.z()*cos(beam_angle_rad) << ", " ;

    //        std::cout << "Angle(deg): " << beam_angle_deg << "  Load(lbf): " << load_lbf << "\n";
    //        std::cout << "X:" << DeltaEndPoint_m.y() << " m (" << DeltaEndPoint_m.y() / 0.0254 << "in)\n";
    //        std::cout << "Z:" << -DeltaEndPoint_m.z() << " m (" << -DeltaEndPoint_m.z() / 0.0254 << "in)\n";
    //        std::cout << "W:" << (DeltaEndPoint_m.y()*cos(beam_angle_rad) + -DeltaEndPoint_m.z()*sin(beam_angle_rad))
    //                  << " m (" << (DeltaEndPoint_m.y()*cos(beam_angle_rad) + -DeltaEndPoint_m.z()*sin(beam_angle_rad)) / 0.0254 << "in)\n";
    //        std::cout << "V:" << (-DeltaEndPoint_m.y()*sin(beam_angle_rad) + -DeltaEndPoint_m.z()*cos(beam_angle_rad))
    //                  << " m (" << (-DeltaEndPoint_m.y()*sin(beam_angle_rad) + -DeltaEndPoint_m.z()*cos(beam_angle_rad)) / 0.0254 << "in)\n\n";

    //    }

    //    file_outX << "\n";
    //    file_outZ << "\n";
    //    file_outW << "\n";
    //    file_outV << "\n";

    //}
    // 


//    //Run an individual case of the Princeton beam experiment
//    double load = 4 * 4.44822; //Experiment loads are given in lbf
//    int num_elements = 100;
//    
//#if false
//    //test case 1 (beam rotated 90 degrees by rotating the position vector gradients)
//    double beam_angle_rad = 90.0 * CH_C_DEG_TO_RAD; //Experiment beam angles are given in deg
//    bool flip_beam_width_height = false;
//#else
//    //test case 2 (beam rotated 90 degrees by flipping the width and height)
//    double beam_angle_rad = 0.0 * CH_C_DEG_TO_RAD; //Experiment beam angles are given in deg
//    bool flip_beam_width_height = true;
//#endif
//
//    TestBeam NonlinearStatics_NoLoad(num_elements, beam_angle_rad, 0*load, flip_beam_width_height);
//    TestBeam NonlinearStatics_Loaded(num_elements, beam_angle_rad, load, flip_beam_width_height);
//
//    NonlinearStatics_NoLoad.NonlinearStatics();
//    NonlinearStatics_Loaded.NonlinearStatics();
//
//    auto DeltaMidPoint_inch = (NonlinearStatics_Loaded.GetBeamMidPointPos() - NonlinearStatics_NoLoad.GetBeamMidPointPos()) / 0.0254;
//    auto DeltaEndPoint_inch = (NonlinearStatics_Loaded.GetBeamEndPointPos() - NonlinearStatics_NoLoad.GetBeamEndPointPos()) / 0.0254;
//    auto DeltaMidPoint_m = (NonlinearStatics_Loaded.GetBeamMidPointPos() - NonlinearStatics_NoLoad.GetBeamMidPointPos());
//    auto DeltaEndPoint_m = (NonlinearStatics_Loaded.GetBeamEndPointPos() - NonlinearStatics_NoLoad.GetBeamEndPointPos());
//
//    //std::cout << "Unloaded Mid Point Location - Model Coordinates:" << NonlinearStatics_NoLoad.GetBeamMidPointPos() / 0.0254 << " inches \n";
//    //std::cout << "Loaded Mid Point Location - Model Coordinates:" << NonlinearStatics_Loaded.GetBeamMidPointPos() / 0.0254 << " inches \n\n";
//    std::cout << "Unloaded End Point Location - Model Coordinates:" << NonlinearStatics_NoLoad.GetBeamEndPointPos() / 0.0254 << " inches \n";
//    std::cout << "Loaded End Point Location - Model Coordinates:" << NonlinearStatics_Loaded.GetBeamEndPointPos() / 0.0254 << " inches \n\n\n";
//
//    //std::cout << "Unloaded Mid Point Location - Model Coordinates:" << NonlinearStatics_NoLoad.GetBeamMidPointPos() << " m \n";
//    //std::cout << "Loaded Mid Point Location - Model Coordinates:" << NonlinearStatics_Loaded.GetBeamMidPointPos() << " m \n\n";
//    std::cout << "Unloaded End Point Location - Model Coordinates:" << NonlinearStatics_NoLoad.GetBeamEndPointPos() << " m \n";
//    std::cout << "Loaded End Point Location - Model Coordinates:" << NonlinearStatics_Loaded.GetBeamEndPointPos() << " m \n\n\n";
//
//    //std::cout << "Mid Point Location - Experiment Coordinates X:" << DeltaMidPoint_inch.y() << "  Z:" << -DeltaMidPoint_inch.z() << " inches \n";
//    //std::cout << "Mid Point Location - Experiment Coordinates W:" << DeltaMidPoint_inch.y()*cos(beam_angle_rad)+ -DeltaMidPoint_inch.z()*sin(beam_angle_rad)
//    //          << "  V:" << -DeltaMidPoint_inch.y()*sin(beam_angle_rad) + -DeltaMidPoint_inch.z()*cos(beam_angle_rad) << " inches \n\n";
//
//    //std::cout << "Mid Point Location - Experiment Coordinates X:" << DeltaMidPoint_m.y() << "  Z:" << -DeltaMidPoint_m.z() << " m \n";
//    //std::cout << "Mid Point Location - Experiment Coordinates W:" << DeltaMidPoint_m.y()*cos(beam_angle_rad) + -DeltaMidPoint_m.z()*sin(beam_angle_rad)
//    //    << "  V:" << -DeltaMidPoint_m.y()*sin(beam_angle_rad) + -DeltaMidPoint_m.z()*cos(beam_angle_rad) << " m \n\n";
//
//    std::cout << "End Point Location - Experiment Coordinates X:" << DeltaEndPoint_inch.y() << "  Z:" << -DeltaEndPoint_inch.z() << " inches \n";
//    //std::cout << "End Point Location - Experiment Coordinates W:" << DeltaEndPoint_inch.y()*cos(beam_angle_rad) + -DeltaEndPoint_inch.z()*sin(beam_angle_rad)
//    //    << "  V:" << -DeltaEndPoint_inch.y()*sin(beam_angle_rad) + -DeltaEndPoint_inch.z()*cos(beam_angle_rad) << " inches \n\n";
//
//    std::cout << "End Point Location - Experiment Coordinates X:" << DeltaEndPoint_m.y() << "  Z:" << -DeltaEndPoint_m.z() << " m \n";
//    //std::cout << "End Point Location - Experiment Coordinates W:" << DeltaEndPoint_m.y()*cos(beam_angle_rad) + -DeltaEndPoint_m.z()*sin(beam_angle_rad)
//    //    << "  V:" << -DeltaEndPoint_m.y()*sin(beam_angle_rad) + -DeltaEndPoint_m.z()*cos(beam_angle_rad) << " m \n\n";





    TestBeam test(8, 0, 4.44822*4, true);
    test.SimulateVis();

    //for (int i = 0; i < 10; i++) {
    //    test.ExecuteStep();
    //}

    return 0;

}
