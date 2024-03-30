#=============================================================================
# CMake configuration file for demos of various Chrono::Vehicle models.
# These example programs require Irrlicht run-time visualization.
#=============================================================================

#--------------------------------------------------------------
# List all demos

set(DEMOS
    demo_VEH_M113_Band_Timing
)

if(ENABLE_MODULE_IRRLICHT)
	set(DEMOS ${DEMOS}
		demo_VEH_M113
		demo_VEH_M113_DefSoil
		demo_VEH_Marder
		demo_VEH_TrackedJSON
		demo_VEH_TrackedJSON_Band
	)
endif()

if(ENABLE_MODULE_IRRLICHT AND(ENABLE_MODULE_PARDISO_MKL OR ENABLE_MODULE_MUMPS))
    set(DEMOS ${DEMOS}
        demo_VEH_M113_Band
    )
endif()

#--------------------------------------------------------------
# Include directories, compiler flags, libraries

if(ENABLE_MODULE_IRRLICHT)
	include_directories(${CH_IRRLICHTINC})
	set(COMPILER_FLAGS "${CH_CXX_FLAGS}  ${CH_IRRLICHT_CXX_FLAGS}")
	set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")
	list(APPEND LIBS "ChronoEngine_irrlicht")
	list(APPEND LIBS "ChronoEngine_vehicle_irrlicht")
endif()

if(ENABLE_MODULE_PARDISO_MKL)
    include_directories(${CH_MKL_INCLUDES})
    set(COMPILER_FLAGS "${COMPILER_FLAGS} ${CH_MKL_CXX_FLAGS}")
    set(LINKER_FLAGS "${LINKER_FLAGS} ${CH_MKL_LINK_FLAGS}")
    list(APPEND LIBS "ChronoEngine_pardisomkl")
endif()

if(ENABLE_MODULE_MUMPS)
    include_directories(${CH_MUMPS_INCLUDES})
    list(APPEND LIBS "ChronoEngine_mumps")
endif()

#--------------------------------------------------------------
# Create the executables

foreach(PROGRAM ${DEMOS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM} ${PROGRAM}.cpp ../SetChronoSolver.h)
    source_group("" FILES ${PROGRAM}.cpp ../SetChronoSolver.h)

    set_target_properties(${PROGRAM} PROPERTIES
                          COMPILE_FLAGS "${COMPILER_FLAGS}"
                          LINK_FLAGS "${LINKER_FLAGS}")
    if(APPLE)
      set_target_properties(${PROGRAM} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${MACOSX_BUNDLE_INFO_PLIST})
    endif()
    set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
    target_link_libraries(${PROGRAM}
                          ChronoEngine
                          ChronoEngine_vehicle
                          ChronoModels_vehicle
                          ${LIBS})

    install(TARGETS ${PROGRAM} DESTINATION ${CH_INSTALL_DEMO})
endforeach()
