set(TESTS
	btest_FEA_ANCFbeam_3333_MR_Roofline
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_8
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_16
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_32
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_64
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_128
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_256
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_512
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_1024
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_2048
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_4096
	btest_FEA_ANCFbeam_3333_MR_RooflineSplit_8192
	utest_FEA_ANCFbeam_3333_MR_InternalFrcKRM
    )

# ------------------------------------------------------------------------------

include_directories(${CH_INCLUDES})
set(COMPILER_FLAGS "${CH_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")
list(APPEND LIBS "ChronoEngine")

if(ENABLE_MODULE_IRRLICHT)
  include_directories(${CH_IRRLICHTINC})
  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${CH_IRRLICHT_CXX_FLAGS}")
  list(APPEND LIBS "ChronoEngine_irrlicht")
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

if(ENABLE_MODULE_MKL OR ENABLE_MODULE_MUMPS)
  set(TESTS ${TESTS} ${TESTS_MKL_MUMPS})
endif()

# ------------------------------------------------------------------------------

message(STATUS "Benchmark test programs for FEA module...")

foreach(PROGRAM ${TESTS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp")
    source_group(""  FILES "${PROGRAM}.cpp")

    set_target_properties(${PROGRAM} PROPERTIES
        FOLDER tests
        COMPILE_FLAGS "${COMPILER_FLAGS}"
        LINK_FLAGS "${LINKER_FLAGS}")
	set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
#    target_link_libraries(${PROGRAM} ${LIBS} benchmark_main)
    target_link_libraries(${PROGRAM} ${LIBS})
endforeach(PROGRAM)
