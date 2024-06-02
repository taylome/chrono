message(STATUS)
message(STATUS "==== THESIS Benchmark programs ====")
message(STATUS)

#--------------------------------------------------------------
# Always use full RPATH (differentiating between the build and install trees)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

#--------------------------------------------------------------
# Add path to google benchmark

#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/chrono_thirdparty/googlebenchmark/include)

#--------------------------------------------------------------
# Propagate to subdirectories

option(BUILD_ANCF_ANALYSIS_BEAMTWO "Build FEA Analysis Programs for ANCF 2 Node Beam Elements" TRUE)
option(BUILD_ANCF_ANALYSIS_BEAMTHREE "Build FEA Analysis Programs for ANCF 3 Node Beam Elements" TRUE)
option(BUILD_ANCF_ANALYSIS_SHELLFOURFULL "Build FEA Analysis Programs for ANCF 4 Node Fully Parameterized Shell Elements" TRUE)
option(BUILD_ANCF_ANALYSIS_SHELLEIGHT "Build FEA Analysis Programs for ANCF 8 Node Higher Order Shell Elements" TRUE)
option(BUILD_ANCF_ANALYSIS_BRICKEIGHT "Build FEA Analysis Programs for ANCF 8 Node Fully Parameterized Brick Elements" TRUE)
option(BUILD_ANCF_ANALYSIS_COMMON "Build Common FEA Analysis Programs for Code Development" TRUE)
mark_as_advanced(FORCE BUILD_ANCF_ANALYSIS_BEAMTWO)
mark_as_advanced(FORCE BUILD_ANCF_ANALYSIS_BEAMTHREE)
mark_as_advanced(FORCE BUILD_ANCF_ANALYSIS_SHELLFOURFULL)
mark_as_advanced(FORCE BUILD_ANCF_ANALYSIS_SHELLEIGHT)
mark_as_advanced(FORCE BUILD_ANCF_ANALYSIS_BRICKEIGHT)
mark_as_advanced(FORCE BUILD_ANCF_ANALYSIS_COMMON)
if(BUILD_ANCF_ANALYSIS_BEAMTWO)
	add_subdirectory(ancf_2node_beam_3243)
endif()
if(BUILD_ANCF_ANALYSIS_BEAMTHREE)
	add_subdirectory(ancf_3node_beam_3333)
endif()
if(BUILD_ANCF_ANALYSIS_SHELLFOURFULL)
	add_subdirectory(ancf_4node_shell_3443)
endif()
if(BUILD_ANCF_ANALYSIS_SHELLEIGHT)
	add_subdirectory(ancf_8node_shell_3833)
endif()
if(BUILD_ANCF_ANALYSIS_BRICKEIGHT)
	add_subdirectory(ancf_8node_brick_3843)
endif()
if(BUILD_ANCF_ANALYSIS_COMMON)
	add_subdirectory(ancf_common)
endif()