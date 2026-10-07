add_custom_target(testresults COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/printResults.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} )
add_custom_target(testclean COMMAND rm -f */*.out */*.wfns */*.fillings */*.ionpos */*.eigenvals */*.fluidState */results */summary WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )

macro(add_jdftx_test testName)
	add_test(NAME ${testName} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/runTest.sh ${testName} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR})
endmacro()

add_jdftx_test(openShell)
add_jdftx_test(vibrations)
add_jdftx_test(moleculeSolvation)
add_jdftx_test(ionSolvation)
add_jdftx_test(latticeOpt)
add_jdftx_test(metalBulk)
add_jdftx_test(plusU)
add_jdftx_test(spinOrbit)
add_jdftx_test(graphene)
add_jdftx_test(metalSurface)
add_jdftx_test(checkpoint)

add_jdftx_test(defectCoulomb)

# End-to-end one-shot generation, native potential comparison and an NSCF solve.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME defectPotential COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/defectCoulomb/checkPotential.py
        ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/defectCoulomb
        ${CMAKE_SOURCE_DIR}/scripts/defectDensityToVscloc)
    set_tests_properties(defectCoulomb PROPERTIES FIXTURES_SETUP defectDensities)
    set_tests_properties(defectPotential PROPERTIES FIXTURES_REQUIRED defectDensities)
    add_test(NAME defectEmbedding COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/defectCoulomb/checkEmbedding.py
        ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/defectCoulomb
        ${CMAKE_SOURCE_DIR}/scripts/defectDensityToVscloc)
endif()



if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE NUMPY_STATUS OUTPUT_QUIET ERROR_QUIET)
    if(NUMPY_STATUS EQUAL 0)
        add_test(NAME dfptInterpolation COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/testInterpolation.py
            ${CMAKE_SOURCE_DIR}/scripts/phononDFPT)
    endif()
endif()

# Enable the physical regression when the SG15 norm-conserving library is available.
find_file(DFPT_H_PSEUDO H_ONCV_PBE.upf PATHS
    ${CMAKE_BINARY_DIR}/pseudopotentials/SG15
    "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0 AND DFPT_H_PSEUDO)
	add_test(NAME dfptElectronPhonon COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhonon.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO})
	add_test(NAME dfptElectronPhononEmbeddedSlab COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhonon.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --embedded)
	set_tests_properties(dfptElectronPhonon dfptElectronPhononEmbeddedSlab PROPERTIES TIMEOUT 300)
	if(EnableMPI AND MPIEXEC_EXECUTABLE)
		add_test(NAME dfptElectronPhononMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhonon.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO}
			${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG})
		set_tests_properties(dfptElectronPhononMPI PROPERTIES PROCESSORS 4 TIMEOUT 300)
	endif()
	add_test(NAME dfptMetallicFermi COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkMetallic.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	add_test(NAME dfptMetallicGauss COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkMetallic.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} Gauss)
	set_tests_properties(dfptMetallicFermi dfptMetallicGauss PROPERTIES TIMEOUT 300)
    add_test(NAME dfptRestart COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkRestart.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO})
    add_test(NAME dfptResponse COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkResponse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
    add_test(NAME dfptResponseSlab COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkResponse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} gga-PBE slab)
    add_test(NAME dfptResponseEmbeddedSlab COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkResponse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} gga-PBE embedded)
    add_test(NAME dfptResponseEmbeddedSlabLDA COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkResponse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} lda-PZ embedded)
    add_test(NAME dfptResponseSlabLDA COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkResponse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} lda-PZ slab)
    add_test(NAME dfptResponseLDA COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkResponse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} lda-PZ)
    if(EnableMPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME dfptQParallel COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQParallel.py ${CMAKE_BINARY_DIR}/jdftx
            ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(dfptQParallel PROPERTIES PROCESSORS 4 TIMEOUT 180)
    endif()
endif()

find_file(DFPT_AL_PSEUDO Al_ONCV_PBE.upf PATHS
    ${CMAKE_BINARY_DIR}/pseudopotentials/SG15 "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0 AND DFPT_AL_PSEUDO)
    add_test(NAME dfptMetallicAl COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkMetallicAl.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_AL_PSEUDO})
    set_tests_properties(dfptMetallicAl PROPERTIES TIMEOUT 600)
    if(EnableMPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME dfptMetallicAlMPI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkMetallicAl.py ${CMAKE_BINARY_DIR}/jdftx
            ${DFPT_AL_PSEUDO} ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(dfptMetallicAlMPI PROPERTIES PROCESSORS 4 TIMEOUT 600)
    endif()
endif()

# Silicon exercises nonsymmorphic atom mappings on an anisotropic q grid.
find_file(DFPT_SI_PSEUDO Si_ONCV_PBE.upf PATHS
    ${CMAKE_BINARY_DIR}/pseudopotentials/SG15
    "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0 AND DFPT_SI_PSEUDO)
    add_test(NAME dfptQSymmetry COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO})
    add_test(NAME dfptAcceleration COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkAcceleration.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO})
    set_tests_properties(dfptAcceleration PROPERTIES TIMEOUT 300)
    add_test(NAME dfptFrozenCache COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO})
    set_tests_properties(dfptFrozenCache PROPERTIES TIMEOUT 300)
    add_test(NAME dfptMainSpeedups COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO}
        --main-speedups)
    set_tests_properties(dfptMainSpeedups PROPERTIES TIMEOUT 400)
    if(EnableMPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME dfptAccelerationMPI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkAcceleration.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO}
            --launcher ${MPIEXEC_EXECUTABLE} --rank-flag=${MPIEXEC_NUMPROC_FLAG} --ranks 4)
        set_tests_properties(dfptAccelerationMPI PROPERTIES PROCESSORS 4 TIMEOUT 300)
    endif()
endif()

find_file(DFPT_NB_PSEUDO Nb_ONCV_PBE.upf PATHS
    ${CMAKE_BINARY_DIR}/pseudopotentials/SG15 "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0 AND DFPT_NB_PSEUDO)
    add_test(NAME dfptFrozenCacheNb COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO} --species Nb)
    set_tests_properties(dfptFrozenCacheNb PROPERTIES TIMEOUT 300)
    add_test(NAME dfptMainSpeedupsNb COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO}
        --species Nb --main-speedups)
    set_tests_properties(dfptMainSpeedupsNb PROPERTIES TIMEOUT 400)
    # The 64-point mesh cuts a degenerate empty manifold at some k points:
    # physical responses, rather than projector-dependent solver variables,
    # must be transformed between electronic symmetry images.
    add_test(NAME dfptKSymmetryNbDense COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO}
        --species Nb --main-speedups --k-only --fold 4)
    set_tests_properties(dfptKSymmetryNbDense PROPERTIES TIMEOUT 400)
    if(EnableMPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME dfptFrozenCacheNbMPI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO}
            --species Nb --launcher ${MPIEXEC_EXECUTABLE} --rank-flag=${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(dfptFrozenCacheNbMPI PROPERTIES PROCESSORS 3 TIMEOUT 400)
        add_test(NAME dfptMainSpeedupsNbMPI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkFrozenCache.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO}
            --species Nb --main-speedups --launcher ${MPIEXEC_EXECUTABLE} --rank-flag=${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(dfptMainSpeedupsNbMPI PROPERTIES PROCESSORS 3 TIMEOUT 500)
    endif()
endif()

# Hexagonal rotations mix Cartesian directions rather than just permuting them.
find_file(DFPT_B_PSEUDO B_ONCV_PBE.upf PATHS ${CMAKE_BINARY_DIR}/pseudopotentials/SG15 "$ENV{JDFTX_PSEUDO_DIR}/SG15")
find_file(DFPT_N_PSEUDO N_ONCV_PBE.upf PATHS ${CMAKE_BINARY_DIR}/pseudopotentials/SG15 "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0 AND DFPT_B_PSEUDO AND DFPT_N_PSEUDO)
    add_test(NAME dfptDisplacementRotations COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkDisplacementRotations.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_B_PSEUDO} ${DFPT_N_PSEUDO})
    set_tests_properties(dfptDisplacementRotations PROPERTIES TIMEOUT 150)
endif()
