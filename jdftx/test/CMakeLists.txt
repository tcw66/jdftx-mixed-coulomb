add_custom_target(testresults COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/printResults.sh ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} )
add_custom_target(testclean COMMAND rm -f */*.out */*.wfns */*.fillings */*.ionpos */*.eigenvals */*.fluidState */results */summary WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )

# Direct numerical regression for ordinary/derivative projector cache entries.
add_executable(checkProjectorCache koopmans/checkProjectorCache.cpp)
target_link_libraries(checkProjectorCache jdftxlib)
set_JDFTx_flags(checkProjectorCache OFF)
add_executable(checkKMeshCorrelation koopmans/checkKMeshCorrelation.cpp)
target_link_libraries(checkKMeshCorrelation jdftxlib)
set_JDFTx_flags(checkKMeshCorrelation OFF)
add_test(NAME kMeshCorrelation COMMAND checkKMeshCorrelation)
set_tests_properties(kMeshCorrelation PROPERTIES TIMEOUT 60)

add_executable(checkBerryPhase berryPhase/checkBerryPhase.cpp)
target_link_libraries(checkBerryPhase jdftxlib)
set_JDFTx_flags(checkBerryPhase OFF)
find_file(BERRY_SI_PSEUDO Si_ONCV_PBE.upf PATHS ${CMAKE_BINARY_DIR}/pseudopotentials/SG15 "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(BERRY_SI_PSEUDO)
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.in
		"ion-species ${BERRY_SI_PSEUDO}\nlattice Face-Centered Cubic 10.26\nion Si 0 0 0 1\nion Si 0.25 0.25 0.25 1\nelec-cutoff 8\nelec-n-bands 4\nkpoint-folding 3 3 3\nberry-phase-field 0.0002 -0.0001 0.0003\n")
	add_test(NAME berryPhase COMMAND checkBerryPhase -i ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.in -o ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.out)
	set_tests_properties(berryPhase PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
	file(READ ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.in BERRY_INPUT)
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-spin.in "${BERRY_INPUT}spintype z-spin\n")
	string(REPLACE "elec-n-bands 4" "elec-n-bands 8" BERRY_SPINOR_INPUT "${BERRY_INPUT}")
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-spinor.in "${BERRY_SPINOR_INPUT}spintype spin-orbit\n")
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-shift.in "${BERRY_INPUT}basis single\nkpoint 0.125 0.25 -0.1 1\n")
	string(REPLACE "kpoint-folding 3 3 3" "kpoint-folding 3 4 5" BERRY_NONCUBIC_INPUT "${BERRY_INPUT}")
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-noncubic.in "${BERRY_NONCUBIC_INPUT}")
	foreach(BERRY_CASE spin spinor shift noncubic)
		add_test(NAME berryPhase_${BERRY_CASE} COMMAND checkBerryPhase
			-i ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-${BERRY_CASE}.in
			-o ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-${BERRY_CASE}.out)
		set_tests_properties(berryPhase_${BERRY_CASE} PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
	endforeach()
	if(EnableMPI)
		add_test(NAME berryPhaseMPI COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
			${MPIEXEC_PREFLAGS} $<TARGET_FILE:checkBerryPhase> ${MPIEXEC_POSTFLAGS}
			-i ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.in -o ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-mpi.out)
		set_tests_properties(berryPhaseMPI PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
	endif()
endif()
add_executable(checkBatchedProjectors koopmans/checkBatchedProjectors.cpp)
target_link_libraries(checkBatchedProjectors jdftxlib)
set_JDFTx_flags(checkBatchedProjectors OFF)
add_test(NAME batchedProjectors COMMAND checkBatchedProjectors)
set_tests_properties(batchedProjectors PROPERTIES TIMEOUT 60)
add_executable(checkSpinKernel koopmans/checkSpinKernel.cpp)
target_link_libraries(checkSpinKernel jdftxlib)
set_JDFTx_flags(checkSpinKernel OFF)
add_test(NAME koopmansSpinKernel COMMAND checkSpinKernel)
set_tests_properties(koopmansSpinKernel PROPERTIES TIMEOUT 60)
if(EnableCUDA)
	add_executable(checkBerryPhase_gpu berryPhase/checkBerryPhase.cpp)
	target_link_libraries(checkBerryPhase_gpu jdftxlib_gpu CUDA::cudart)
	set_JDFTx_flags(checkBerryPhase_gpu ON)
	if(BERRY_SI_PSEUDO)
		add_test(NAME berryPhaseGPU COMMAND checkBerryPhase_gpu
			-i ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.in -o ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-gpu.out)
		set_tests_properties(berryPhaseGPU PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
		if(EnableMPI)
			add_test(NAME berryPhaseGPUMPI COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
				${MPIEXEC_PREFLAGS} $<TARGET_FILE:checkBerryPhase_gpu> ${MPIEXEC_POSTFLAGS}
				-i ${CMAKE_CURRENT_BINARY_DIR}/berry-phase.in -o ${CMAKE_CURRENT_BINARY_DIR}/berry-phase-gpu-mpi.out)
			set_tests_properties(berryPhaseGPUMPI PROPERTIES TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
		endif()
	endif()
	add_executable(checkProjectorCache_gpu koopmans/checkProjectorCache.cpp)
	target_link_libraries(checkProjectorCache_gpu jdftxlib_gpu)
	set_JDFTx_flags(checkProjectorCache_gpu ON)
	add_executable(checkKMeshCorrelation_gpu koopmans/checkKMeshCorrelation.cpp)
	target_link_libraries(checkKMeshCorrelation_gpu jdftxlib_gpu)
	set_JDFTx_flags(checkKMeshCorrelation_gpu ON)
	add_test(NAME kMeshCorrelationGPU COMMAND checkKMeshCorrelation_gpu)
	set_tests_properties(kMeshCorrelationGPU PROPERTIES TIMEOUT 60)
	add_executable(checkBatchedProjectors_gpu koopmans/checkBatchedProjectors.cpp)
	target_link_libraries(checkBatchedProjectors_gpu jdftxlib_gpu)
	set_JDFTx_flags(checkBatchedProjectors_gpu ON)
	add_test(NAME batchedProjectorsGPU COMMAND checkBatchedProjectors_gpu)
	set_tests_properties(batchedProjectorsGPU PROPERTIES TIMEOUT 60)
	add_executable(checkSpinKernel_gpu koopmans/checkSpinKernel.cpp)
	target_link_libraries(checkSpinKernel_gpu jdftxlib_gpu CUDA::cudart)
	set_JDFTx_flags(checkSpinKernel_gpu ON)
	add_test(NAME koopmansSpinKernelGPU COMMAND checkSpinKernel_gpu)
	set_tests_properties(koopmansSpinKernelGPU PROPERTIES TIMEOUT 60)
endif()

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
		if(BERRY_SI_PSEUDO)
			add_test(NAME berryPhaseWorkflow COMMAND ${Python3_EXECUTABLE}
				${CMAKE_CURRENT_SOURCE_DIR}/berryPhase/checkWorkflow.py ${CMAKE_BINARY_DIR}
				${CMAKE_SOURCE_DIR}/../examples/berry-dielectric-silicon)
			set_tests_properties(berryPhaseWorkflow PROPERTIES TIMEOUT 300)
		endif()
        add_test(NAME dfptInterpolation COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/testInterpolation.py
            ${CMAKE_SOURCE_DIR}/scripts/phononDFPT)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import scipy"
            RESULT_VARIABLE SCIPY_STATUS OUTPUT_QUIET ERROR_QUIET)
        if(SCIPY_STATUS EQUAL 0)
            add_test(NAME migdalEliashberg COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/testMigdalEliashberg.py)
            set_tests_properties(migdalEliashberg PROPERTIES TIMEOUT 60
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            add_test(NAME selfConsistentBdG COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/testSelfConsistentBdG.py)
            set_tests_properties(selfConsistentBdG PROPERTIES TIMEOUT 60
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            add_test(NAME localBdGProfiles COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/testLocalBdGProfiles.py)
            set_tests_properties(localBdGProfiles PROPERTIES TIMEOUT 60
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
            add_test(NAME embeddedBdG COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/testEmbeddedBdG.py)
            set_tests_properties(embeddedBdG PROPERTIES TIMEOUT 60
                ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
        endif()
    endif()
endif()

# Enable the physical regression when the SG15 norm-conserving library is available.
find_file(DFPT_H_PSEUDO H_ONCV_PBE.upf PATHS
    ${CMAKE_BINARY_DIR}/pseudopotentials/SG15
    "$ENV{JDFTX_PSEUDO_DIR}/SG15")
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0 AND DFPT_H_PSEUDO)
	file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/projector-cache.in
		"ion-species ${DFPT_H_PSEUDO}\nlattice Cubic 8\nion H 0.4 0.5 0.5 1\nion H 0.6 0.5 0.5 1\nelec-cutoff 6\nelec-n-bands 4\nsymmetries none\n")
	add_test(NAME projectorCache COMMAND checkProjectorCache
		-i ${CMAKE_CURRENT_BINARY_DIR}/projector-cache.in -o ${CMAKE_CURRENT_BINARY_DIR}/projector-cache.out)
	set_tests_properties(projectorCache PROPERTIES TIMEOUT 60)
	if(EnableCUDA)
		add_test(NAME projectorCacheGPU COMMAND checkProjectorCache_gpu
			-i ${CMAKE_CURRENT_BINARY_DIR}/projector-cache.in -o ${CMAKE_CURRENT_BINARY_DIR}/projector-cache-gpu.out)
		set_tests_properties(projectorCacheGPU PROPERTIES TIMEOUT 60)
	endif()
	add_test(NAME koopmansKI COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkKI.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	set_tests_properties(koopmansKI PROPERTIES TIMEOUT 300)
	add_test(NAME koopmansSpinKI COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkKI.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} --spin)
	set_tests_properties(koopmansSpinKI PROPERTIES TIMEOUT 300)
	add_test(NAME koopmansSpinKIPBE COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkKI.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} --spin --gga)
	set_tests_properties(koopmansSpinKIPBE PROPERTIES TIMEOUT 300)
	add_test(NAME koopmansSymmetry COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkSymmetry.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	set_tests_properties(koopmansSymmetry PROPERTIES TIMEOUT 300)
	add_test(NAME koopmansReuse COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkReuse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	set_tests_properties(koopmansReuse PROPERTIES TIMEOUT 300)
	add_test(NAME koopmansScreeningMesh COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkScreeningMesh.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	set_tests_properties(koopmansScreeningMesh PROPERTIES TIMEOUT 300)
	execute_process(COMMAND ${Python3_EXECUTABLE} -c "import scipy"
		RESULT_VARIABLE SCIPY_STATUS OUTPUT_QUIET ERROR_QUIET)
	if(SCIPY_STATUS EQUAL 0)
		add_test(NAME koopmansCoarseWorkflow COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkCoarseWorkflow.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
		set_tests_properties(koopmansCoarseWorkflow PROPERTIES TIMEOUT 600)
		if(EnableMPI AND MPIEXEC_EXECUTABLE)
			add_test(NAME koopmansCoarseWorkflowMPI COMMAND ${Python3_EXECUTABLE}
				${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkCoarseWorkflow.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} --launcher ${MPIEXEC_EXECUTABLE})
			set_tests_properties(koopmansCoarseWorkflowMPI PROPERTIES PROCESSORS 2 TIMEOUT 900)
		endif()
	endif()
	add_test(NAME koopmansOccupied COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkOccupied.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	set_tests_properties(koopmansOccupied PROPERTIES TIMEOUT 300)
	add_test(NAME koopmansResponseSpeedups COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkResponseSpeedups.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO})
	set_tests_properties(koopmansResponseSpeedups PROPERTIES TIMEOUT 600)
	if(EnableMPI AND MPIEXEC_EXECUTABLE)
		add_test(NAME koopmansScreeningMeshMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkScreeningMesh.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} --launcher ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansScreeningMeshMPI PROPERTIES PROCESSORS 2 TIMEOUT 300)
		add_test(NAME koopmansResponseSpeedupsMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkResponseSpeedups.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO}
			--launcher ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansResponseSpeedupsMPI PROPERTIES PROCESSORS 2 TIMEOUT 900)
		add_test(NAME koopmansQGroupsMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkQGroups.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansQGroupsMPI PROPERTIES PROCESSORS 4 TIMEOUT 600)
		add_test(NAME koopmansKIMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkKI.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansKIMPI PROPERTIES PROCESSORS 2 TIMEOUT 300)
		add_test(NAME koopmansSpinKIMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkKI.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE} --spin)
		set_tests_properties(koopmansSpinKIMPI PROPERTIES PROCESSORS 2 TIMEOUT 300)
		add_test(NAME koopmansSymmetryMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkSymmetry.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansSymmetryMPI PROPERTIES PROCESSORS 2 TIMEOUT 300)
		add_test(NAME koopmansReuseMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkReuse.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansReuseMPI PROPERTIES PROCESSORS 2 TIMEOUT 300)
		add_test(NAME koopmansOccupiedMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/koopmans/checkOccupied.py ${CMAKE_BINARY_DIR} ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE})
		set_tests_properties(koopmansOccupiedMPI PROPERTIES PROCESSORS 2 TIMEOUT 300)
	endif()
	add_test(NAME dfptElectronPhonon COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhonon.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO})
	add_test(NAME dfptQBatches COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO})
	add_test(NAME dfptQBatchesMetallic COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --metallic)
	add_test(NAME dfptQBatchesPhononOnly COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --phonon-only)
	set_tests_properties(dfptQBatches dfptQBatchesMetallic dfptQBatchesPhononOnly PROPERTIES TIMEOUT 600)
	add_test(NAME dfptKBatchesMetallic COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --metallic)
	add_test(NAME dfptKBatchesPhononOnly COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --phonon-only)
	set_tests_properties(dfptKBatchesMetallic dfptKBatchesPhononOnly PROPERTIES TIMEOUT 600
		ENVIRONMENT "DFPT_K_BATCH_SIZE=1")
	add_test(NAME dfptIBZStorageNSCF COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --metallic)
	add_test(NAME dfptIBZStorageOffMesh COMMAND ${Python3_EXECUTABLE}
		${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --phonon-only)
	set_tests_properties(dfptIBZStorageNSCF PROPERTIES TIMEOUT 600
		ENVIRONMENT "DFPT_K_BATCH_SIZE=1;DFPT_IBZ_STORAGE=1;DFPT_IBZ_NSCF=1")
	set_tests_properties(dfptIBZStorageOffMesh PROPERTIES TIMEOUT 600
		ENVIRONMENT "DFPT_K_BATCH_SIZE=1;DFPT_IBZ_STORAGE=1")
	if(EnableCUDA)
		add_test(NAME dfptQBatchesGPU COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx_gpu ${DFPT_H_PSEUDO})
		set_tests_properties(dfptQBatchesGPU PROPERTIES TIMEOUT 600 RESOURCE_LOCK jdftx_gpu
			ENVIRONMENT "DFPT_BATCH_BINARY=${CMAKE_BINARY_DIR}/jdftx")
		add_test(NAME dfptKBatchesGPU COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx_gpu ${DFPT_H_PSEUDO} --metallic)
		set_tests_properties(dfptKBatchesGPU PROPERTIES TIMEOUT 600 RESOURCE_LOCK jdftx_gpu
			ENVIRONMENT "DFPT_K_BATCH_SIZE=1")
		add_test(NAME dfptIBZStorageGPU COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx_gpu ${DFPT_H_PSEUDO} --metallic)
		set_tests_properties(dfptIBZStorageGPU PROPERTIES TIMEOUT 600 RESOURCE_LOCK jdftx_gpu
			ENVIRONMENT "DFPT_K_BATCH_SIZE=1;DFPT_IBZ_STORAGE=1;DFPT_IBZ_NSCF=1")
	endif()
	if(EnableMPI AND MPIEXEC_EXECUTABLE)
		add_test(NAME dfptQBatchesMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} ${MPIEXEC_EXECUTABLE})
		set_tests_properties(dfptQBatchesMPI PROPERTIES TIMEOUT 600 PROCESSORS 4)
		add_test(NAME dfptKBatchesMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --metallic ${MPIEXEC_EXECUTABLE})
		set_tests_properties(dfptKBatchesMPI PROPERTIES TIMEOUT 600 PROCESSORS 4
			ENVIRONMENT "DFPT_K_BATCH_SIZE=1")
		add_test(NAME dfptIBZStorageMPI COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkQBatches.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_H_PSEUDO} --metallic ${MPIEXEC_EXECUTABLE})
		set_tests_properties(dfptIBZStorageMPI PROPERTIES TIMEOUT 600 PROCESSORS 4
			ENVIRONMENT "DFPT_K_BATCH_SIZE=1;DFPT_IBZ_STORAGE=1")
	endif()
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
	if(EnableMPI AND MPIEXEC_EXECUTABLE)
		add_test(NAME dfptIBZStorageSymmetry COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO}
			--ibz-storage --launcher ${MPIEXEC_EXECUTABLE} --rank-flag=${MPIEXEC_NUMPROC_FLAG})
		set_tests_properties(dfptIBZStorageSymmetry PROPERTIES TIMEOUT 900 PROCESSORS 3)
	else()
		add_test(NAME dfptIBZStorageSymmetry COMMAND ${Python3_EXECUTABLE}
			${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO} --ibz-storage)
		set_tests_properties(dfptIBZStorageSymmetry PROPERTIES TIMEOUT 900)
	endif()
    add_test(NAME dfptElectronPhononSymmetry COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO})
    set_tests_properties(dfptElectronPhononSymmetry PROPERTIES TIMEOUT 1200)
    if(EnableMPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME dfptElectronPhononSymmetryMPI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_SI_PSEUDO}
            --launcher ${MPIEXEC_EXECUTABLE} --rank-flag=${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(dfptElectronPhononSymmetryMPI PROPERTIES PROCESSORS 3 TIMEOUT 1200)
    endif()
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
    add_test(NAME dfptElectronPhononSymmetryNb COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO} --species Nb)
    set_tests_properties(dfptElectronPhononSymmetryNb PROPERTIES TIMEOUT 1200)
    if(EnableMPI AND MPIEXEC_EXECUTABLE)
        add_test(NAME dfptElectronPhononSymmetryNbMPI COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_NB_PSEUDO}
            --species Nb --launcher ${MPIEXEC_EXECUTABLE} --rank-flag=${MPIEXEC_NUMPROC_FLAG})
        set_tests_properties(dfptElectronPhononSymmetryNbMPI PROPERTIES PROCESSORS 3 TIMEOUT 1200)
    endif()
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
    add_test(NAME dfptElectronPhononSymmetryRotations COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkElectronPhononSymmetry.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_B_PSEUDO}
        --species BN --nitrogen ${DFPT_N_PSEUDO})
    set_tests_properties(dfptElectronPhononSymmetryRotations PROPERTIES TIMEOUT 300)
    add_test(NAME dfptDisplacementRotations COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/dfpt/checkDisplacementRotations.py ${CMAKE_BINARY_DIR}/jdftx ${DFPT_B_PSEUDO} ${DFPT_N_PSEUDO})
    set_tests_properties(dfptDisplacementRotations PROPERTIES TIMEOUT 150)
endif()

# Joint subspace/gauge gradient and minimization checks for the TDC measure.
if(Python3_Interpreter_FOUND AND NUMPY_STATUS EQUAL 0)
    add_test(NAME wannierFixedCenters COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/wannier/checkFixedCenters.py ${CMAKE_BINARY_DIR})
    set_tests_properties(wannierFixedCenters PROPERTIES TIMEOUT 180)
    add_test(NAME wannierFixedCentersScaling COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/wannier/benchmarkFixedCenters.py ${CMAKE_BINARY_DIR})
    set_tests_properties(wannierFixedCentersScaling PROPERTIES TIMEOUT 180)
endif()
if(Python3_Interpreter_FOUND AND BERRY_SI_PSEUDO)
    add_test(NAME wannierTruncatedDC COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/wannier/checkTruncatedDC.py ${CMAKE_BINARY_DIR})
    set_tests_properties(wannierTruncatedDC PROPERTIES TIMEOUT 180)
    if(NUMPY_STATUS EQUAL 0)
        add_test(NAME wannierFixedAtomicCenters COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/wannier/checkFixedAtomicCenters.py ${CMAKE_BINARY_DIR})
        set_tests_properties(wannierFixedAtomicCenters PROPERTIES TIMEOUT 180)
    endif()
endif()
