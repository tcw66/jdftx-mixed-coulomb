# Populate the build-tree library even for explicit executable/preset targets.
# A stamp avoids extracting the bundled archives on every incremental build.
set(libraryOutputs
 "${CMAKE_CURRENT_BINARY_DIR}/.library-ready"
 "${CMAKE_CURRENT_BINARY_DIR}/SG15/H_ONCV_PBE.upf"
 "${CMAKE_CURRENT_BINARY_DIR}/SG15/Si_ONCV_PBE.upf"
 "${CMAKE_CURRENT_BINARY_DIR}/SG15/Si_ONCV_PBE.pulay"
 "${CMAKE_CURRENT_BINARY_DIR}/GBRV/h_pbe.uspp")
add_custom_command(OUTPUT ${libraryOutputs}
 COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/update.sh"
  "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}"
 COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/.library-ready"
 DEPENDS update.sh GBRV.tgz GBRV-2017.tgz SG15.tgz SG15-pulay.tgz
 COMMENT "Preparing bundled SG15 and GBRV pseudopotentials"
 VERBATIM)
add_custom_target(PseudopotentialLibrary ALL DEPENDS ${libraryOutputs})

foreach(executable IN ITEMS jdftx wannier phonon ConstructDefectPotential
 jdftx_gpu wannier_gpu phonon_gpu ConstructDefectPotential_gpu)
 if(TARGET ${executable})
  add_dependencies(${executable} PseudopotentialLibrary)
 endif()
endforeach()
