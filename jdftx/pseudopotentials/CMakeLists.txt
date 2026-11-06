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

# Keep the ordinary ion-species wildcard syntax for native PAW datasets:
# PAW/$ID_PBE.xml resolves through the same build/install prefixes as SG15/GBRV.
# COPYONLY preserves dataset fingerprints and makes installed files standalone.
file(GLOB pawDatasets CONFIGURE_DEPENDS "${PAW_JTH_DIR}/*.GGA_PBE-JTH*.xml")
foreach(dataset IN LISTS pawDatasets)
 get_filename_component(datasetName "${dataset}" NAME)
 string(REPLACE ".GGA_PBE-JTH" "_PBE" libraryName "${datasetName}")
 configure_file("${dataset}" "${CMAKE_CURRENT_BINARY_DIR}/PAW/${libraryName}" COPYONLY)
endforeach()
if(pawDatasets)
 install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/PAW"
  DESTINATION share/jdftx/pseudopotentials)
endif()

foreach(executable IN ITEMS jdftx wannier phonon ConstructDefectPotential
 jdftx_gpu wannier_gpu phonon_gpu ConstructDefectPotential_gpu)
 if(TARGET ${executable})
  add_dependencies(${executable} PseudopotentialLibrary)
 endif()
endforeach()
