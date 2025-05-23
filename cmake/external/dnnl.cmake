include(ExternalProject)

set(DNNL_TAR ${CMAKE_SOURCE_DIR}/third_party/oneDNN/v3.0.tar.gz)
set(DNNL_SHARED_LIB libdnnl.so.3)

if(${HYDRAULIS_COMPILE_DNNL})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${DNNL_TAR} 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )

  set(DNNL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/third_party/oneDNN-3.0)
  set(DNNL_INSTALL ${DNNL_SOURCE}/install)
  set(DNNL_INCLUDE_DIR ${DNNL_INSTALL}/include)
  set(DNNL_LIB_DIR ${DNNL_INSTALL}/lib)
  set(DNNL_DLL_PATH ${DNNL_LIB_DIR}/${DNNL_SHARED_LIB})
  set(DNNL_CMAKE_EXTRA_ARGS)

  ExternalProject_Add(project_dnnl
    PREFIX dnnl
    # PATCH_COMMAND ${MKLDNN_PATCH_DISCARD_COMMAND} COMMAND ${DNNL_PATCH_COMMAND}
    SOURCE_DIR ${DNNL_SOURCE}
    CMAKE_ARGS -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_CONCURRENT_EXEC=ON -DDNNL_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${DNNL_INSTALL}
  )
  link_directories(${DNNL_LIB_DIR})
endif()
