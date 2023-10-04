# Install script for directory: /home/thaer/pygmo2/pygmo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/thaer/pygmo2/build/pygmo/plotting/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/lib/python3.10/site-packages/pygmo" TYPE MODULE FILES "/home/thaer/pygmo2/build/pygmo/core.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/usr/local/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/lib/python3.10/site-packages/pygmo/core.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/python3.10/site-packages/pygmo/__init__.py;/usr/local/lib/python3.10/site-packages/pygmo/_check_deps.py;/usr/local/lib/python3.10/site-packages/pygmo/test.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_problem.py;/usr/local/lib/python3.10/site-packages/pygmo/_problem_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_algorithm.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_bfe.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_island.py;/usr/local/lib/python3.10/site-packages/pygmo/_py_bfes.py;/usr/local/lib/python3.10/site-packages/pygmo/_mp_utils.py;/usr/local/lib/python3.10/site-packages/pygmo/_py_islands.py;/usr/local/lib/python3.10/site-packages/pygmo/_py_algorithms.py;/usr/local/lib/python3.10/site-packages/pygmo/_algorithm_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_bfe_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_island_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_r_policy.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_s_policy.py;/usr/local/lib/python3.10/site-packages/pygmo/_patch_topology.py;/usr/local/lib/python3.10/site-packages/pygmo/_s_policy_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_r_policy_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_topology_test.py;/usr/local/lib/python3.10/site-packages/pygmo/_py_problems.py;/usr/local/lib/python3.10/site-packages/pygmo/_ipyparallel_utils.py;/usr/local/lib/python3.10/site-packages/pygmo/_version.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/lib/python3.10/site-packages/pygmo" TYPE FILE FILES
    "/home/thaer/pygmo2/pygmo/__init__.py"
    "/home/thaer/pygmo2/pygmo/_check_deps.py"
    "/home/thaer/pygmo2/pygmo/test.py"
    "/home/thaer/pygmo2/pygmo/_patch_problem.py"
    "/home/thaer/pygmo2/pygmo/_problem_test.py"
    "/home/thaer/pygmo2/pygmo/_patch_algorithm.py"
    "/home/thaer/pygmo2/pygmo/_patch_bfe.py"
    "/home/thaer/pygmo2/pygmo/_patch_island.py"
    "/home/thaer/pygmo2/pygmo/_py_bfes.py"
    "/home/thaer/pygmo2/pygmo/_mp_utils.py"
    "/home/thaer/pygmo2/pygmo/_py_islands.py"
    "/home/thaer/pygmo2/pygmo/_py_algorithms.py"
    "/home/thaer/pygmo2/pygmo/_algorithm_test.py"
    "/home/thaer/pygmo2/pygmo/_bfe_test.py"
    "/home/thaer/pygmo2/pygmo/_island_test.py"
    "/home/thaer/pygmo2/pygmo/_patch_r_policy.py"
    "/home/thaer/pygmo2/pygmo/_patch_s_policy.py"
    "/home/thaer/pygmo2/pygmo/_patch_topology.py"
    "/home/thaer/pygmo2/pygmo/_s_policy_test.py"
    "/home/thaer/pygmo2/pygmo/_r_policy_test.py"
    "/home/thaer/pygmo2/pygmo/_topology_test.py"
    "/home/thaer/pygmo2/pygmo/_py_problems.py"
    "/home/thaer/pygmo2/pygmo/_ipyparallel_utils.py"
    "/home/thaer/pygmo2/build/pygmo/_version.py"
    )
endif()

