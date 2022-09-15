cmake_minimum_required(VERSION 3.11.0)
project(pydpomdp VERSION 0.1.0)

find_package(pybind11 REQUIRED)

include(FetchContent)
FetchContent_Declare(project_madp
    GIT_REPOSITORY	   https://github.com/MADPToolbox/MADP
    GIT_TAG			   a6c1bb7ca18236fc7214d05c6c08dd7d515c4e96
)
FetchContent_MakeAvailable(project_madp)
FetchContent_GetProperties(project_madp SOURCE_DIR MADP_SOURCE_DIR)

set(MADP_INCLUDE_DIRECTORY ${MADP_SOURCE_DIR}/src/include)
set(MADP_BASE_DIRECTORY ${MADP_SOURCE_DIR}/src/base)
set(MADP_PARSER_DIRECTORY ${MADP_SOURCE_DIR}/src/parser)

include_directories(include
                    ${PYBIND11_INCLUDE_DIRS}
                    SYSTEM ${MADP_INCLUDE_DIRECTORY}
					SYSTEM ${MADP_BASE_DIRECTORY}
                    SYSTEM ${MADP_PARSER_DIRECTORY}
					)

add_library(MADPBase 
                ${MADP_BASE_DIRECTORY}/DecPOMDP.cpp
                ${MADP_BASE_DIRECTORY}/DecPOMDPDiscrete.cpp
                ${MADP_BASE_DIRECTORY}/EventObservationModelMapping.cpp
                ${MADP_BASE_DIRECTORY}/EventObservationModelMappingSparse.cpp
                ${MADP_BASE_DIRECTORY}/Globals.cpp
                ${MADP_BASE_DIRECTORY}/IndexTools.cpp
                ${MADP_BASE_DIRECTORY}/JointActionDiscrete.cpp
                ${MADP_BASE_DIRECTORY}/JointObservationDiscrete.cpp
                ${MADP_BASE_DIRECTORY}/MADPComponentDiscreteActions.cpp
                ${MADP_BASE_DIRECTORY}/MADPComponentDiscreteObservations.cpp
                ${MADP_BASE_DIRECTORY}/MADPComponentDiscreteStates.cpp
                ${MADP_BASE_DIRECTORY}/MultiAgentDecisionProcess.cpp
                ${MADP_BASE_DIRECTORY}/MultiAgentDecisionProcessDiscrete.cpp
                ${MADP_BASE_DIRECTORY}/NamedDescribedEntity.cpp
                ${MADP_BASE_DIRECTORY}/ObservationModelDiscrete.cpp
                ${MADP_BASE_DIRECTORY}/ObservationModelMapping.cpp
                ${MADP_BASE_DIRECTORY}/ObservationModelMappingSparse.cpp
                ${MADP_BASE_DIRECTORY}/RewardModelMapping.cpp
                ${MADP_BASE_DIRECTORY}/RewardModelMappingSparse.cpp
                ${MADP_BASE_DIRECTORY}/Scope.cpp
                ${MADP_BASE_DIRECTORY}/TransitionModelDiscrete.cpp
                ${MADP_BASE_DIRECTORY}/TransitionModelMapping.cpp
                ${MADP_BASE_DIRECTORY}/TransitionModelMappingSparse.cpp  
                )
set_property(TARGET MADPBase PROPERTY POSITION_INDEPENDENT_CODE ON)
            
add_library(MADPParser
                ${MADP_PARSER_DIRECTORY}/ParserDPOMDPFormat_Spirit.cpp)
set_property(TARGET MADPParser PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(MADPParser MADPBase)

pybind11_add_module(${PROJECT_NAME} src/main.cpp src/madp_wrapper.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE MADPBase MADPParser)