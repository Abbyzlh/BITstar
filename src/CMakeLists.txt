
#message(${PROJECT_SOURCE_DIR})  ##src
#message(${PROJECT_NAME})
#message(${CMAKE_BINARY_DIR})
#message(STATUS ${PROJECT_VERSION})

add_executable(${PROJECT_NAME} main.cpp)

add_library(BatchInformedTree SHARED BatchInformedTree.cpp)
add_library(WorkingSpace STATIC WorkingSpace.cpp)

### Bullet
INCLUDE_DIRECTORIES(
        ${BULLET_PHYSICS_SOURCE_DIR}/src
)
LINK_LIBRARIES(
        BulletDynamics BulletCollision LinearMath
)
find_package(Bullet REQUIRED)
include_directories(${BULLET_INCLUDE_DIR})
### end Bullet