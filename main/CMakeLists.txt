project(test.exe LANGUAGES CXX)

set(SOURCE_EXE  main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_EXE})

target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
install(TARGETS test.exe DESTINATION ${BIN_PATH})
