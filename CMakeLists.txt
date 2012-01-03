
SET(src init.c)

SET(luasrc init.lua lushio.lua)

ADD_TORCH_PACKAGE(kex "${src}"  "${luasrc}" "Extra Tools")

TARGET_LINK_LIBRARIES(kex luaT TH)
