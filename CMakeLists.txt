
SET(src)

SET(luasrc init.lua lushio.lua)

ADD_TORCH_PACKAGE(kktools "${src}"  "${luasrc}" "Tools")

##TARGET_LINK_LIBRARIES(lushio luaT TH)
