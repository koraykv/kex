
SET(src init.c)

SET(luasrc init.lua 
  lushio.lua
  CriterionModule.lua
  SpatialFullConvolution.lua 
  WeightedMSECriterion.lua
  L1Cost.lua
  test/test.lua
  test/test_module.lua
  )

ADD_TORCH_PACKAGE(kex "${src}"  "${luasrc}" "Extra Tools")

TARGET_LINK_LIBRARIES(kex luaT TH)
