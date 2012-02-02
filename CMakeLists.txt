
SET(src init.c)

SET(luasrc init.lua 
  lushio.lua
  CriterionModule.lua
  SpatialFullConvolution.lua 
  WeightedMSECriterion.lua
  L1Cost.lua
  Diag.lua
  SpatialMaxPyramid.lua
  SpatialMaxPooling2.lua
  Vectorize.lua
  TanhShrink.lua
  Crop.lua
  SqrtBias.lua
  stochasticrates.lua
  test/test.lua
  test/test_module.lua
  )

ADD_TORCH_PACKAGE(kex "${src}"  "${luasrc}" "Extra Tools")
ADD_TORCH_DOK(dok kex "3rd Party" "Extensions from Koray" 111111.)

TARGET_LINK_LIBRARIES(kex luaT TH)
